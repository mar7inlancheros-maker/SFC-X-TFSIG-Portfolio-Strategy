"""Orquestador: de la configuracion a un unico resultado con todo el analisis.

Separado del reporte a proposito. `run_analysis` calcula; `reporting/` solo
presenta. Asi el mismo resultado sirve a la terminal, al JSON, al HTML y a los
tests, y ningun numero se recalcula de forma distinta en dos sitios.

**Ventanas temporales:**

    data_start = eval_start - 1 ano   (calentamiento: la primera estimacion de
                                       covarianza necesita 252 sesiones previas)
    eval_start = hoy - lookback       (todo lo que se reporta sale de aqui)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import engine as bt
from .backtest import metrics as btm
from .data import cleaning, loader, validation, wrds_data
from .factors import beta as beta_mod
from .factors import (
    factor_model,
    liquidity,
    mean_reversion,
    momentum,
    performance,
    volatility,
)
from .inference import correlation, covariance, tests
from .portfolio import analytics as pa
from .portfolio import construction, neutral
from .portfolio.constraints import ConstructionParams
from .risk import montecarlo, stress
from .settings import EngineSettings
from .signals import agreement, composite, score_validation

log = logging.getLogger(__name__)
Progress = Callable[[str], None]


@dataclass
class AnalysisResult:
    settings: EngineSettings
    as_of: pd.Timestamp
    eval_start: pd.Timestamp
    quality: pd.DataFrame
    excluded: list[str]
    longs: list[str]
    shorts: list[str]
    clean: cleaning.CleanData
    sectors: dict[str, str]
    sector_source: dict[str, str]
    features: pd.DataFrame
    trailing: pd.DataFrame
    risk: pd.DataFrame
    regression: pd.DataFrame
    rolling_betas: pd.DataFrame
    momentum: pd.DataFrame
    mean_reversion: pd.DataFrame
    volatility: pd.DataFrame
    liquidity: pd.DataFrame
    correlation: dict[str, object]
    covariances: dict[str, dict[str, object]]
    factors: factor_model.FactorModelResult
    scores: pd.DataFrame
    feature_matrix: pd.DataFrame
    agreement: pd.DataFrame
    rank_stability: dict[str, float]
    long_vs_short: pd.DataFrame
    spread: dict[str, float]
    portfolios: dict[str, dict[str, object]]
    beta_comparison: pd.DataFrame
    sector_neutral: dict[str, object]
    primary: dict[str, object]
    historical: dict[str, object] | None
    stress: dict[str, object]
    montecarlo: dict[str, object]
    robustness: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    wrds: wrds_data.WrdsData | None = None
    fundamentals: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_plan: dict[str, object] = field(default_factory=dict)
    reference_peers: list[str] = field(default_factory=list)
    score_validation: dict[str, object] = field(default_factory=dict)
    earnings: dict[str, pd.Timestamp] = field(default_factory=dict)
    ex_dividend: dict[str, pd.Timestamp] = field(default_factory=dict)
    valuation: dict[str, dict] = field(default_factory=dict)


def _params(settings: EngineSettings, beta_neutral: bool = False) -> ConstructionParams:
    gross, net = settings.exposure
    return ConstructionParams(
        gross=gross, net=net,
        max_position=float(settings.get("portfolio.max_position", 0.2)),
        beta_neutral=beta_neutral,
    )


def reference_peers(settings: EngineSettings, n_names: int | None = None) -> list[str]:
    """Nombres contra los que se puntua una canasta solo LONG o muy pequena.

    El Quant Score son z-scores de seccion cruzada. Dentro de una canasta de
    dos nombres no hay seccion cruzada (con menos de tres, el z-score sale
    vacio), y dentro de una canasta solo LONG la mitad queda "en contra" por
    construccion, porque los z-scores suman cero. Con el universo de
    referencia la pregunta pasa a ser "como esta este nombre frente a acciones
    grandes y liquidas de todos los sectores". Los pares solo puntuan: nunca
    entran en la cartera, el plan ni las tablas.
    """
    min_cs = int(settings.get("scoring.min_cross_section", 5))
    # `n_names`: los nombres que quedan tras la exclusion por calidad. Con
    # LONG ARES,AMC y SHORT SKHY,ATZ.TO,GOLD se pidieron 5, SKHY no tenia
    # historia y se puntuaron 4 entre si: justo lo que la regla debia evitar.
    n = len(settings.tickers) if n_names is None else n_names
    if not settings.long_only and n >= min_cs:
        return []
    return reference_list(settings)


def reference_list(settings: EngineSettings) -> list[str]:
    """`scoring.reference_peers` sin los nombres de la canasta ni el benchmark."""
    basket = set(settings.tickers) | {settings.benchmark}
    peers = [str(t).upper() for t in (settings.get("scoring.reference_peers", []) or [])]
    return [t for t in dict.fromkeys(peers) if t not in basket]


VALUE_QUALITY_COLUMNS = ("earnings_yield", "fcf_yield", "book_to_price", "sales_to_price", "roe",
                         "gross_profitability", "fcf_margin", "debt_to_equity", "accruals")


def slice_clean(clean: cleaning.CleanData, d: pd.Timestamp) -> cleaning.CleanData:
    """Los datos tal como estaban al cierre de `d`: nada posterior."""
    cut = lambda f: f.loc[f.index <= d]  # noqa: E731
    return cleaning.CleanData(close=cut(clean.close), high=cut(clean.high), low=cut(clean.low),
                              volume=cut(clean.volume), returns=cut(clean.returns), benchmark=clean.benchmark,
                              stats=clean.stats, fx=cut(clean.fx), currency=clean.currency)


def features_at(settings: EngineSettings, clean: cleaning.CleanData, names: list[str], d: pd.Timestamp,
                wrds: wrds_data.WrdsData | None, fx_rates: dict[str, pd.Series]) -> pd.DataFrame:
    """Caracteristicas del Quant Score con los datos que habia el dia `d`.

    Las mismas funciones que el analisis de hoy (`_per_asset`, `_features`,
    `ttm_panel`): validar otro score distinto del que decide no serviria. Solo
    entran los nombres con la historia minima que exige la calidad de datos.
    """
    sliced = slice_clean(clean, d)
    alive = [t for t in names if sliced.close[t].notna().sum() >= validation.MIN_SESSIONS]
    if len(alive) < 3:
        return pd.DataFrame()
    ev = max(d - pd.DateOffset(years=settings.lookback_years), sliced.close.index[0])
    parts = _per_asset(settings, sliced, alive, ev)
    r_ev = sliced.returns.loc[sliced.returns.index >= ev, alive]
    f = _features(parts, r_ev.corr(), alive)
    if wrds is None or not wrds.available or wrds.ids.empty:
        return f
    ids = wrds.ids.loc[[t for t in alive if t in wrds.ids.index]]
    concepts = wrds_data.ttm_panel(wrds.fundq, ids, d) if not wrds.fundq.empty else pd.DataFrame()
    if not concepts.empty:
        usd = {c: float(r.loc[:d].iloc[-1]) for c, r in fx_rates.items() if len(r.loc[:d])}
        prices = sliced.close[list(concepts.index)].ffill().iloc[-1]
        reporting = concepts["currency"] if "currency" in concepts else pd.Series(dtype=object)
        fm = wrds_data.fundamental_metrics(concepts, wrds_data.price_in_currency(prices, reporting, usd))
        for col in VALUE_QUALITY_COLUMNS:
            if col in fm:
                f[col] = pd.to_numeric(fm[col], errors="coerce")
    analysts = wrds_data.analyst_panel(wrds.ibes_recs, wrds.ibes_eps, ids, d)
    for col in ("eps_revision_3m", "rec_change_3m", "meanrec"):
        if col in analysts:
            f[col] = pd.to_numeric(analysts[col], errors="coerce")
    # Interes corto con el retraso de publicacion: el de la fecha de
    # liquidacion de ayer no se conocia hasta casi dos semanas despues.
    si = wrds_data.short_interest_at(wrds.short_interest_raw, ids, d, wrds_data.SHORT_INTEREST_LAG_DAYS)
    if not si.empty and not concepts.empty and "shares_mm" in concepts:
        shares = pd.to_numeric(concepts["shares_mm"], errors="coerce") * 1e6
        f["si_pct_float"] = pd.to_numeric(si["shortint"], errors="coerce") / shares.reindex(si.index)
    return f


def fx_plan(tickers) -> tuple[dict[str, str], dict[str, float], list[str]]:
    """Moneda y escala de precio de cada ticker, y los de moneda desconocida."""
    currency, scale, unknown = {}, {}, []
    for t in dict.fromkeys(tickers):
        ccy, k = loader.quote_currency(t)
        if ccy is None:
            unknown.append(t)
        else:
            currency[t], scale[t] = ccy, k
    return currency, scale, unknown


def fx_by_ticker(currency: dict[str, str], scale: dict[str, float],
                 rates: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """USD por unidad de precio local de cada ticker que no cotiza en USD."""
    return {t: rates[c] * scale.get(t, 1.0) for t, c in currency.items()
            if c != loader.BASE_CURRENCY and c in rates}


def _load_fx(tickers, data_start, as_of, warnings: list[str],
             warn_for: set[str]) -> tuple[dict[str, pd.Series], dict[str, str], dict[str, pd.Series]]:
    """Tipos de cambio para convertir a USD, avisando de lo que no se puede
    (solo de los nombres de `warn_for`: los pares no merecen un aviso)."""
    currency, scale, unknown = fx_plan(tickers)
    rates, failed = loader.load_fx(set(currency.values()), data_start, as_of)
    unknown = [t for t in unknown if t in warn_for]
    if unknown:
        warnings.append(f"moneda desconocida para {', '.join(unknown)} (sufijo de bolsa fuera de "
                        "loader.QUOTE_CURRENCY): se analiza(n) sin convertir a USD; anade el sufijo a la tabla")
    for c, why in failed.items():
        names = [t for t, ccy in currency.items() if ccy == c and t in warn_for]
        if names:
            warnings.append(f"sin tipo de cambio {c}/USD ({why}): {', '.join(names)} sin convertir a USD; "
                            "beta, liquidez y tamano de la orden salen en moneda local")
    return fx_by_ticker(currency, scale, rates), currency, rates


def _restrict(parts: dict[str, pd.DataFrame], keep: list[str]) -> dict[str, pd.DataFrame]:
    """Las tablas por activo, sin los pares de referencia."""
    out = {}
    for name, frame in parts.items():
        if name == "rolling_betas":
            out[name] = frame[[c for c in frame.columns if c in keep]]
        else:
            out[name] = frame.loc[[t for t in frame.index if t in keep]]
    return out


def backtest_config(settings: EngineSettings, **overrides) -> bt.BacktestConfig:
    values = dict(
        method=settings.construction,
        rebalance=settings.rebalance,
        params=_params(settings),
        capital=settings.initial_capital,
        transaction_cost=settings.transaction_cost,
        borrow_cost=settings.borrow_cost,
        risk_free_rate=settings.risk_free_rate,
        estimation_window_d=int(settings.get("portfolio.estimation_window_d", 252)),
        ewma_lambda=float(settings.get("portfolio.ewma_lambda", 0.94)),
    )
    values.update(overrides)
    return bt.BacktestConfig(**values)


def _per_asset(settings: EngineSettings, clean: cleaning.CleanData, tickers: list[str],
               eval_start: pd.Timestamp) -> dict[str, pd.DataFrame]:
    rf = settings.risk_free_rate
    bench = clean.benchmark
    r_eval = clean.returns.loc[clean.returns.index >= eval_start]
    lam = float(settings.get("portfolio.ewma_lambda", 0.94))
    vol_thresholds = settings.get("volatility_regime", {}) or {}
    regime_cfg = settings.get("regime", {}) or {}

    trailing, risk, reg, mom, mr, vol, liq, rolling = {}, {}, {}, {}, {}, {}, {}, {}
    for t in tickers + [bench]:
        close = clean.close[t].dropna()
        r = r_eval[t].dropna()
        trailing[t] = performance.trailing_returns(close)
        regression = beta_mod.regression(r_eval[t], r_eval[bench], rf) if t != bench else {"beta": 1.0}
        risk[t] = {**performance.risk_metrics(r, rf),
                   **performance.risk_adjusted(r, rf, beta=regression.get("beta"),
                                               benchmark=r_eval[bench] if t != bench else None)}
        risk[t]["sharpe_1y"] = performance.risk_adjusted(r.iloc[-252:], rf).get("sharpe", np.nan)
        if t == bench:
            continue
        reg[t] = {**regression, **beta_mod.beta_stability(clean.returns[t], clean.returns[bench])}
        rolling[t] = beta_mod.rolling_beta(clean.returns[t], clean.returns[bench], 252)
        high, low = clean.high.get(t), clean.low.get(t)
        mom[t] = {**momentum.momentum_features(close, high, low),
                  "trend_quality": momentum.trend_quality(close),
                  "pct_52w_high": momentum.high_52w_ratio(close, high),
                  "resid_mom": momentum.residual_momentum(clean.returns[t], clean.returns[bench])}
        mr_feat = mean_reversion.mean_reversion_features(close, clean.returns[t])
        mr_feat["regime"] = mean_reversion.classify_regime(
            mr_feat, float(regime_cfg.get("trend_threshold", 0.5)),
            float(regime_cfg.get("reversion_threshold", 1.5)))
        mr[t] = mr_feat
        vol[t] = volatility.volatility_features(clean.returns[t], close, high, low,
                                                lam=lam, thresholds=vol_thresholds)
        liq[t] = liquidity.liquidity_features(close, clean.volume.get(t, pd.Series(dtype=float)),
                                              clean.returns[t])
    return {
        "trailing": pd.DataFrame(trailing).T,
        "risk": pd.DataFrame(risk).T,
        "regression": pd.DataFrame(reg).T,
        "rolling_betas": pd.DataFrame(rolling),
        "momentum": pd.DataFrame(mom).T,
        "mean_reversion": pd.DataFrame(mr).T,
        "volatility": pd.DataFrame(vol).T,
        "liquidity": pd.DataFrame(liq).T,
    }


def _features(parts: dict[str, pd.DataFrame], corr: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    f = pd.DataFrame(index=tickers)
    f["mom_12_1"] = parts["momentum"]["mom_12_1"]
    f["mom_6m"] = parts["momentum"]["mom_6m"]
    f["mom_3m"] = parts["momentum"]["mom_3m"]
    f["price_to_sma_200"] = parts["momentum"]["price_to_sma_200"]
    f["macd_hist_pct"] = parts["momentum"]["macd_hist_pct"]
    f["trend_quality"] = parts["momentum"]["trend_quality"]
    f["pct_52w_high"] = parts["momentum"]["pct_52w_high"]
    f["resid_mom"] = parts["momentum"]["resid_mom"]
    f["sharpe_1y"] = parts["risk"]["sharpe_1y"]
    f["vol_annual"] = parts["risk"]["vol_annual"]
    f["max_dd"] = parts["risk"]["max_dd"]
    f["skewness"] = parts["risk"]["skewness"]
    f["kurtosis"] = parts["risk"]["kurtosis"]
    f["zscore_20d"] = parts["mean_reversion"]["zscore_20d"]
    f["beta"] = parts["regression"]["beta"]
    f["log_adv"] = np.log(parts["liquidity"]["adv_usd"].astype(float))
    f["ret_1y"] = parts["trailing"]["1Y"]
    c = corr.loc[tickers, tickers].copy()
    np.fill_diagonal(c.values, np.nan)
    f["avg_corr"] = c.mean(axis=1)
    return f.apply(pd.to_numeric, errors="coerce")


def _portfolio_block(settings, clean, signs, method, params, eval_start, bench_nav) -> dict[str, object]:
    window = clean.returns.loc[clean.returns.index <= clean.returns.index[-1]].iloc[
        -int(settings.get("portfolio.estimation_window_d", 252)):]
    names = list(signs.index)
    cov, cov_name = covariance.for_optimization(window[names].dropna())
    vol = pd.Series(np.sqrt(np.diag(cov.to_numpy()) * 252), index=cov.index)
    betas = pd.Series({t: beta_mod.regression(window[t], window[clean.benchmark], settings.risk_free_rate).get("beta", 1.0)
                       for t in names})
    built = construction.build(method, signs, cov=cov, vol=vol, params=params,
                               mu=construction.shrink_means(window[names]), betas=betas)
    block: dict[str, object] = {"method": method, "status": built.status, "note": built.note,
                                "covariance": cov_name, "weights": built.weights}
    if built.status != "ok":
        return block
    ex = pa.ex_ante(built.weights, cov, betas)
    daily = pa.static_returns(built.weights, clean.returns.loc[clean.returns.index >= eval_start])
    from sfc_tfsig.risk.var import historical_es, historical_var

    ex["var_95_1d"] = historical_var(daily, 0.95)
    ex["cvar_95_1d"] = historical_es(daily, 0.95)
    block["ex_ante"] = ex
    block["betas"] = betas
    block["cov"] = cov

    cfg = backtest_config(settings, method=method, params=params)
    result = bt.run(clean.returns, clean.benchmark, bt.constant_schedule(signs), cfg,
                    start=eval_start, mode="A")
    block["backtest"] = result
    block["summary"] = btm.summary(result, bench_nav)
    return block


def run_analysis(settings: EngineSettings, *, signals_csv: Path | None = None,
                 progress: Progress = lambda _msg: None, run_robustness: bool = True) -> AnalysisResult:
    warnings: list[str] = list(settings.get("_notes", []) or [])
    as_of = pd.Timestamp.today().normalize()
    eval_start = as_of - pd.DateOffset(years=settings.lookback_years)
    data_start = eval_start - pd.DateOffset(years=1, days=10)

    proxies = settings.get("factor_proxies", {}) or {}
    etfs = sorted({e for pair in proxies.values() for e in pair} - {settings.benchmark})
    universe = list(settings.tickers)
    hist_schedule = None
    if signals_csv:
        hist_schedule = bt.csv_schedule(signals_csv)
        universe = sorted(set(universe) | set(hist_schedule.tickers))

    progress("descargando datos")
    market = loader.load_ohlcv([*universe, settings.benchmark, *etfs], data_start, as_of)

    if market.stale:
        warnings.append("Yahoo fallo al actualizar " + ", ".join(
            f"{k} (precios hasta {v:%Y-%m-%d})" for k, v in market.stale.items()) + ": se usa la cache")

    progress("limpiando y validando")
    fx, currency, fx_rates = _load_fx([*universe, settings.benchmark], data_start, as_of, warnings,
                                      warn_for=set(universe) | {settings.benchmark})
    clean = cleaning.align(market, settings.benchmark, fx=fx, currency=currency)
    quality = validation.quality_report(clean, market, universe, data_start)
    excluded = validation.failed(quality)
    longs = [t for t in settings.longs if t not in excluded]
    shorts = [t for t in settings.shorts if t not in excluded]
    if excluded:
        warnings.append(f"excluidos por calidad de datos: {', '.join(excluded)}")
    if not longs:
        raise ValueError("tras excluir datos inservibles no queda ningun LONG")
    if settings.shorts and not shorts:
        raise ValueError("tras excluir datos inservibles no queda ningun SHORT: quita los SHORT para "
                         "correr la canasta como solo LONG")
    tickers = longs + shorts
    # Los pares se deciden con los nombres que SOBREVIVEN a la calidad de datos.
    # Se descargan despues y se realinea: la cache del loader hace que la
    # segunda llamada no vuelva a pedir lo que ya tiene.
    peers = reference_peers(settings, n_names=len(tickers))
    # La validacion del score necesita seccion cruzada: canasta + pares siempre,
    # aunque la canasta sea grande y no puntue contra ellos.
    val_cfg = settings.get("validation", {}) or {}
    validate_score = bool(val_cfg.get("enabled", True))
    extra = list(dict.fromkeys([*peers, *(reference_list(settings) if validate_score else [])]))
    if extra:
        market = loader.load_ohlcv([*universe, *extra, settings.benchmark, *etfs], data_start, as_of)
        missing_extra = [k for k in extra if k in market.failures]
        if len(missing_extra) > len(extra) / 4:
            warnings.append(f"Yahoo no devolvio {len(missing_extra)} de {len(extra)} pares de referencia "
                            f"({', '.join(missing_extra[:8])}...): score y validacion con menos seccion cruzada")
        fx, currency, fx_rates = _load_fx([*universe, *extra, settings.benchmark], data_start, as_of, warnings,
                                          warn_for=set())
        clean = cleaning.align(market, settings.benchmark, fx=fx, currency=currency)
        bad_peers = validation.failed(validation.quality_report(clean, market, extra, data_start))
        extra = [t for t in extra if t not in bad_peers]
        peers = [t for t in peers if t not in bad_peers]
    if peers:
        warnings.append(f"Quant Score contra {len(peers)} pares de referencia (scoring.reference_peers): "
                        "la canasta es solo LONG o tiene menos nombres que scoring.min_cross_section")
    score_names = tickers + peers
    val_names = tickers + [t for t in extra if t not in tickers] if validate_score else []
    signs = pd.Series({**{t: 1.0 for t in longs}, **{t: -1.0 for t in shorts}})

    progress("sectores (SEC)")
    progress("WRDS: Fama-French, GICS, Compustat, IBES, interes corto")
    wrds = wrds_data.load(list(dict.fromkeys(score_names + val_names)), data_start, as_of)
    if not wrds.available:
        warnings.append(f"WRDS no disponible, se usa Yahoo + ETFs + SIC: {wrds.reason}")
    elif [t for t in wrds.unmatched if t in tickers]:
        warnings.append(f"sin identificador en Compustat: {', '.join(t for t in wrds.unmatched if t in tickers)}")

    sectors, sector_source = loader.load_sectors(tickers)
    for t, gics in wrds.sectors.items():
        if t in sectors:
            sectors[t], sector_source[t] = gics, "GICS (Compustat)"

    progress("metricas por activo")
    parts_all = _per_asset(settings, clean, score_names, eval_start)
    parts = _restrict(parts_all, tickers + [clean.benchmark])
    r_eval = clean.returns.loc[clean.returns.index >= eval_start]

    progress("correlacion y covarianza")
    corr_summary = correlation.correlation_summary(r_eval, longs, shorts)
    cov_estimates = covariance.estimate_all(r_eval[tickers], float(settings.get("portfolio.ewma_lambda", 0.94)))

    progress("modelo de factores")
    if wrds.available and not wrds.factors.empty:
        fm = factor_model.fama_french_exposures(r_eval, tickers, wrds.factors)
        last_ff = wrds.factors.index.max()
        if (as_of - last_ff).days > 45:
            warnings.append(f"factores Fama-French hasta {last_ff.date()}: los ultimos dias quedan fuera de la regresion")
    else:
        fm = factor_model.factor_exposures(r_eval, tickers, settings.benchmark, proxies, settings.risk_free_rate)
    for name, reason in fm.unavailable.items():
        warnings.append(f"factor {name} no disponible: {reason}")

    progress("senales")
    corr_all = corr_summary["pearson"] if not peers else r_eval[score_names].corr()
    features = _features(parts_all, corr_all, score_names)
    fundamentals = pd.DataFrame()
    if wrds.available:
        # Valor = beneficio / capitalizacion: los dos en la moneda en que reporta
        # la empresa. Shopify cotiza en CAD en Toronto y reporta en USD.
        last_close = clean.close[score_names].ffill().iloc[-1]
        reporting = (wrds.fundamentals["currency"] if "currency" in wrds.fundamentals
                     else pd.Series(dtype=object))
        missing_ccy = {c for c in reporting.dropna().unique() if c != loader.BASE_CURRENCY and c not in fx_rates}
        if missing_ccy:
            fx_rates = {**fx_rates, **loader.load_fx(missing_ccy, data_start, as_of)[0]}
        usd_per_unit = {c: float(r.loc[:as_of].iloc[-1]) for c, r in fx_rates.items() if len(r.loc[:as_of])}
        fundamentals = wrds_data.fundamental_metrics(
            wrds.fundamentals, wrds_data.price_in_currency(last_close, reporting, usd_per_unit))
        for col in ("earnings_yield", "fcf_yield", "book_to_price", "sales_to_price", "roe",
                    "gross_profitability", "fcf_margin", "debt_to_equity", "accruals"):
            if col in fundamentals:
                features[col] = pd.to_numeric(fundamentals[col], errors="coerce")
        for col in ("eps_revision_3m", "rec_change_3m", "meanrec"):
            if col in wrds.analysts:
                features[col] = pd.to_numeric(wrds.analysts[col], errors="coerce")
        if not wrds.short_interest.empty and not wrds.fundamentals.empty:
            adv_shares = clean.volume[score_names].iloc[-63:].median()
            si = wrds_data.short_interest_metrics(wrds.short_interest.astype({"shortint": float}),
                                                  wrds.fundamentals["shares_mm"].astype(float), adv_shares)
            features["si_pct_float"] = pd.to_numeric(si["si_pct_float"], errors="coerce")
            fundamentals = fundamentals.join(si[["si_pct_float", "days_to_cover", "si_date"]], how="outer")
        fundamentals = fundamentals.join(wrds.analysts, how="outer") if not wrds.analysts.empty else fundamentals
    weights = settings.get("signal_weights", {}) or {}
    # Con pares, el score y el rango se calculan sobre canasta + pares y se
    # muestran solo los de la canasta: el rango sale "de N nombres puntuados".
    scores_all = composite.quant_score(features, weights)
    scores = scores_all.loc[tickers]
    agree = agreement.agreement_table(scores, signs.astype(int).to_dict(), settings.get("agreement", {}) or {})
    stability = composite.rank_stability(features, weights, int(settings.get("robustness.signal_weight_draws", 500)),
                                         int(settings.get("montecarlo.seed", 0)))
    feature_matrix = composite.feature_matrix(features).loc[tickers]
    features = features.loc[tickers]
    if not fundamentals.empty:
        fundamentals = fundamentals.loc[[t for t in fundamentals.index if t in tickers]]

    seed = int(settings.get("montecarlo.seed", 0))
    lvs_rows: dict[str, dict] = {}
    spread: dict[str, float] = {}
    if shorts:
        progress("largos frente a cortos")
        for label, col in (("return_1y", "ret_1y"), ("volatility", "vol_annual"), ("sharpe_1y", "sharpe_1y"),
                           ("momentum_12_1", "mom_12_1"), ("beta", "beta"), ("max_drawdown", "max_dd"),
                           ("skewness", "skewness"), ("kurtosis", "kurtosis"), ("avg_correlation", "avg_corr"),
                           ("log_liquidity", "log_adv")):
            lvs_rows[label] = tests.compare_groups(features.loc[longs, col], features.loc[shorts, col], seed=seed)
        long_basket = r_eval[longs].mean(axis=1)
        short_basket = r_eval[shorts].mean(axis=1)
        spread = tests.spread_test(long_basket, short_basket)

    bench_nav = (1.0 + r_eval[settings.benchmark].fillna(0.0)).cumprod() * settings.initial_capital

    progress("construccion de carteras (5 metodos, backtest walk-forward)")
    params = _params(settings)
    legs = [("LONG", longs, params.long_leg)] + ([("SHORT", shorts, params.short_leg)] if shorts else [])
    for side, names, leg in legs:
        if len(names) * params.max_position <= leg + 1e-9:
            warnings.append(
                f"{side}: {len(names)} nombres x tope {params.max_position:.0%} = pata de {leg:.0%}. "
                "El tope obliga a equiponderar: los metodos de construccion no pueden diferir en esta pata."
            )
    portfolios = {m: _portfolio_block(settings, clean, signs, m, params, eval_start, bench_nav)
                  for m in construction.METHODS}

    ew = construction.equal_weight(signs, params)
    if shorts:
        progress("beta neutral y sector neutral")
        betas_now = portfolios["equal_weight"].get("betas", pd.Series(1.0, index=tickers))
        bn, bn_note = neutral.beta_neutral_legs(ew, betas_now, params)
        beta_cmp = pd.DataFrame({"50/50 dollar neutral": neutral.describe(ew, betas_now),
                                 "beta neutral": neutral.describe(bn, betas_now)}).T
        if bn_note:
            warnings.append(bn_note)
        bn_block = _portfolio_block(settings, clean, signs, "equal_weight", _params(settings, beta_neutral=True),
                                    eval_start, bench_nav)
        sn_w, sn_note, one_sided = neutral.sector_neutral(signs, sectors, params)
        sector_block = {
            "before": neutral.sector_exposure(ew, sectors),
            "after": neutral.sector_exposure(sn_w, sectors) if sn_w is not None else None,
            "weights": sn_w, "note": sn_note, "one_sided": one_sided,
            "beta_neutral_backtest": bn_block,
        }
    else:
        # Neutralizar beta o sector exige una pata corta que compense.
        beta_cmp = pd.DataFrame()
        sector_block = {"before": neutral.sector_exposure(ew, sectors), "after": None, "weights": None,
                        "note": "", "one_sided": [], "beta_neutral_backtest": {}}

    primary = portfolios[settings.construction]
    if primary.get("status") != "ok":
        warnings.append(f"el metodo elegido ({settings.construction}) fallo: {primary.get('note')}; "
                        "se usa equal_weight para estres y Monte Carlo")
        primary = portfolios["equal_weight"]

    historical = None
    if hist_schedule is not None:
        progress("backtest historico de senales (modo B)")
        start = max(eval_start, pd.Timestamp(hist_schedule.first_date))
        res = bt.run(clean.returns, clean.benchmark, hist_schedule, backtest_config(settings), start=start, mode="B")
        historical = {"backtest": res, "summary": btm.summary(res, bench_nav)}

    progress("estres")
    w = primary["weights"]
    # Beta de cartera en el peor caso razonable: cada largo con el percentil 95
    # de su beta movil observada, cada corto con el percentil 5. Un corto con
    # beta baja cubre MENOS, asi que su peor caso es la beta baja, no la alta.
    rb = parts["rolling_betas"][tickers]
    worst_each = pd.Series({t: rb[t].quantile(0.95) if w.get(t, 0) > 0 else rb[t].quantile(0.05)
                            for t in w.index})
    worst_beta = float((worst_each * w).sum())
    stress_block = {
        "historical": stress.historical_scenarios(w, r_eval, settings.benchmark),
        "covariance": stress.covariance_shocks(w, primary["cov"], signs,
                                               list(settings.get("stress.volatility_multipliers", [1.5, 2.0]))),
        "market": stress.market_shocks(float(primary["ex_ante"]["beta"]), worst_beta,
                                       list(settings.get("stress.market_shocks", [-0.1, -0.2]))),
    }
    liq_table, exit_cost = stress.liquidity_stress(
        w, settings.initial_capital, parts["liquidity"]["adv_usd"].astype(float),
        float(settings.get("stress.liquidity_participation", 0.1)), settings.transaction_cost,
        float(settings.get("stress.liquidity_spread_multiplier", 3.0)))
    stress_block["liquidity"], stress_block["exit_cost"] = liq_table, exit_cost

    progress("Monte Carlo")
    mc = settings.get("montecarlo", {}) or {}
    n_paths, horizon = int(mc.get("n_paths", 10000)), int(mc.get("horizon_d", 252))
    thresholds = list(mc.get("drawdown_thresholds", [0.1, 0.2, 0.3]))
    # Deriva NEUTRA por defecto. Los retornos del backtest del modo A llevan
    # dentro la seleccion hecha con informacion de hoy; remuestrearlos tal cual
    # proyecta esa retrospectiva hacia el futuro (la primera version prometia
    # +37% de riqueza esperada a un ano). Se retira la media historica y se
    # centra en la tasa libre de riesgo: la simulacion mide RIESGO -- dispersion
    # y drawdowns --, no un retorno esperado.
    bt_returns = primary["backtest"].returns
    # Deriva neutra de la CARTERA: la caja rinde rf y los cortos pagan prestamo.
    short_exposure = float(-w[w < 0].sum())
    neutral_drift = cleaning.daily_rf(settings.risk_free_rate) - cleaning.daily_rf(settings.borrow_cost) * short_exposure
    neutral_returns = bt_returns - bt_returns.mean() + neutral_drift
    block = int(mc.get("block_d", 10))
    boot = montecarlo.stationary_bootstrap(neutral_returns, n_paths, horizon, block, seed)
    param = montecarlo.parametric_normal(w, pd.Series(0.0, index=w.index), primary["cov"],
                                         n_paths, horizon, seed, drift=neutral_drift)
    boot_hist = montecarlo.stationary_bootstrap(bt_returns, n_paths, horizon, block, seed)
    mc_block = {"bootstrap_neutral": montecarlo.summarize(boot, thresholds),
                "parametric_neutral": montecarlo.summarize(param, thresholds),
                "bootstrap_historical_drift": montecarlo.summarize(boot_hist, thresholds),
                "bootstrap_terminal": montecarlo.terminal_distribution(boot)}

    result = AnalysisResult(
        settings=settings, as_of=as_of, eval_start=eval_start, quality=quality, excluded=excluded,
        longs=longs, shorts=shorts, clean=clean, sectors=sectors, sector_source=sector_source,
        features=features, trailing=parts["trailing"], risk=parts["risk"], regression=parts["regression"],
        rolling_betas=parts["rolling_betas"], momentum=parts["momentum"], mean_reversion=parts["mean_reversion"],
        volatility=parts["volatility"], liquidity=parts["liquidity"], correlation=corr_summary,
        covariances=cov_estimates, factors=fm, scores=scores, feature_matrix=feature_matrix,
        wrds=wrds, fundamentals=fundamentals,
        agreement=agree, rank_stability=stability, long_vs_short=pd.DataFrame(lvs_rows).T, spread=spread,
        portfolios=portfolios, beta_comparison=beta_cmp, sector_neutral=sector_block, primary=primary,
        historical=historical, stress=stress_block, montecarlo=mc_block, warnings=warnings,
        reference_peers=peers,
    )
    for block in portfolios.values():
        if "backtest" in block:
            warnings.extend(f"[{block['method']}] {w}" for w in block["backtest"].warnings[:3])

    if validate_score:
        months = int(val_cfg.get("months", 36))
        horizon_v = int(val_cfg.get("horizon_d", 21))
        progress(f"validacion del Quant Score ({months} meses, {len(val_names)} nombres)")
        # Las caracteristicas de cada fecha se calculan una vez: el segundo
        # horizonte solo vuelve a puntuar.
        memo: dict[pd.Timestamp, pd.DataFrame] = {}

        def feats(d: pd.Timestamp) -> pd.DataFrame:
            if d not in memo:
                memo[d] = features_at(settings, clean, val_names, d, wrds, fx_rates)
            return memo[d]

        def run_horizon(h: int) -> dict[str, object]:
            last = clean.calendar[-1 - h] if len(clean.calendar) > h else clean.calendar[0]
            dates = score_validation.month_end_dates(clean.calendar, clean.calendar[0] + pd.DateOffset(years=1),
                                                     last, months)
            fwd = score_validation.forward_returns(clean.close[val_names], dates, h)
            return score_validation.validate(feats, fwd, weights, min_names=int(val_cfg.get("min_names", 20)),
                                             horizon=h)

        result.score_validation = run_horizon(horizon_v)
        sv = result.score_validation
        # El plan mantiene hasta `horizon_d` sesiones: se mide tambien ahi. Las
        # ventanas se solapan y la t lleva Newey-West; decide el horizonte corto.
        plan_h = int(val_cfg.get("plan_horizon_d", 63))
        if sv.get("available") and plan_h and plan_h != horizon_v:
            sv_long = run_horizon(plan_h)
            sv["plan_horizon"] = {k: sv_long.get(k) for k in ("available", "reason", "months", "horizon_d",
                                                              "ic_mean", "ic_t", "ic_hit", "spread_mean",
                                                              "spread_t")}
        if not sv.get("available"):
            warnings.append(f"validacion del Quant Score no disponible: {sv.get('reason')}")
        elif score_validation.evidence(sv, 2.0) == "inverse":
            warnings.append(f"el Quant Score ordeno AL REVES en {sv['months']} meses (IC {sv['ic_mean']:+.3f}, "
                            f"t {sv['ic_t']:+.1f}): sus veredictos no son fiables en este universo")

    progress("fechas de resultados (Yahoo)")
    result.earnings = loader.load_earnings_dates(tickers, as_of)
    if shorts:
        result.ex_dividend = loader.load_ex_dividend_dates(shorts + [t for t in longs if t not in shorts], as_of)

    progress("valoracion DCF (SEC / Compustat)")
    from .valuation.engine import run_valuations  # noqa: PLC0415

    # Tambien los excluidos con algo de historia (salidas a bolsa recientes): su
    # ficha limitada lleva el DCF aunque no tengan score.
    recent = [t for t in excluded if t in clean.close and clean.close[t].notna().sum() >= 30]
    result.valuation = run_valuations(result, wrds, fx_rates, as_of, tickers + recent)
    for t, v in result.valuation.items():
        if not v.get("available"):
            warnings.append(f"DCF de {t} no disponible: {v.get('reason')}")

    progress("plan de operacion")
    from .trade_plan import build_plan, params_from_settings  # noqa: PLC0415

    result.trade_plan = build_plan(result, params_from_settings(settings))
    if result.trade_plan.get("note"):
        warnings.append(result.trade_plan["note"])

    if run_robustness:
        from .robustness import run_robustness as rr

        progress("robustez")
        result.robustness = rr(result)
    return result
