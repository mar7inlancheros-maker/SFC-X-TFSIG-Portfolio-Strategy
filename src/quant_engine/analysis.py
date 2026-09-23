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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import engine as bt
from .backtest import metrics as btm
from .data import cleaning, loader, validation
from .factors import beta as beta_mod
from .factors import factor_model, liquidity, mean_reversion, momentum, performance, volatility
from .inference import correlation, covariance, tests
from .portfolio import analytics as pa
from .portfolio import construction, neutral
from .portfolio.constraints import ConstructionParams
from .risk import montecarlo, stress
from .settings import EngineSettings
from .signals import agreement, composite

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


def _params(settings: EngineSettings, beta_neutral: bool = False) -> ConstructionParams:
    return ConstructionParams(
        gross=float(settings.get("portfolio.gross_exposure", 2.0)),
        net=float(settings.get("portfolio.net_exposure", 0.0)),
        max_position=float(settings.get("portfolio.max_position", 0.2)),
        beta_neutral=beta_neutral,
    )


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
                  "trend_quality": momentum.trend_quality(close)}
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
    warnings: list[str] = []
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

    progress("limpiando y validando")
    clean = cleaning.align(market, settings.benchmark)
    quality = validation.quality_report(clean, market, universe, data_start)
    excluded = validation.failed(quality)
    longs = [t for t in settings.longs if t not in excluded]
    shorts = [t for t in settings.shorts if t not in excluded]
    if excluded:
        warnings.append(f"excluidos por calidad de datos: {', '.join(excluded)}")
    if not longs or not shorts:
        raise ValueError("tras excluir datos inservibles no queda al menos un LONG y un SHORT")
    tickers = longs + shorts
    signs = pd.Series({**{t: 1.0 for t in longs}, **{t: -1.0 for t in shorts}})

    progress("sectores (SEC)")
    sectors, sector_source = loader.load_sectors(tickers)

    progress("metricas por activo")
    parts = _per_asset(settings, clean, tickers, eval_start)
    r_eval = clean.returns.loc[clean.returns.index >= eval_start]

    progress("correlacion y covarianza")
    corr_summary = correlation.correlation_summary(r_eval, longs, shorts)
    cov_estimates = covariance.estimate_all(r_eval[tickers], float(settings.get("portfolio.ewma_lambda", 0.94)))

    progress("modelo de factores")
    fm = factor_model.factor_exposures(r_eval, tickers, settings.benchmark, proxies, settings.risk_free_rate)
    for name, reason in fm.unavailable.items():
        warnings.append(f"factor {name} no disponible: {reason}")

    progress("senales")
    features = _features(parts, corr_summary["pearson"], tickers)
    weights = settings.get("signal_weights", {}) or {}
    scores = composite.quant_score(features, weights)
    agree = agreement.agreement_table(scores, signs.astype(int).to_dict(), settings.get("agreement", {}) or {})
    stability = composite.rank_stability(features, weights, int(settings.get("robustness.signal_weight_draws", 500)),
                                         int(settings.get("montecarlo.seed", 0)))

    progress("largos frente a cortos")
    seed = int(settings.get("montecarlo.seed", 0))
    lvs_rows = {}
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
    for side, names, leg in (("LONG", longs, params.long_leg), ("SHORT", shorts, params.short_leg)):
        if len(names) * params.max_position <= leg + 1e-9:
            warnings.append(
                f"{side}: {len(names)} nombres x tope {params.max_position:.0%} = pata de {leg:.0%}. "
                "El tope obliga a equiponderar: los metodos de construccion no pueden diferir en esta pata."
            )
    portfolios = {m: _portfolio_block(settings, clean, signs, m, params, eval_start, bench_nav)
                  for m in construction.METHODS}

    progress("beta neutral y sector neutral")
    ew = construction.equal_weight(signs, params)
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
    port_daily = pa.static_returns(w, r_eval)
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
        covariances=cov_estimates, factors=fm, scores=scores, feature_matrix=composite.feature_matrix(features),
        agreement=agree, rank_stability=stability, long_vs_short=pd.DataFrame(lvs_rows).T, spread=spread,
        portfolios=portfolios, beta_comparison=beta_cmp, sector_neutral=sector_block, primary=primary,
        historical=historical, stress=stress_block, montecarlo=mc_block, warnings=warnings,
    )
    for block in portfolios.values():
        if "backtest" in block:
            warnings.extend(f"[{block['method']}] {w}" for w in block["backtest"].warnings[:3])

    if run_robustness:
        from .robustness import run_robustness as rr

        progress("robustez")
        result.robustness = rr(result)
    return result
