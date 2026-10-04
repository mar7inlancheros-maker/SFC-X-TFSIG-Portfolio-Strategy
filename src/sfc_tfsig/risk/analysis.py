"""Junta todas las medidas de riesgo sobre una cartera concreta. Funciones puras.

Dos niveles, porque hay dos momentos en que se mira el riesgo:

- **`analyze_book`**: la cartera y los precios. Es lo que se puede calcular
  ANTES de operar, sobre la cartera objetivo de `ordenes`: riesgo ex-ante,
  modelo factorial, concentracion, liquidez y su coste en crisis, Monte Carlo
  parametrico, GARCH filtrado y VaR estresado de la cartera, estres
  hipotetico y de covarianza, y limites.
- **`analyze`**: lo anterior mas todo lo que necesita la historia de la
  estrategia -- VaR por cinco metodos y su backtest, bootstrap a varios anos,
  estres historico, riesgo movil, calibracion del modelo de riesgo,
  capacidad y drawdown actual. Es el reporte de riesgo del comite.

Nada toca red ni disco: entran DataFrames ya cargados, sale un `RiskAnalysis`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config, ConfigError
from ..metrics import TRADING_DAYS, drawdown_series
from ..universe import SECTORS
from . import (
    exposure,
    factor_model,
    garch,
    history,
    limits,
    liquidity,
    montecarlo,
    stress,
    var,
)

WORST_WINDOWS_D = (1, 5, 21, 63)
VOL_TERM_HORIZONS_D = (1, 5, 21, 63, 252)


@dataclass
class RiskAnalysis:
    as_of: pd.Timestamp
    nav: float
    weights: pd.Series
    sectors: pd.Series
    benchmark: str
    risk_fingerprint: str
    strategy_fingerprint: str
    concentration: dict
    ex_ante: exposure.ExAnteRisk
    factor_exposure: pd.DataFrame
    liquidity: pd.DataFrame
    liquidity_summary: dict
    book_var: pd.DataFrame
    component_es: pd.DataFrame
    hypothetical: pd.DataFrame
    hypothetical_detail: pd.DataFrame
    reverse: pd.DataFrame
    measures: dict
    limits: pd.DataFrame
    status: str
    warnings: list[str] = field(default_factory=list)
    # -- de la cartera, con historia de precios de sus nombres --------------------
    factor_risk: factor_model.FactorRisk | None = None
    covariance_stress: pd.DataFrame | None = None
    stressed: dict | None = None
    garch_fits: dict = field(default_factory=dict)
    garch_sims: dict = field(default_factory=dict)
    conditional_var: pd.DataFrame | None = None
    vol_terms: pd.DataFrame | None = None
    liquidation: pd.DataFrame | None = None
    liquidation_detail: pd.DataFrame | None = None
    capacity: pd.DataFrame | None = None
    capacity_nav: float = float("nan")
    # -- solo con historia de la estrategia (None en la comprobacion previa) --
    strategy_var: pd.DataFrame | None = None
    var_backtest: pd.DataFrame | None = None
    var_forecasts: pd.DataFrame | None = None
    tail: dict | None = None
    bootstrap: montecarlo.BootstrapResult | None = None
    bootstrap_summary: pd.DataFrame | None = None
    historical: pd.DataFrame | None = None
    historical_detail: pd.DataFrame | None = None
    worst_windows: pd.DataFrame | None = None
    rolling: pd.DataFrame | None = None
    rolling_summary: pd.DataFrame | None = None
    calibration: pd.DataFrame | None = None
    calibration_summary: dict | None = None
    current_drawdown: float = float("nan")


# ---------------------------------------------------------------------------
#  Politica y cartera
# ---------------------------------------------------------------------------


def validate_policy(risk_cfg: Config) -> None:
    """Lo que `config.validate_risk` no puede comprobar sin conocer el modelo.

    Un sector mal escrito en un escenario no casa con ninguna posicion y el
    choque sectorial desaparece sin avisar. Se comprueba antes de calcular.
    """
    limits.validate_limit_keys(risk_cfg)
    for scenario in risk_cfg.get("stress.hypothetical", []):
        unknown = sorted(set(scenario.get("sectors", {})) - set(SECTORS))
        if unknown:
            raise ConfigError(
                f"escenario '{scenario['name']}': sectores desconocidos {unknown}. "
                f"Validos: {list(SECTORS)}"
            )


def book_from_backtest(holdings: pd.DataFrame, close_wide: pd.DataFrame,
                       nav: pd.Series) -> tuple[pd.Series, pd.Series, float, pd.Timestamp]:
    """Cartera actual del backtest: acciones del ultimo rebalanceo, a precio de hoy.

    Devuelve `(pesos, sectores, nav, fecha)`. Los pesos son los DERIVADOS desde
    el rebalanceo, no los objetivo: el riesgo que se corre hoy es el de lo que
    hay, y entre rebalanceos lo que sube pesa mas.
    """
    if holdings.empty:
        raise ValueError("el backtest no dejo posiciones: no hay cartera que medir")
    as_of = pd.Timestamp(nav.index[-1])
    last = holdings[holdings["date"] == holdings["date"].max()].set_index("ticker")
    visible = close_wide.loc[close_wide.index <= as_of].reindex(columns=last.index)
    prices = visible.ffill().iloc[-1].fillna(last["price"].astype(float))
    value = last["shares"].astype(float) * prices
    nav_value = float(nav.iloc[-1])
    weights = (value / nav_value).rename("weight")
    sectors = last["sector"].fillna("Unknown") if "sector" in last else pd.Series("Unknown", index=last.index)
    return weights, sectors, nav_value, as_of


def book_from_target(target: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Cartera objetivo de `portfolio.build_portfolio` -> (pesos, sectores)."""
    frame = target.set_index("ticker")
    sectors = frame["sector"].fillna("Unknown") if "sector" in frame else pd.Series("Unknown", index=frame.index)
    return frame["weight"].astype(float), sectors


# ---------------------------------------------------------------------------
#  Piezas
# ---------------------------------------------------------------------------


def _confidences(risk_cfg: Config) -> list[float]:
    # El 99% entra siempre: es el nivel del limite de VaR.
    return sorted({float(c) for c in risk_cfg.get("var.confidence")} | {0.99})


def _book_historical_var(returns: pd.DataFrame, weights: pd.Series, confidences) -> list[dict]:
    """VaR historico a UN dia de la cartera actual con los retornos del ultimo ano.

    Solo a un dia: un ano da doce meses independientes, y un cuantil del 99%
    mensual sacado de ahi es ruido con decimales.
    """
    pnl = returns.reindex(columns=weights.index).fillna(0.0) @ weights
    return [{"method": "historico_cartera", "horizon_d": 1, "confidence": c,
             "var": var.historical_var(pnl, c), "es": var.historical_es(pnl, c)}
            for c in confidences]


def _garch_block(label: str, returns: pd.Series, risk_cfg: Config, confidences,
                 seed: int) -> tuple[garch.GarchFit, garch.FilteredSimulation, pd.DataFrame]:
    fit = garch.fit_garch(returns)
    sim = garch.simulate_filtered(
        fit, [int(h) for h in risk_cfg.get("var.garch.horizons_d")],
        n_paths=int(risk_cfg.get("var.garch.n_paths")), seed=seed,
    )
    table = garch.simulation_var_table(sim, confidences).assign(series=label)
    return fit, sim, table


def _set_garch(result: RiskAnalysis, label: str, returns: pd.Series, risk_cfg: Config,
               confidences, seed: int) -> None:
    """Ajusta, simula y guarda un bloque GARCH; si no hay historia, avisa."""
    try:
        fit, sim, table = _garch_block(label, returns, risk_cfg, confidences, seed)
    except ValueError as exc:
        result.warnings.append(f"GARCH de la {label}: {exc}")
        return
    result.garch_fits[label] = fit
    result.garch_sims[label] = sim
    others = (result.conditional_var[result.conditional_var["series"] != label]
              if result.conditional_var is not None else pd.DataFrame())
    result.conditional_var = pd.concat([others, table], ignore_index=True)
    terms = garch.vol_term_structure(fit, VOL_TERM_HORIZONS_D)["vol_annual"].rename(label)
    result.vol_terms = (terms.to_frame() if result.vol_terms is None
                        else result.vol_terms.drop(columns=label, errors="ignore").join(terms, how="outer"))


def _set_book_history(result: RiskAnalysis, prices: pd.DataFrame, risk_cfg: Config,
                      confidences, seed: int) -> bool:
    """VaR estresado y GARCH de la cartera de hoy sobre su historia reconstruida.

    False si no se pudo reconstruir (sin precios del benchmark): se avisa y el
    reporte sigue, como hace el estres historico, en vez de tumbar `riesgo`.
    """
    try:
        backcast, proxied = stress.book_backcast(
            result.weights, result.ex_ante.betas, prices, result.benchmark
        )
    except ValueError as exc:
        result.warnings.append(f"historia de la cartera: {exc}")
        return False
    sv_cfg = risk_cfg.section("stress.stressed_var")
    result.stressed = stress.stressed_var(
        backcast, float(sv_cfg["confidence"]), int(sv_cfg["window_d"]), proxied
    ) or None
    if result.stressed is None:
        result.warnings.append("historia de precios insuficiente para el VaR estresado")
    _set_garch(result, "cartera", backcast, risk_cfg, confidences, seed)
    return True


def _measures(result: RiskAnalysis) -> dict[str, float]:
    book_var = result.book_var
    t_rows = book_var[(book_var["method"] == "mc_t_student") & (book_var["confidence"] == 0.99)]
    t_rows = t_rows[t_rows["horizon_d"] == t_rows["horizon_d"].min()] if len(t_rows) else t_rows
    hyp = result.hypothetical
    worst_stress = hyp["portfolio"].min() if not hyp.empty else float("nan")
    liq = result.liquidation
    return {
        "ex_ante_vol": result.ex_ante.volatility,
        "var_99_1d": float(t_rows["var"].iloc[0]) if len(t_rows) else float("nan"),
        "beta": result.ex_ante.beta,
        "tracking_error": result.ex_ante.tracking_error,
        "max_name_risk_share": float(result.ex_ante.contributions["risk_share"].max()),
        "max_sector_risk_share": float(result.ex_ante.sectors["risk_share"].max()),
        "max_liquidation_days": result.liquidity_summary.get("max_days", float("nan")),
        "effective_names": result.concentration.get("effective_names", float("nan")),
        "max_stress_loss": max(0.0, -float(worst_stress)) if np.isfinite(worst_stress) else float("nan"),
        "stressed_var_99_1d": (result.stressed or {}).get("stressed_var", float("nan")),
        "liquidation_cost_stress": (float(liq.loc["crisis", "cost"])
                                    if liq is not None and "crisis" in liq.index else float("nan")),
        "current_drawdown": result.current_drawdown,
    }


def _refresh_limits(result: RiskAnalysis, risk_cfg: Config) -> None:
    result.measures = _measures(result)
    result.limits = limits.check_limits(result.measures, risk_cfg)
    result.status = limits.overall_status(result.limits)


# ---------------------------------------------------------------------------
#  Analisis
# ---------------------------------------------------------------------------


def analyze_book(
    weights: pd.Series,
    sectors: pd.Series,
    nav: float,
    close_wide: pd.DataFrame,
    benchmark: str,
    risk_cfg: Config,
    *,
    as_of: pd.Timestamp,
    volume_wide: pd.DataFrame | None = None,
    factor_scores: pd.DataFrame | None = None,
    universe_scores: pd.DataFrame | None = None,
    panel: pd.DataFrame | None = None,
    cost_bps: float | None = None,
    turnover: float | None = None,
    rebalances_per_year: float | None = None,
    strategy_fingerprint: str = "",
    book_history: bool = True,
) -> RiskAnalysis:
    """Riesgo de una cartera concreta sin historia de la estrategia.

    `panel` (el panel puntuado) activa el modelo factorial. `cost_bps` es el
    coste fijo por lado del backtest, para el coste de liquidacion; `turnover`
    y `rebalances_per_year` activan la curva de capacidad. `book_history=False`
    deja el VaR estresado y el GARCH de la cartera para quien llama: `analyze`
    los calcula con la historia larga, y hacerlos aqui tambien costaba el doble
    y dejaba en el reporte avisos de la pasada corta que la larga ya resolvia.
    """
    validate_policy(risk_cfg)
    weights = weights[weights.abs() > 0].astype(float)
    sectors = sectors.reindex(weights.index).fillna("Unknown")
    warnings: list[str] = []
    as_of = pd.Timestamp(as_of)
    confidences = _confidences(risk_cfg)
    seed = int(risk_cfg.get("montecarlo.seed"))

    # -- ex-ante ---------------------------------------------------------------
    returns, coverage = exposure.trailing_returns(
        close_wide, list(weights.index) + [benchmark], as_of, int(risk_cfg.get("exposure.cov_window_d"))
    )
    if coverage.get(benchmark, 0.0) == 0.0:
        raise ValueError(
            f"sin precios de {benchmark} en la ventana de covarianza: anadelo a los "
            "precios cargados antes de medir el riesgo"
        )
    ex_ante = exposure.ex_ante_risk(
        weights, sectors, returns, benchmark,
        confidence=0.99, coverage=coverage.drop(benchmark),
        min_coverage=float(risk_cfg.get("exposure.min_coverage")),
    )
    warnings += ex_ante.warnings
    conc = exposure.concentration(weights, sectors)

    factors = pd.DataFrame()
    if factor_scores is not None and not factor_scores.empty:
        factors = exposure.factor_exposure(weights, factor_scores, universe_scores)

    # -- liquidez ----------------------------------------------------------------
    participation = float(risk_cfg.get("exposure.participation"))
    max_days = float(risk_cfg.get("limits.max_liquidation_days", 5.0))
    if volume_wide is not None and not volume_wide.empty:
        adv = exposure.average_dollar_volume(
            close_wide.reindex(columns=weights.index), volume_wide, as_of,
            int(risk_cfg.get("exposure.adv_window_d")),
        ).reindex(weights.index)
    else:
        adv = pd.Series(np.nan, index=weights.index)
        warnings.append("sin volumen: no se mide la liquidez ni la capacidad")
    liq_table, liq_summary = exposure.liquidity(
        weights, nav, adv, participation=participation, max_days=max_days
    )
    if liq_summary.get("weight_without_adv", 0.0) > 0 and volume_wide is not None:
        warnings.append(
            f"{liq_summary['weight_without_adv']:.1%} del NAV sin volumen medible: "
            "su liquidez no entra en el limite"
        )

    # -- VaR de la cartera actual ---------------------------------------------
    horizons = [int(h) for h in risk_cfg.get("montecarlo.parametric.horizons_d")]
    mc_table, component_es = montecarlo.parametric_portfolio_mc(
        weights, ex_ante.cov_daily,
        horizons_d=horizons, confidences=confidences,
        n_sims=int(risk_cfg.get("montecarlo.parametric.n_sims")),
        t_dof=float(risk_cfg.get("montecarlo.parametric.t_dof")),
        seed=seed,
    )
    mc_table = mc_table.assign(method="mc_" + mc_table["distribution"]).drop(columns="distribution")
    book_rows = _book_historical_var(returns, weights, confidences)
    book_var = pd.concat([pd.DataFrame(book_rows), mc_table], ignore_index=True)
    if len(component_es):
        component_es = component_es.assign(
            sector=component_es["ticker"].map(sectors).fillna("Unknown")
        )

    # -- estres hipotetico e inverso ------------------------------------------
    hypothetical, hypothetical_detail = stress.hypothetical_stress(
        weights, sectors, ex_ante.betas, risk_cfg.get("stress.hypothetical", []),
        factor_scores=factor_scores,
    )
    if not hypothetical.empty and (hypothetical["missing_factors"] != "").any():
        warnings.append(
            "escenarios con choque de factor sin scores de la cartera: el choque de "
            "factor se trato como cero"
        )
    reverse = stress.reverse_stress(ex_ante.beta, conc.get("invested", 0.0),
                                    risk_cfg.get("stress.reverse.loss_thresholds"))

    result = RiskAnalysis(
        as_of=as_of,
        nav=float(nav),
        weights=weights,
        sectors=sectors,
        benchmark=benchmark,
        risk_fingerprint=risk_cfg.fingerprint,
        strategy_fingerprint=strategy_fingerprint,
        concentration=conc,
        ex_ante=ex_ante,
        factor_exposure=factors,
        liquidity=liq_table,
        liquidity_summary=liq_summary,
        book_var=book_var,
        component_es=component_es,
        hypothetical=hypothetical,
        hypothetical_detail=hypothetical_detail,
        reverse=reverse,
        measures={},
        limits=pd.DataFrame(),
        status="SIN DATO",
        warnings=warnings,
    )

    # -- estres de covarianza --------------------------------------------------
    result.covariance_stress = stress.covariance_shocks(
        weights, ex_ante.cov_daily,
        [{"name": "Base", "vol_mult": 1.0, "corr_blend": 0.0}]
        + list(risk_cfg.get("stress.covariance", [])),
    )

    # -- VaR estresado y GARCH de la cartera ------------------------------------
    if book_history:
        _set_book_history(result, close_wide, risk_cfg, confidences, seed)

    # -- modelo factorial ------------------------------------------------------------
    if panel is not None and not panel.empty:
        try:
            result.factor_risk = factor_model.factor_risk(
                weights, sectors, close_wide, panel, as_of,
                int(risk_cfg.get("factor_model.window_d")),
            )
            result.warnings += result.factor_risk.warnings
            ratio = result.factor_risk.total_vol / ex_ante.volatility if ex_ante.volatility else np.nan
            if np.isfinite(ratio) and ratio > 1.25:
                # La covarianza de Ledoit-Wolf promedia el ano; el modelo
                # factorial aplica las exposiciones de HOY. Si las de hoy son
                # mas extremas, el riesgo que viene es el factorial.
                result.warnings.append(
                    f"el modelo factorial ve {ratio - 1:.0%} mas volatilidad que Ledoit-Wolf "
                    f"({result.factor_risk.total_vol:.1%} frente a {ex_ante.volatility:.1%}): las "
                    "exposiciones de hoy son mas extremas que las del ano de la covarianza, y los "
                    "VaR ex-ante basados en ella se quedan cortos"
                )
            elif np.isfinite(ratio) and ratio < 1 / 1.25:
                # El caso inverso tambien es una discrepancia de mas del 25%: las
                # exposiciones de hoy son mas suaves que las del ano y los VaR
                # ex-ante salen conservadores. No es alarma, pero hay que saberlo.
                result.warnings.append(
                    f"el modelo factorial ve {1 - ratio:.0%} menos volatilidad que Ledoit-Wolf "
                    f"({result.factor_risk.total_vol:.1%} frente a {ex_ante.volatility:.1%}): las "
                    "exposiciones de hoy son mas suaves que las del ano de la covarianza, y los "
                    "VaR ex-ante basados en ella salen conservadores"
                )
        except ValueError as exc:
            result.warnings.append(f"modelo factorial: {exc}")

    # -- coste de liquidacion y capacidad ------------------------------------------
    liq_cfg = risk_cfg.section("liquidity")
    if cost_bps is None:
        cost_bps = 0.0
        result.warnings.append("sin coste fijo por lado: el coste de liquidacion solo cuenta impacto")
    daily_vol = pd.Series(np.sqrt(np.clip(np.diag(ex_ante.cov_daily.loc[weights.index, weights.index]), 0.0, None)),
                          index=weights.index)
    if adv.notna().any():
        _, normal_cost = liquidity.liquidation_cost(
            weights, nav, adv, daily_vol, cost_bps=cost_bps,
            impact_coef=float(liq_cfg["impact_coef"]),
        )
        crisis_detail, crisis_cost = liquidity.liquidation_cost(
            weights, nav, adv, daily_vol, cost_bps=cost_bps,
            impact_coef=float(liq_cfg["impact_coef"]),
            adv_mult=float(liq_cfg["stress_adv_mult"]),
            vol_mult=float(liq_cfg["stress_vol_mult"]),
            spread_mult=float(liq_cfg["stress_cost_mult"]),
        )
        result.liquidation = pd.DataFrame(
            {"cost": [normal_cost, crisis_cost]}, index=["normal", "crisis"]
        ).assign(cost_usd=lambda f: f["cost"] * nav)
        result.liquidation_detail = crisis_detail
        if turnover is not None and rebalances_per_year:
            result.capacity = liquidity.capacity_curve(
                weights, adv, daily_vol, turnover=float(turnover),
                rebalances_per_year=float(rebalances_per_year), cost_bps=cost_bps,
                impact_coef=float(liq_cfg["impact_coef"]),
                navs=[float(v) for v in liq_cfg["capacity_navs"]],
            )
            result.capacity_nav = liquidity.capacity_at_budget(
                weights, adv, daily_vol, turnover=float(turnover),
                rebalances_per_year=float(rebalances_per_year),
                impact_coef=float(liq_cfg["impact_coef"]),
                budget=float(liq_cfg["capacity_budget"]),
            )

    _refresh_limits(result, risk_cfg)
    return result


def analyze(
    weights: pd.Series,
    sectors: pd.Series,
    nav: float,
    close_wide: pd.DataFrame,
    benchmark: str,
    risk_cfg: Config,
    *,
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series | None,
    as_of: pd.Timestamp,
    volume_wide: pd.DataFrame | None = None,
    stress_close: pd.DataFrame | None = None,
    factor_scores: pd.DataFrame | None = None,
    universe_scores: pd.DataFrame | None = None,
    panel: pd.DataFrame | None = None,
    holdings: pd.DataFrame | None = None,
    cost_bps: float | None = None,
    turnover: float | None = None,
    rebalances_per_year: float | None = None,
    strategy_fingerprint: str = "",
) -> RiskAnalysis:
    """Reporte de riesgo completo: la cartera de hoy y la historia de la estrategia.

    `stress_close` son precios con historia larga (desde `stress.history_start`)
    para las posiciones actuales y el benchmark. Si llega, el estres historico,
    el VaR estresado y el GARCH de la cartera usan esa historia; si no, la de
    `close_wide`. `holdings` (las posiciones de cada rebalanceo del backtest)
    activa la calibracion del modelo de riesgo.
    """
    long_history = stress_close is not None and not stress_close.empty
    result = analyze_book(
        weights, sectors, nav, close_wide, benchmark, risk_cfg,
        as_of=as_of, volume_wide=volume_wide, factor_scores=factor_scores,
        universe_scores=universe_scores, panel=panel, cost_bps=cost_bps,
        turnover=turnover, rebalances_per_year=rebalances_per_year,
        strategy_fingerprint=strategy_fingerprint, book_history=not long_history,
    )
    lam = float(risk_cfg.get("var.ewma_lambda"))
    confidences = _confidences(risk_cfg)
    seed = int(risk_cfg.get("montecarlo.seed"))
    nav_series = strategy_nav.dropna().astype(float)
    returns = nav_series.pct_change().dropna()

    # -- VaR de la estrategia y su backtest --------------------------------------
    result.strategy_var = var.var_table(
        returns, confidences, [int(h) for h in risk_cfg.get("var.horizons_d")], lam
    )
    result.var_backtest, result.var_forecasts = var.var_backtest(
        returns,
        float(risk_cfg.get("var.backtest.confidence")),
        int(risk_cfg.get("var.backtest.window_d")),
        lam,
        recent_d=int(risk_cfg.get("var.backtest.recent_d")),
    )
    result.tail = var.tail_statistics(returns)
    _set_garch(result, "estrategia", returns, risk_cfg, confidences, seed)

    # -- bootstrap -----------------------------------------------------------------
    frame = returns.rename("strategy").to_frame()
    if benchmark_nav is not None:
        bench_returns = benchmark_nav.reindex(nav_series.index).ffill().pct_change()
        frame["benchmark"] = bench_returns
    frame = frame.dropna()
    boot_cfg = risk_cfg.section("montecarlo.bootstrap")
    result.bootstrap = montecarlo.bootstrap_paths(
        frame,
        [int(round(float(y) * TRADING_DAYS)) for y in boot_cfg["horizons_y"]],
        n_paths=int(boot_cfg["n_paths"]),
        mean_block=int(boot_cfg["mean_block_d"]),
        seed=seed,
        haircut_annual=float(boot_cfg["return_haircut"]),
    )
    result.bootstrap_summary = montecarlo.bootstrap_summary(
        result.bootstrap, [float(x) for x in boot_cfg["drawdown_thresholds"]]
    )

    # -- estres historico, VaR estresado con historia larga y peores ventanas ----
    prices = stress_close if long_history else close_wide
    result.historical, result.historical_detail = stress.historical_stress(
        result.weights, result.ex_ante.betas, prices, benchmark,
        risk_cfg.get("stress.historical", []), strategy_nav=nav_series,
    )
    if not result.historical.empty and "status" in result.historical:
        missing = result.historical[result.historical["status"] != "ok"]
        if len(missing):
            result.warnings.append(
                f"{len(missing)} episodios historicos sin precios del benchmark: "
                f"{', '.join(missing['scenario'])}"
            )
    # Si la historia larga no sirve (sin benchmark), se cae a la del panel: un
    # VaR estresado sobre menos anos es mejor que ninguno, y el aviso lo dice.
    if long_history and not _set_book_history(result, stress_close, risk_cfg, confidences, seed):
        _set_book_history(result, close_wide, risk_cfg, confidences, seed)
    result.worst_windows = stress.worst_windows(nav_series, benchmark_nav, WORST_WINDOWS_D)

    # -- historia del riesgo y calibracion -------------------------------------------
    hist_cfg = risk_cfg.section("history")
    result.rolling = history.rolling_risk(nav_series, benchmark_nav, int(hist_cfg["rolling_window_d"]))
    result.rolling_summary = history.rolling_summary(result.rolling)
    if holdings is not None and not holdings.empty:
        result.calibration, result.calibration_summary = history.risk_model_calibration(
            holdings, close_wide, nav_series, window=int(hist_cfg["calibration_window_d"])
        )

    # -- drawdown actual y limites con historia ----------------------------------
    result.current_drawdown = float(-drawdown_series(nav_series).iloc[-1])
    _refresh_limits(result, risk_cfg)
    return result
