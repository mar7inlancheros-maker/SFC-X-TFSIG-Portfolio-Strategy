"""Junta todas las medidas de riesgo sobre una cartera concreta. Funciones puras.

Dos niveles, porque hay dos momentos en que se mira el riesgo:

- **`analyze_book`**: solo la cartera y los precios del ultimo ano. Es lo que
  se puede calcular ANTES de operar, sobre la cartera objetivo de `ordenes`:
  riesgo ex-ante, concentracion, liquidez, Monte Carlo parametrico, estres
  hipotetico y limites.
- **`analyze`**: lo anterior mas todo lo que necesita la historia de la
  estrategia -- VaR por cuatro metodos y su backtest, bootstrap a varios anos,
  estres historico, peores ventanas y drawdown actual. Es el reporte de riesgo
  del comite.

Nada toca red ni disco: entran DataFrames ya cargados, sale un `RiskAnalysis`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config, ConfigError
from ..metrics import TRADING_DAYS, drawdown_series
from ..universe import SECTORS
from . import exposure, limits, montecarlo, stress, var

WORST_WINDOWS_D = (1, 5, 21, 63)


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
#  Analisis
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


def _measures(ex_ante: exposure.ExAnteRisk, conc: dict, liq_summary: dict,
              book_var: pd.DataFrame, hypothetical: pd.DataFrame) -> dict[str, float]:
    t_rows = book_var[(book_var["method"] == "mc_t_student") & (book_var["confidence"] == 0.99)]
    t_rows = t_rows[t_rows["horizon_d"] == t_rows["horizon_d"].min()] if len(t_rows) else t_rows
    worst_stress = hypothetical["portfolio"].min() if not hypothetical.empty else float("nan")
    return {
        "ex_ante_vol": ex_ante.volatility,
        "var_99_1d": float(t_rows["var"].iloc[0]) if len(t_rows) else float("nan"),
        "beta": ex_ante.beta,
        "tracking_error": ex_ante.tracking_error,
        "max_name_risk_share": float(ex_ante.contributions["risk_share"].max()),
        "max_sector_risk_share": float(ex_ante.sectors["risk_share"].max()),
        "max_liquidation_days": liq_summary.get("max_days", float("nan")),
        "effective_names": conc.get("effective_names", float("nan")),
        "max_stress_loss": max(0.0, -float(worst_stress)) if np.isfinite(worst_stress) else float("nan"),
    }


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
    strategy_fingerprint: str = "",
) -> RiskAnalysis:
    """Riesgo de una cartera concreta sin historia de la estrategia."""
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

    measures = _measures(ex_ante, conc, liq_summary, book_var, hypothetical)
    checks = limits.check_limits(measures, risk_cfg)

    return RiskAnalysis(
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
        measures=measures,
        limits=checks,
        status=limits.overall_status(checks),
        warnings=warnings,
    )


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
    strategy_fingerprint: str = "",
) -> RiskAnalysis:
    """Reporte de riesgo completo: la cartera de hoy y la historia de la estrategia.

    `stress_close` son precios con historia larga (desde `stress.history_start`)
    para las posiciones actuales y el benchmark. Si no llega, el estres
    historico usa `close_wide` y los episodios anteriores quedan sin datos.
    """
    result = analyze_book(
        weights, sectors, nav, close_wide, benchmark, risk_cfg,
        as_of=as_of, volume_wide=volume_wide, factor_scores=factor_scores,
        universe_scores=universe_scores, strategy_fingerprint=strategy_fingerprint,
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

    # -- estres historico y peores ventanas -----------------------------------------
    prices = stress_close if stress_close is not None and not stress_close.empty else close_wide
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
    result.worst_windows = stress.worst_windows(nav_series, benchmark_nav, WORST_WINDOWS_D)

    # -- drawdown actual y limites con historia ----------------------------------
    result.current_drawdown = float(-drawdown_series(nav_series).iloc[-1])
    result.measures["current_drawdown"] = result.current_drawdown
    result.limits = limits.check_limits(result.measures, risk_cfg)
    result.status = limits.overall_status(result.limits)
    return result
