"""Simulaciones de Monte Carlo: la distribucion de lo que puede pasar, no un punto.

Dos simulaciones con preguntas distintas:

1. **Bootstrap de la historia** (`bootstrap_paths`). Que rango de resultados
   cabe esperar en 1, 3 y 5 anos, con que probabilidad se pierde dinero, con
   cual se queda detras del benchmark y que drawdown hay que estar dispuesto a
   aguantar. Remuestrea bloques de dias REALES de la estrategia y el benchmark
   con los mismos indices: conserva la correlacion entre ambos, la
   autocorrelacion y los racimos de volatilidad dentro de cada bloque. Es el
   bootstrap estacionario de Politis y Romano (1994): bloques de longitud
   geometrica, para que la serie remuestreada siga siendo estacionaria.

   Su limite, declarado: no puede producir un dia peor que el peor de la
   muestra. 2008 no esta en el backtest. Para eso estan las pruebas de estres.

2. **Parametrica sobre la cartera actual** (`parametric_portfolio_mc`). Cuanto
   puede perder el libro de HOY manana y en un mes, con su covarianza actual.
   Normal y t de Student con la MISMA covarianza: la distancia entre las dos es
   la prima de cola gorda que un VaR normal no ve.

Todo con semilla explicita: dos corridas con la misma semilla dan lo mismo, y un
numero del reporte se puede reproducir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..metrics import TRADING_DAYS

_PERCENTILES = (5, 25, 50, 75, 95)


@dataclass
class BootstrapResult:
    horizons_d: list[int]
    # horizonte -> matriz (n_paths, n_series) de retornos acumulados
    terminal: dict[int, np.ndarray]
    # horizonte -> vector (n_paths,) del peor drawdown de la estrategia
    max_drawdown: dict[int, np.ndarray]
    # percentiles de la riqueza de la estrategia por sesion simulada
    fan: pd.DataFrame
    series: list[str]
    haircut: float


def bootstrap_paths(
    returns: pd.DataFrame,
    horizons_d,
    *,
    n_paths: int,
    mean_block: int,
    seed: int,
    haircut_annual: float = 0.0,
    fan_step: int = 5,
) -> BootstrapResult:
    """Bootstrap estacionario de bloques sobre retornos diarios.

    `returns` trae la estrategia en la primera columna y, si hay, el benchmark
    en la segunda. `haircut_annual` se resta SOLO a la estrategia: es el
    descuento por sesgo de supervivencia, que el benchmark no tiene en la misma
    medida.

    Se simula paso a paso sin guardar las trayectorias enteras: con 10.000
    caminos a cinco anos serian cien millones de numeros para quedarse con los
    percentiles.
    """
    data = returns.dropna().to_numpy(dtype=float).copy()
    if len(data) < mean_block * 2:
        raise ValueError(
            f"{len(data)} dias de historia para bloques de {mean_block}: "
            "hace falta al menos el doble de historia que la longitud del bloque"
        )
    data[:, 0] -= haircut_annual / TRADING_DAYS

    horizons = sorted({int(h) for h in horizons_d})
    horizon_max = horizons[-1]
    n_obs, n_series = data.shape
    rng = np.random.default_rng(seed)
    restart_prob = 1.0 / max(mean_block, 1)

    index = rng.integers(0, n_obs, n_paths)
    wealth = np.ones((n_paths, n_series))
    peak = np.ones(n_paths)
    worst = np.zeros(n_paths)

    terminal: dict[int, np.ndarray] = {}
    drawdowns: dict[int, np.ndarray] = {}
    fan_rows: list[dict] = [{"day": 0, **{f"p{q}": 1.0 for q in _PERCENTILES}}]

    for step in range(1, horizon_max + 1):
        if step > 1:
            restart = rng.random(n_paths) < restart_prob
            index = np.where(restart, rng.integers(0, n_obs, n_paths), (index + 1) % n_obs)
        wealth *= 1.0 + data[index]
        peak = np.maximum(peak, wealth[:, 0])
        worst = np.minimum(worst, wealth[:, 0] / peak - 1.0)

        if step in horizons:
            terminal[step] = wealth.copy() - 1.0
            drawdowns[step] = worst.copy()
        if step % fan_step == 0 or step == horizon_max:
            levels = np.percentile(wealth[:, 0], _PERCENTILES)
            fan_rows.append({"day": step, **{f"p{q}": float(v) for q, v in zip(_PERCENTILES, levels)}})

    return BootstrapResult(
        horizons_d=horizons,
        terminal=terminal,
        max_drawdown=drawdowns,
        fan=pd.DataFrame(fan_rows).set_index("day"),
        series=list(returns.columns),
        haircut=haircut_annual,
    )


def bootstrap_summary(result: BootstrapResult, drawdown_thresholds) -> pd.DataFrame:
    """Una fila por horizonte: percentiles, probabilidades y drawdowns."""
    rows = []
    for h in result.horizons_d:
        paths = result.terminal[h]
        strategy = paths[:, 0]
        years = h / TRADING_DAYS
        tail = np.quantile(strategy, 0.05)
        row = {
            "horizon_d": h,
            "years": years,
            **{f"p{q}": float(np.percentile(strategy, q)) for q in _PERCENTILES},
            "median_annualized": float((1.0 + np.median(strategy)) ** (1.0 / years) - 1.0),
            "prob_loss": float((strategy < 0).mean()),
            "var_95": float(-tail),
            "es_95": float(-strategy[strategy <= tail].mean()),
            "median_max_drawdown": float(np.median(result.max_drawdown[h])),
            "p95_max_drawdown": float(np.quantile(result.max_drawdown[h], 0.05)),
        }
        if paths.shape[1] > 1:
            row["prob_underperform"] = float((strategy < paths[:, 1]).mean())
            row["benchmark_p50"] = float(np.median(paths[:, 1]))
        for threshold in drawdown_thresholds:
            row[f"prob_dd_{int(round(threshold * 100))}"] = float(
                (result.max_drawdown[h] <= -threshold).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("horizon_d")


def _cholesky(cov: np.ndarray) -> np.ndarray:
    """Cholesky robusta: si la matriz no es definida positiva, se recortan los
    autovalores negativos (redondeo numerico, o nombres sin variacion)."""
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(cov)
        values = np.clip(values, 1e-12, None)
        return vectors @ np.diag(np.sqrt(values))


def parametric_portfolio_mc(
    weights: pd.Series,
    cov_daily: pd.DataFrame,
    *,
    horizons_d,
    confidences,
    n_sims: int,
    t_dof: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """VaR y ES de la cartera actual por simulacion normal y t de Student.

    Media cero: a uno y veintiun dias el drift es ruido frente a la
    volatilidad, y suponerlo positivo reduce el VaR por la via de la fe.

    La t multivariante se construye como normal / sqrt(chi2/dof) y se reescala
    por (dof-2)/dof para que su covarianza sea exactamente la estimada. A un
    mes, la suma de 21 dias t es mas normal que una sola t: usar una t a ese
    horizonte es conservador, y se declara.

    Devuelve `(resumen, contribuciones)`. Las contribuciones son el ES por
    componentes de la t al primer horizonte y la mayor confianza: la perdida
    media de cada nombre en los escenarios de la cola. Suman el ES total.
    """
    names = list(weights.index)
    w = weights.to_numpy(dtype=float)
    cov = cov_daily.loc[names, names].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    horizons = sorted({int(h) for h in horizons_d})
    confidences = sorted(confidences)

    base_normal = rng.standard_normal((n_sims, len(names)))
    chi2 = rng.chisquare(t_dof, n_sims)
    t_scale = np.sqrt((t_dof - 2.0) / chi2)[:, None]

    rows = []
    component = pd.DataFrame()
    for h in horizons:
        factor = _cholesky(cov * h)
        normal_draws = base_normal @ factor.T
        t_draws = normal_draws * t_scale
        for label, draws in (("normal", normal_draws), ("t_student", t_draws)):
            pnl = draws @ w
            for c in confidences:
                cutoff = np.quantile(pnl, 1.0 - c)
                tail = pnl <= cutoff
                rows.append({
                    "distribution": label,
                    "horizon_d": h,
                    "confidence": c,
                    "var": float(-cutoff),
                    "es": float(-pnl[tail].mean()),
                })
                if label == "t_student" and h == horizons[0] and c == confidences[-1]:
                    component = pd.DataFrame({
                        "ticker": names,
                        "weight": w,
                        "component_es": -(draws[tail] * w).mean(axis=0),
                    })
    if not component.empty:
        total = component["component_es"].sum()
        component["es_share"] = component["component_es"] / total if total else np.nan
        component = component.sort_values("component_es", ascending=False).reset_index(drop=True)
    return pd.DataFrame(rows), component
