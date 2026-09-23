"""Riesgo ex-ante de una cartera concreta: de donde viene y cuanto cuesta salir.

El backtest dice cuanto se movio la estrategia. Esto dice cuanto puede moverse
la cartera que hay HOY, con sus pesos de hoy y la covarianza reciente de sus
nombres. Son preguntas distintas: la cartera de hoy puede estar mucho mas
concentrada, o en un regimen de volatilidad distinto, que la media historica.

**Covarianza con contraccion de Ledoit-Wolf** (2004, "Honey, I shrunk the sample
covariance matrix"), hacia correlacion constante. Con 30 nombres y 252 dias la
matriz muestral tiene 465 covarianzas estimadas con ruido; sus autovalores
extremos se exageran y la cartera de minimo riesgo aparente es la que mejor
explota ese ruido. La contraccion tira de cada correlacion hacia la media, en
la proporcion que minimiza el error esperado. No es un parametro: sale de los
datos.

**Descomposicion de Euler.** La volatilidad de la cartera es homogenea de grado
1 en los pesos, asi que se reparte EXACTAMENTE entre las posiciones:
`sigma_p = sum_i w_i * (Sigma w)_i / sigma_p`. La contribucion de un nombre
depende de su peso, de su volatilidad y de su correlacion con el resto; el
tope de peso por nombre solo controla el primero de los tres.

**Nombres con historia corta.** Los dias sin dato cuentan como retorno cero.
Subestima la volatilidad de ese nombre; se avisa en el reporte y el aviso dice
cuales son.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
#  Datos de entrada
# ---------------------------------------------------------------------------


def trailing_returns(
    close_wide: pd.DataFrame,
    tickers,
    as_of: pd.Timestamp,
    window: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Retornos diarios de la ventana que termina en `as_of`, y su cobertura.

    Devuelve `(retornos, cobertura)`. Los huecos quedan a cero y la cobertura
    (fraccion de dias con dato) permite avisar de ello. Un ticker que no esta en
    `close_wide` sale con cobertura cero y retornos cero.
    """
    tickers = list(dict.fromkeys(tickers))
    visible = close_wide.loc[close_wide.index <= pd.Timestamp(as_of)]
    prices = visible.reindex(columns=tickers).tail(window + 1)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    coverage = returns.notna().mean() if len(returns) else pd.Series(0.0, index=tickers)
    return returns.fillna(0.0), coverage.reindex(tickers).fillna(0.0)


def average_dollar_volume(
    close_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    as_of: pd.Timestamp,
    window: int,
) -> pd.Series:
    """Mediana del volumen en dolares de las ultimas `window` sesiones.

    Mediana y no media, como en el filtro de liquidez del universo: un dia de
    inclusion en indice no hace liquida a una accion que no lo es el resto del
    mes.
    """
    columns = close_wide.columns.intersection(volume_wide.columns)
    visible = close_wide.index <= pd.Timestamp(as_of)
    dollar = close_wide.loc[visible, columns] * volume_wide.reindex(
        index=close_wide.index[visible], columns=columns
    )
    return dollar.tail(window).median()


# ---------------------------------------------------------------------------
#  Covarianza
# ---------------------------------------------------------------------------


def ledoit_wolf(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Covarianza contraida hacia correlacion constante. Devuelve (Sigma, delta).

    Traduccion directa de `covCor.m` de Ledoit y Wolf. `delta` es la
    intensidad de la contraccion: 0 es la muestral, 1 es el blanco puro.
    """
    x = returns.to_numpy(dtype="float64")
    t, n = x.shape
    if t < 2 or n == 0:
        raise ValueError("hacen falta al menos dos observaciones para una covarianza")
    x = x - x.mean(axis=0)
    sample = x.T @ x / t
    var = np.diag(sample).copy()
    std = np.sqrt(var)

    if n == 1:
        return pd.DataFrame(sample, index=returns.columns, columns=returns.columns), 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = sample / np.outer(std, std)
    valid = np.outer(std > 0, std > 0)
    np.fill_diagonal(valid, False)
    r_bar = float(np.nanmean(corr[valid])) if valid.any() else 0.0

    prior = r_bar * np.outer(std, std)
    np.fill_diagonal(prior, var)

    y2 = x ** 2
    pi_mat = y2.T @ y2 / t - sample ** 2
    pi_hat = float(pi_mat.sum())

    theta = (x ** 3).T @ x / t - var[:, None] * sample
    np.fill_diagonal(theta, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(std[:, None] > 0, std[None, :] / std[:, None], 0.0)
    rho_hat = float(np.trace(pi_mat) + r_bar * np.nansum(ratio * theta))

    gamma_hat = float(((sample - prior) ** 2).sum())
    if gamma_hat <= 0:
        delta = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = float(min(1.0, max(0.0, kappa / t)))

    shrunk = delta * prior + (1.0 - delta) * sample
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns), delta


# ---------------------------------------------------------------------------
#  Riesgo ex-ante y su descomposicion
# ---------------------------------------------------------------------------


@dataclass
class ExAnteRisk:
    """Riesgo de una cartera concreta. Volatilidades anuales; VaR diario."""

    volatility: float
    benchmark_volatility: float
    beta: float
    tracking_error: float
    correlation: float
    systematic_share: float
    diversification_ratio: float
    var_normal_1d: float
    confidence: float
    shrinkage: float
    contributions: pd.DataFrame
    sectors: pd.DataFrame
    betas: pd.Series
    cov_daily: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def ex_ante_risk(
    weights: pd.Series,
    sectors: pd.Series,
    returns: pd.DataFrame,
    benchmark: str,
    *,
    confidence: float = 0.99,
    coverage: pd.Series | None = None,
    min_coverage: float = 0.8,
) -> ExAnteRisk:
    """Volatilidad, beta, tracking error y contribuciones de cada posicion.

    `weights` es fraccion del NAV (la caja es lo que falta hasta 1 y no aporta
    riesgo). `returns` trae una columna por nombre de `weights` y otra con el
    benchmark.
    """
    if benchmark not in returns.columns:
        raise ValueError(
            f"falta el benchmark {benchmark} en los retornos: sin el no hay beta "
            "ni tracking error. Anadelo a los precios antes de medir el riesgo."
        )
    names = list(weights.index)
    cov, delta = ledoit_wolf(returns.reindex(columns=names + [benchmark]).fillna(0.0))
    sigma = cov.loc[names, names].to_numpy()
    w = weights.reindex(names).astype(float).to_numpy()
    cross = cov.loc[names, benchmark].to_numpy()
    var_b = float(cov.loc[benchmark, benchmark])

    var_p = float(w @ sigma @ w)
    vol_p = math.sqrt(max(var_p, 0.0))
    marginal = sigma @ w / vol_p if vol_p > 0 else np.zeros_like(w)
    component = w * marginal
    betas = cross / var_b if var_b > 0 else np.full_like(w, np.nan)
    beta_p = float(w @ betas)
    te_var = var_p - 2.0 * float(w @ cross) + var_b
    stdev = np.sqrt(np.diag(sigma))

    z = -NormalDist().inv_cdf(1.0 - confidence)
    annual = math.sqrt(TRADING_DAYS)

    contributions = pd.DataFrame({
        "sector": sectors.reindex(names).fillna("Unknown").to_numpy(),
        "weight": w,
        "volatility": stdev * annual,
        "beta": betas,
        "marginal": marginal * annual,
        "contribution": component * annual,
        "risk_share": component / vol_p if vol_p > 0 else np.nan,
        "component_var_1d": component * z,
    }, index=pd.Index(names, name="ticker")).sort_values("risk_share", ascending=False)

    by_sector = contributions.groupby("sector").agg(
        weight=("weight", "sum"),
        risk_share=("risk_share", "sum"),
        n_names=("weight", "size"),
    ).sort_values("risk_share", ascending=False)

    warnings: list[str] = []
    if coverage is not None:
        short = coverage.reindex(names)
        short = short[short < min_coverage]
        if len(short):
            listed = ", ".join(f"{t} ({c:.0%})" for t, c in short.sort_values().items())
            warnings.append(
                f"historia incompleta en la ventana de covarianza: {listed}. Los dias sin "
                "dato cuentan como retorno cero, lo que SUBESTIMA su volatilidad."
            )

    return ExAnteRisk(
        volatility=vol_p * annual,
        benchmark_volatility=math.sqrt(var_b) * annual,
        beta=beta_p,
        tracking_error=math.sqrt(max(te_var, 0.0)) * annual,
        correlation=float(w @ cross) / (vol_p * math.sqrt(var_b)) if vol_p > 0 and var_b > 0 else float("nan"),
        systematic_share=beta_p ** 2 * var_b / var_p if var_p > 0 else float("nan"),
        diversification_ratio=float(w @ stdev) / vol_p if vol_p > 0 else float("nan"),
        var_normal_1d=z * vol_p,
        confidence=confidence,
        shrinkage=delta,
        contributions=contributions,
        sectors=by_sector,
        betas=pd.Series(betas, index=names),
        cov_daily=cov,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
#  Concentracion, factores y liquidez
# ---------------------------------------------------------------------------


def concentration(weights: pd.Series, sectors: pd.Series) -> dict[str, object]:
    """Cuantas apuestas hay de verdad en la cartera.

    El numero efectivo de nombres (1 / HHI de los pesos invertidos) cuenta
    posiciones como si fueran iguales: treinta nombres con cinco al 15% son,
    en riesgo idiosincratico, una cartera de unos diez.
    """
    w = weights.astype(float)
    invested = float(w.sum())
    if invested <= 0:
        return {"n_names": 0, "invested": 0.0, "cash": 1.0}
    normalized = w / invested
    hhi = float((normalized ** 2).sum())
    by_sector = w.groupby(sectors.reindex(w.index).fillna("Unknown")).sum()
    return {
        "n_names": int((w > 0).sum()),
        "invested": invested,
        "cash": 1.0 - invested,
        "max_weight": float(w.max()),
        "max_weight_name": str(w.idxmax()),
        "top5_weight": float(w.nlargest(5).sum()),
        "hhi": hhi,
        "effective_names": 1.0 / hhi if hhi > 0 else float("nan"),
        "max_sector_weight": float(by_sector.max()),
        "max_sector": str(by_sector.idxmax()),
    }


def factor_exposure(
    weights: pd.Series,
    holding_scores: pd.DataFrame,
    universe_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Exposicion media ponderada a cada factor, frente a la del universo.

    Los scores son z-scores sectoriales, asi que 1.0 quiere decir "una
    desviacion por encima de su sector". Lo que el comite tiene que ver es si
    la cartera expresa los cuatro factores o solo uno: una exposicion de 1.5 en
    momentum y 0 en calidad es un fondo de momentum con otro nombre.
    """
    columns = [c for c in holding_scores.columns if c.startswith("score_")]
    w = weights.reindex(holding_scores.index).fillna(0.0)
    rows = []
    for column in columns:
        values = pd.to_numeric(holding_scores[column], errors="coerce")
        known = values.notna() & (w > 0)
        coverage = float(w[known].sum() / w.sum()) if w.sum() > 0 else float("nan")
        exposure = float((values[known] * w[known]).sum() / w[known].sum()) if known.any() else float("nan")
        universe = float("nan")
        if universe_scores is not None and column in universe_scores.columns:
            universe = float(pd.to_numeric(universe_scores[column], errors="coerce").mean())
        rows.append({
            "factor": column.removeprefix("score_"),
            "portfolio": exposure,
            "universe": universe,
            "active": exposure - universe if np.isfinite(universe) else float("nan"),
            "weight_covered": coverage,
        })
    return pd.DataFrame(rows).set_index("factor")


def liquidity(
    weights: pd.Series,
    nav: float,
    adv: pd.Series,
    *,
    participation: float,
    max_days: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Dias para liquidar cada posicion y capital maximo que admite la cartera.

    `adv` es el volumen diario en dolares. Se opera como mucho `participation`
    de ese volumen al dia. La capacidad es el NAV al que la posicion menos
    liquida tardaria `max_days` sesiones en venderse: por encima de ese capital,
    los costes del backtest dejan de ser creibles.
    """
    w = weights.astype(float)
    value = w * nav
    daily = participation * adv.reindex(w.index)
    days = value / daily
    capacity = max_days * daily / w.where(w > 0)
    table = pd.DataFrame({
        "weight": w,
        "value": value,
        "adv": adv.reindex(w.index),
        "days_to_liquidate": days,
        "capacity_nav": capacity,
    }).sort_values("days_to_liquidate", ascending=False)

    known = daily.notna() & (daily > 0)
    total = float(value[known].sum())

    def _liquid_within(d: float) -> float:
        if total <= 0:
            return float("nan")
        return float(np.minimum(value[known], d * daily[known]).sum() / total)

    summary = {
        "max_days": float(days[known].max()) if known.any() else float("nan"),
        "max_days_name": str(days[known].idxmax()) if known.any() else "",
        "liquid_1d": _liquid_within(1.0),
        "liquid_5d": _liquid_within(5.0),
        "capacity_nav": float(capacity[known].min()) if known.any() else float("nan"),
        "capacity_name": str(capacity[known].idxmin()) if known.any() else "",
        "weight_without_adv": float(w[~known].sum()),
    }
    return table, summary
