"""Volatilidad y su regimen. Funciones puras.

El regimen se define contra la PROPIA historia del activo, con percentiles
declarados en el YAML, nunca con umbrales absolutos: un 35% de volatilidad es
calma para una biotecnologica y panico para una electrica.

    percentil = fraccion de la historia de vol de 21 dias por debajo de la actual
    LOW < p_low <= NORMAL < p_high <= HIGH < p_extreme <= EXTREME

EWMA: sigma^2_t = lambda sigma^2_{t-1} + (1 - lambda) r^2_{t-1}, RiskMetrics
(lambda = 0,94). Reacciona mas rapido que la ventana movil a un cambio de
regimen, que es justo cuando importa.

ATR(14) de Wilder, reportado como fraccion del precio para poder comparar entre
activos.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TD = 252


def ewma_vol(returns: pd.Series, lam: float) -> float:
    r = returns.dropna()
    if len(r) < 20:
        return float("nan")
    var = float(r.iloc[:20].var(ddof=1))
    for x in r.iloc[20:]:
        var = lam * var + (1.0 - lam) * x * x
    return math.sqrt(var * TD)


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> float:
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    last = close.dropna().iloc[-1] if close.notna().any() else np.nan
    value = atr.dropna()
    return float(value.iloc[-1] / last) if len(value) and last else float("nan")


def volatility_features(
    returns: pd.Series,
    close: pd.Series,
    high: pd.Series | None,
    low: pd.Series | None,
    *,
    lam: float,
    thresholds: dict[str, float],
) -> dict[str, float | str]:
    r = returns.dropna()
    if len(r) < 63:
        return {}
    rolling21 = r.rolling(21).std(ddof=1).dropna() * math.sqrt(TD)
    current = float(rolling21.iloc[-1])
    percentile = float((rolling21 < current).mean())

    if percentile >= thresholds.get("extreme", 0.95):
        regime = "EXTREME"
    elif percentile >= thresholds.get("high", 0.80):
        regime = "HIGH"
    elif percentile < thresholds.get("low", 0.20):
        regime = "LOW"
    else:
        regime = "NORMAL"

    down = r[r < 0]
    return {
        "vol_21d": current,
        "vol_63d": float(r.iloc[-63:].std(ddof=1) * math.sqrt(TD)),
        "vol_252d": float(r.iloc[-252:].std(ddof=1) * math.sqrt(TD)),
        "vol_ewma": ewma_vol(r, lam),
        "downside_vol_semidev": float(math.sqrt((np.minimum(r, 0.0) ** 2).mean()) * math.sqrt(TD)),
        "downside_share": float(len(down) / len(r)),
        "vol_percentile": percentile,
        "vol_regime": regime,
        # Vol de la vol: dispersion de la vol de 21 dias, relativa a su media.
        "vol_of_vol": float(rolling21.std() / rolling21.mean()) if rolling21.mean() > 0 else float("nan"),
        "atr_pct": atr_pct(high, low, close) if high is not None and low is not None else float("nan"),
    }
