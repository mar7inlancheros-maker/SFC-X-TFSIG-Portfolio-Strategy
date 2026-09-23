"""Reversion a la media, persistencia y clasificacion de regimen. Funciones puras.

La clasificacion MOMENTUM / NEUTRAL / MEAN REVERSION es por activo y contra su
PROPIA historia, no contra los demas (eso lo hace el score de seccion cruzada).
Reglas exactas, con los umbrales del YAML:

1. **MEAN REVERSION** si el precio esta estirado -- |z de 20 dias| >=
   `reversion_threshold` -- Y la serie muestra anti-persistencia: ratio de
   varianzas VR(5) < 1. Estirado sin anti-persistencia no basta: una accion en
   tendencia fuerte esta "estirada" todo el tiempo.
2. **MOMENTUM** si el momentum ajustado por volatilidad -- retorno 12-1 dividido
   por la volatilidad anual, que se lee como un Sharpe de la tendencia -- supera
   `trend_threshold` en valor absoluto Y el precio esta del mismo lado de su
   media de 200 sesiones. Direccion: UP o DOWN.
3. **NEUTRAL** en cualquier otro caso.

VR(q) de Lo-MacKinlay: varianza de retornos de q dias / (q x varianza diaria).
1 = paseo aleatorio; > 1 persistencia; < 1 reversion.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TD = 252


def price_zscore(close: pd.Series, window: int = 20) -> float:
    s = close.dropna()
    if len(s) < window:
        return float("nan")
    tail = s.iloc[-window:]
    std = tail.std(ddof=1)
    return float((s.iloc[-1] - tail.mean()) / std) if std > 0 else 0.0


def bollinger_position(close: pd.Series, window: int = 20, k: float = 2.0) -> float:
    """%B: 0 en la banda inferior, 1 en la superior, 0,5 en la media."""
    s = close.dropna()
    if len(s) < window:
        return float("nan")
    tail = s.iloc[-window:]
    mid, std = tail.mean(), tail.std(ddof=1)
    lower, upper = mid - k * std, mid + k * std
    return float((s.iloc[-1] - lower) / (upper - lower)) if upper > lower else 0.5


def variance_ratio(returns: pd.Series, q: int = 5, window: int = 252) -> float:
    r = returns.dropna().iloc[-window:]
    if len(r) < q * 10:
        return float("nan")
    var1 = r.var(ddof=1)
    rq = r.rolling(q).sum().dropna()
    return float(rq.var(ddof=1) / (q * var1)) if var1 > 0 else float("nan")


def autocorrelation(returns: pd.Series, lag: int = 1, window: int = 252) -> float:
    r = returns.dropna().iloc[-window:]
    return float(r.autocorr(lag)) if len(r) > lag + 10 else float("nan")


def mean_reversion_features(close: pd.Series, returns: pd.Series) -> dict[str, float]:
    s = close.dropna()
    r = returns.dropna()
    vol = float(r.iloc[-TD:].std(ddof=1) * math.sqrt(TD)) if len(r) > 20 else float("nan")
    mom_12_1 = float(s.iloc[-22] / s.iloc[-253] - 1.0) if len(s) > 253 else float("nan")
    sma200 = s.rolling(200).mean().iloc[-1] if len(s) >= 200 else np.nan
    return {
        "zscore_20d": price_zscore(s, 20),
        "bollinger_pct_b": bollinger_position(s),
        "reversal_1m": float(s.iloc[-1] / s.iloc[-22] - 1.0) if len(s) > 22 else float("nan"),
        "vol_adj_momentum": mom_12_1 / vol if vol and not np.isnan(vol) and vol > 0 else float("nan"),
        "autocorr_1": autocorrelation(r, 1),
        "variance_ratio_5": variance_ratio(r, 5),
        "above_sma200": float(np.sign(s.iloc[-1] - sma200)) if not np.isnan(sma200) else float("nan"),
    }


def classify_regime(features: dict[str, float], trend_threshold: float, reversion_threshold: float) -> str:
    z = features.get("zscore_20d", np.nan)
    vr = features.get("variance_ratio_5", np.nan)
    vam = features.get("vol_adj_momentum", np.nan)
    side = features.get("above_sma200", np.nan)

    if not np.isnan(z) and not np.isnan(vr) and abs(z) >= reversion_threshold and vr < 1.0:
        return "MEAN REVERSION"
    if not np.isnan(vam) and not np.isnan(side) and abs(vam) >= trend_threshold and np.sign(vam) == side:
        return "MOMENTUM UP" if vam > 0 else "MOMENTUM DOWN"
    return "NEUTRAL"
