"""Liquidez. Funciones puras.

- ADV: MEDIANA del volumen en dolares de 63 sesiones. Mediana porque un dia de
  resultados o de rebalanceo de indices no hace liquida a una accion.
- Amihud (2002): media de |r| / volumen en dolares, x 1e6. Cuanto se mueve el
  precio por cada millon operado. Mas alto = menos liquido.
- Dias para liquidar: |posicion en dolares| / (participacion x ADV). Con 10% de
  participacion, operar mas es mover el precio contra uno mismo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def liquidity_features(close: pd.Series, volume: pd.Series, returns: pd.Series) -> dict[str, float]:
    dollar = (close * volume).dropna()
    if len(dollar) < 20:
        return {"adv_usd": float("nan"), "amihud": float("nan")}
    tail = dollar.iloc[-63:]
    r = returns.reindex(tail.index).abs()
    amihud = (r / tail.replace(0.0, np.nan)).mean() * 1e6
    return {"adv_usd": float(tail.median()), "amihud": float(amihud)}


def days_to_liquidate(position_usd: float, adv_usd: float, participation: float) -> float:
    if not adv_usd or np.isnan(adv_usd) or participation <= 0:
        return float("nan")
    return float(abs(position_usd) / (participation * adv_usd))
