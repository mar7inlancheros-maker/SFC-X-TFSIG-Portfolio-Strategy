"""Riesgo ex-ante de una cartera con signo. Funciones puras.

- Volatilidad ex-ante anual: sqrt(w' Sigma w x 252), Sigma diaria.
- Contribucion al riesgo de i: w_i (Sigma w)_i / (w' Sigma w). Suman 1. En una
  cartera long/short pueden ser NEGATIVAS: un corto que cubre a los largos
  resta riesgo, y eso es exactamente lo que se le pide.
- Beta de cartera: suma de w_i beta_i.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .constraints import exposures


def ex_ante(weights: pd.Series, cov: pd.DataFrame, betas: pd.Series | None = None) -> dict[str, object]:
    names = [t for t in weights.index if t in cov.index]
    w = weights.reindex(names).fillna(0.0).to_numpy(float)
    sigma = cov.loc[names, names].to_numpy(float)
    var = float(w @ sigma @ w)
    marginal = sigma @ w
    contrib = pd.Series(w * marginal / var if var > 0 else np.nan, index=names)
    out: dict[str, object] = exposures(weights)
    out["vol_annual"] = math.sqrt(var * 252) if var > 0 else float("nan")
    out["risk_contribution"] = contrib
    if betas is not None:
        out["beta"] = float((weights * betas.reindex(weights.index)).sum())
    return out


def static_returns(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """Retorno diario de mantener los pesos CONSTANTES (rebalanceo diario implicito).

    Solo para medidas de riesgo de la cartera tal como es hoy (VaR, estres,
    correlaciones). El rendimiento de la cartera se mide en el backtest, que
    deja derivar los pesos y cobra costes: esta serie regala el rebalanceo.
    """
    cols = [t for t in weights.index if t in returns.columns]
    return (returns[cols].fillna(0.0) * weights[cols]).sum(axis=1)
