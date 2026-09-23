"""Beta, alfa y su estabilidad. Funciones puras.

Regresion sobre retornos EN EXCESO de la tasa libre de riesgo:

    (R_i - rf) = alpha + beta (R_m - rf) + e

Sin restar rf, el alfa absorbe (1 - beta) x rf: con tipos al 4% y beta 0,5, un
activo sin habilidad alguna mostraria 2 puntos de "alfa" al ano.

Los errores estandar del alfa son HAC (Newey-West, 5 rezagos): los residuos
diarios tienen heterocedasticidad y algo de autocorrelacion, y el error clasico
infla el t del alfa. Un alfa de 5% con t = 0,8 no es un alfa: es ruido.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import _sm as sm
from ..data.cleaning import daily_rf

TD = 252
ROLLING_WINDOWS = (60, 120, 252)


def regression(asset: pd.Series, market: pd.Series, rf_annual: float) -> dict[str, float]:
    joined = pd.concat([asset, market], axis=1, keys=["y", "x"]).dropna()
    if len(joined) < 60:
        return {}
    rf_d = daily_rf(rf_annual)
    y = joined["y"] - rf_d
    x = sm.add_constant(joined["x"] - rf_d)
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    alpha_d, beta = float(fit.params.iloc[0]), float(fit.params.iloc[1])

    active = joined["y"] - joined["x"]
    te = float(active.std(ddof=1) * math.sqrt(TD))
    return {
        "beta": beta,
        "beta_t": float(fit.tvalues.iloc[1]),
        "alpha_annual": float((1.0 + alpha_d) ** TD - 1.0),
        "alpha_t": float(fit.tvalues.iloc[0]),
        "alpha_p": float(fit.pvalues.iloc[0]),
        "r2": float(fit.rsquared),
        "correlation": float(joined["y"].corr(joined["x"])),
        "tracking_error": te,
        "information_ratio": float(active.mean() * TD / te) if te > 0 else float("nan"),
        "residual_vol": float(np.std(fit.resid, ddof=2) * math.sqrt(TD)),
    }


def rolling_beta(asset: pd.Series, market: pd.Series, window: int) -> pd.Series:
    """cov(r_i, r_m) / var(r_m) en ventana movil. Sin restar rf: en una ventana
    corta rf es practicamente constante y se cancela en la covarianza."""
    joined = pd.concat([asset, market], axis=1, keys=["y", "x"]).dropna()
    cov = joined["y"].rolling(window).cov(joined["x"])
    var = joined["x"].rolling(window).var()
    return (cov / var).rename(asset.name)


def beta_stability(asset: pd.Series, market: pd.Series) -> dict[str, float]:
    """Beta movil actual en tres ventanas y cuanto se ha movido la de 252 dias.

    `beta_252_range` = percentil 95 menos percentil 5 de la beta de 252 dias en
    toda la muestra. Por encima de ~0,5 la beta no es una propiedad estable del
    activo, y neutralizar la cartera con la beta de hoy protege poco.
    """
    out: dict[str, float] = {}
    for w in ROLLING_WINDOWS:
        rb = rolling_beta(asset, market, w).dropna()
        out[f"beta_{w}d"] = float(rb.iloc[-1]) if len(rb) else float("nan")
        if w == 252 and len(rb) > 20:
            out["beta_252_std"] = float(rb.std())
            out["beta_252_range"] = float(rb.quantile(0.95) - rb.quantile(0.05))
    return out
