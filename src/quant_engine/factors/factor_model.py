"""Exposicion a factores por regresion de series temporales.

**Factores construidos con ETF reales**, como carteras largo-menos-referencia:

    Market        = benchmark - rf
    Size          = IWM - SPY       (pequenas menos grandes)
    Value         = IWD - IWF       (Russell 1000 Value menos Growth)
    Momentum      = MTUM - SPY
    Low volatility = USMV - SPY
    Quality       = QUAL - SPY

No son los factores academicos de Fama-French: son lo que se puede construir con
datos publicos, diarios y verificables. Si un ETF no tiene datos para el
periodo, el factor se declara NO DISPONIBLE. No se rellena ni se inventa.

**Multicolinealidad.** Los factores ETF estan correlacionados entre si (MTUM y
QUAL comparten muchos nombres). Se reporta el numero de condicion y el VIF de
cada factor: con VIF > 5, la beta individual de ese factor es poco fiable
aunque el R2 conjunto sea bueno.

Errores estandar HAC (Newey-West, 5 rezagos).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import _sm as sm
from ..data.cleaning import daily_rf


@dataclass
class FactorModelResult:
    exposures: pd.DataFrame          # ticker x factor: beta
    t_stats: pd.DataFrame            # ticker x factor
    r2: pd.Series
    available: list[str]
    unavailable: dict[str, str] = field(default_factory=dict)
    condition_number: float = float("nan")
    vif: pd.Series = field(default_factory=pd.Series)
    factor_returns: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_factor_returns(
    returns: pd.DataFrame,
    benchmark: str,
    proxies: dict[str, list[str]],
    rf_annual: float,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rf_d = daily_rf(rf_annual)
    factors = {"market": returns[benchmark] - rf_d}
    unavailable: dict[str, str] = {}
    for name, (long_etf, short_etf) in proxies.items():
        missing = [t for t in (long_etf, short_etf) if t not in returns.columns]
        if missing:
            unavailable[name] = f"sin datos de {', '.join(missing)}"
            continue
        spread = returns[long_etf] - returns[short_etf]
        if spread.notna().sum() < 252:
            unavailable[name] = f"historia insuficiente ({int(spread.notna().sum())} sesiones)"
            continue
        factors[name] = spread
    return pd.DataFrame(factors), unavailable


def _vif(x: pd.DataFrame) -> pd.Series:
    out = {}
    for col in x.columns:
        others = x.drop(columns=col)
        if others.empty:
            out[col] = 1.0
            continue
        r2 = sm.OLS(x[col], sm.add_constant(others)).fit().rsquared
        out[col] = float(1.0 / (1.0 - r2)) if r2 < 1 else float("inf")
    return pd.Series(out)


def factor_exposures(
    returns: pd.DataFrame,
    tickers: list[str] | tuple[str, ...],
    benchmark: str,
    proxies: dict[str, list[str]],
    rf_annual: float,
) -> FactorModelResult:
    factors, unavailable = build_factor_returns(returns, benchmark, proxies, rf_annual)
    factors = factors.dropna()
    rf_d = daily_rf(rf_annual)

    betas, tstats, r2 = {}, {}, {}
    for ticker in tickers:
        if ticker not in returns.columns:
            continue
        joined = pd.concat([returns[ticker] - rf_d, factors], axis=1).dropna()
        if len(joined) < 126:
            continue
        y = joined.iloc[:, 0]
        x = sm.add_constant(joined.iloc[:, 1:])
        fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        betas[ticker] = fit.params.drop("const")
        tstats[ticker] = fit.tvalues.drop("const")
        r2[ticker] = float(fit.rsquared)

    x_std = (factors - factors.mean()) / factors.std()
    cond = float(np.linalg.cond(x_std.to_numpy())) if len(x_std) > len(x_std.columns) else float("nan")

    return FactorModelResult(
        exposures=pd.DataFrame(betas).T,
        t_stats=pd.DataFrame(tstats).T,
        r2=pd.Series(r2),
        available=list(factors.columns),
        unavailable=unavailable,
        condition_number=cond,
        vif=_vif(factors) if len(factors.columns) > 1 else pd.Series(dtype=float),
        factor_returns=factors,
    )
