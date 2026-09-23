"""Retornos, riesgo y retorno ajustado por riesgo, por activo. Funciones puras.

Formulas, declaradas una vez aqui y usadas igual en todo el motor:

- Retorno de ventana: prod(1 + r) - 1 sobre las sesiones de la ventana.
- Anualizado a N anos: (1 + R)^(1/N) - 1, con N en anos naturales (dias/365.25).
- Volatilidad anual: std(r diario, ddof=1) x sqrt(252).
- Volatilidad a la baja: sqrt(mean(min(r - rf_d, 0)^2)) x sqrt(252). Se promedia
  sobre TODAS las sesiones, no solo las negativas: es la semidesviacion respecto
  al objetivo, la que usa Sortino. Promediar solo las negativas la infla.
- Sharpe: mean(r - rf_d) / std(r - rf_d) x sqrt(252). Media aritmetica: es la
  que corresponde a la varianza del denominador.
- Sortino: mean(r - rf_d) x 252 / volatilidad a la baja.
- Calmar: CAGR / |maximo drawdown|.
- Treynor: (retorno anual - rf) / beta.
- rf_d = (1 + rf)^(1/252) - 1, por composicion.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..data.cleaning import annualize_return, compound, daily_rf

TD = 252
WINDOWS_D = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}


def trailing_returns(close: pd.Series) -> dict[str, float]:
    """Retornos de ventana hasta el ultimo cierre disponible."""
    s = close.dropna()
    out: dict[str, float] = {}
    if len(s) < 2:
        return {k: float("nan") for k in (*WINDOWS_D, "YTD", "3Y_ann", "5Y_ann")}
    last = s.iloc[-1]
    for label, n in WINDOWS_D.items():
        out[label] = float(last / s.iloc[-n - 1] - 1.0) if len(s) > n else float("nan")

    year_start = s[s.index < pd.Timestamp(year=s.index[-1].year, month=1, day=1)]
    out["YTD"] = float(last / year_start.iloc[-1] - 1.0) if len(year_start) else float("nan")

    for label, years in (("3Y_ann", 3), ("5Y_ann", 5)):
        cutoff = s.index[-1] - pd.DateOffset(years=years)
        base = s[s.index <= cutoff]
        if len(base) and (s.index[0] <= cutoff + pd.Timedelta(days=7)):
            span_years = (s.index[-1] - base.index[-1]).days / 365.25
            out[label] = annualize_return(float(last / base.iloc[-1] - 1.0), span_years)
        else:
            out[label] = float("nan")
    return out


def drawdown_stats(returns: pd.Series) -> dict[str, float]:
    """Maximo, medio, duracion maxima bajo el agua y tiempo de recuperacion.

    - Drawdown medio: profundidad media de los EPISODIOS (de pico a
      recuperacion), no la media diaria de la serie de drawdown, que pesa mas
      los episodios largos.
    - Recuperacion: sesiones desde el minimo del peor drawdown hasta volver al
      pico anterior. NaN si no se ha recuperado.
    """
    r = returns.dropna()
    if len(r) < 2:
        return {"max_dd": float("nan"), "avg_dd": float("nan"),
                "max_dd_duration_d": float("nan"), "recovery_d": float("nan")}
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0

    underwater = dd < 0
    episode = (~underwater).cumsum()
    depths = dd[underwater].groupby(episode[underwater]).min()
    lengths = underwater[underwater].groupby(episode[underwater]).size()

    trough = dd.idxmin()
    after = wealth.loc[trough:]
    recovered = after[after >= peak.loc[trough]]
    recovery = float(len(wealth.loc[trough:recovered.index[0]]) - 1) if len(recovered) else float("nan")

    return {
        "max_dd": float(dd.min()),
        "avg_dd": float(depths.mean()) if len(depths) else 0.0,
        "max_dd_duration_d": float(lengths.max()) if len(lengths) else 0.0,
        "recovery_d": recovery,
    }


def risk_metrics(returns: pd.Series, rf_annual: float) -> dict[str, float]:
    r = returns.dropna()
    if len(r) < 20:
        return {}
    rf_d = daily_rf(rf_annual)
    excess = r - rf_d
    downside = math.sqrt(float((np.minimum(excess, 0.0) ** 2).mean())) * math.sqrt(TD)
    out = {
        "vol_daily": float(r.std(ddof=1)),
        "vol_annual": float(r.std(ddof=1) * math.sqrt(TD)),
        "downside_vol": downside,
        "skewness": float(r.skew()),
        "kurtosis": float(r.kurt()),  # exceso de curtosis (normal = 0)
    }
    out.update(drawdown_stats(r))
    return out


def risk_adjusted(
    returns: pd.Series,
    rf_annual: float,
    *,
    beta: float | None = None,
    benchmark: pd.Series | None = None,
) -> dict[str, float]:
    r = returns.dropna()
    if len(r) < 20:
        return {}
    rf_d = daily_rf(rf_annual)
    excess = r - rf_d
    years = len(r) / TD
    cagr = annualize_return(float(compound(r)), years)
    std = float(excess.std(ddof=1))
    sharpe = float(excess.mean() / std * math.sqrt(TD)) if std > 0 else float("nan")

    downside = math.sqrt(float((np.minimum(excess, 0.0) ** 2).mean())) * math.sqrt(TD)
    sortino = float(excess.mean() * TD / downside) if downside > 0 else float("nan")

    dd = drawdown_stats(r)["max_dd"]
    calmar = float(cagr / abs(dd)) if dd and not np.isnan(dd) and dd < 0 else float("nan")

    treynor = float((cagr - rf_annual) / beta) if beta not in (None, 0) and not np.isnan(beta) else float("nan")

    ir = float("nan")
    if benchmark is not None:
        active = (r - benchmark.reindex(r.index)).dropna()
        te = float(active.std(ddof=1) * math.sqrt(TD))
        ir = float(active.mean() * TD / te) if te > 0 else float("nan")

    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino,
            "calmar": calmar, "treynor": treynor, "information_ratio": ir}
