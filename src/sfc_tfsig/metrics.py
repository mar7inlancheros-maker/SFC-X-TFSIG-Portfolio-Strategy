"""Metricas de rendimiento y riesgo. Funciones puras sobre series de NAV.

El vocabulario es el del comite y el de una carta a inversores: CAGR, Sharpe,
maximo drawdown, tracking error, information ratio. Un backtest que solo reporta
retorno total no dice si el resultado es repetible ni si alguien habria aguantado
el camino.

**Convenios, declarados para que nadie los adivine:**

- El CAGR se calcula sobre anos de 365.25 dias naturales, no sobre numero de
  observaciones: asi no cambia segun se muestree diario o mensual.
- El Sharpe usa la tasa libre de riesgo del TOML, restada del retorno. Un Sharpe
  calculado sin restarla -- error comun -- infla el resultado justo cuando los
  tipos estan altos, que es cuando mas importa la comparacion.
- El maximo drawdown se mide sobre la serie diaria de NAV, no sobre la mensual.
  El drawdown mensual siempre sale menor y no es el que se sufre.
- Sortino penaliza solo la desviacion a la baja: la volatilidad al alza no es
  riesgo para un fondo long-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DAYS_PER_YEAR = 365.25


def _to_returns(nav: pd.Series) -> pd.Series:
    return nav.astype(float).pct_change().dropna()


def years_elapsed(nav: pd.Series) -> float:
    if len(nav) < 2:
        return 0.0
    span = (nav.index[-1] - nav.index[0]).days
    return max(span / DAYS_PER_YEAR, 1e-9)


def total_return(nav: pd.Series) -> float:
    if len(nav) < 2 or nav.iloc[0] <= 0:
        return float("nan")
    return float(nav.iloc[-1] / nav.iloc[0] - 1.0)


def cagr(nav: pd.Series) -> float:
    years = years_elapsed(nav)
    if years <= 0 or len(nav) < 2 or nav.iloc[0] <= 0 or nav.iloc[-1] <= 0:
        return float("nan")
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)


def volatility(nav: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    returns = _to_returns(nav)
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe(nav: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Sharpe anualizado con la tasa libre de riesgo restada periodo a periodo."""
    returns = _to_returns(nav)
    if len(returns) < 2:
        return float("nan")
    rf_period = (1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_period
    std = excess.std(ddof=1)
    if std == 0 or pd.isna(std):
        return float("nan")
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def sortino(nav: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    returns = _to_returns(nav)
    if len(returns) < 2:
        return float("nan")
    rf_period = (1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_period
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    dd = downside.std(ddof=1)
    if dd == 0 or pd.isna(dd):
        return float("nan")
    return float(excess.mean() / dd * math.sqrt(periods_per_year))


def drawdown_series(nav: pd.Series) -> pd.Series:
    running_max = nav.cummax()
    return nav / running_max - 1.0


def max_drawdown(nav: pd.Series) -> float:
    if len(nav) < 2:
        return float("nan")
    return float(drawdown_series(nav).min())


def drawdown_detail(nav: pd.Series) -> dict[str, object]:
    """Peor caida: cuanto, cuando empezo, cuando toco fondo y si se recupero.

    La fecha de recuperacion es la que el comite pregunta de verdad: "cuanto
    tiempo estuvimos bajo el agua".
    """
    if len(nav) < 2:
        return {"max_drawdown": float("nan"), "peak": None, "trough": None,
                "recovery": None, "underwater_days": None}
    dd = drawdown_series(nav)
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    after = nav.loc[trough:]
    recovered = after[after >= nav.loc[peak]]
    recovery = recovered.index[0] if len(recovered) else None
    underwater = ((recovery or nav.index[-1]) - peak).days
    return {
        "max_drawdown": float(dd.min()),
        "peak": peak,
        "trough": trough,
        "recovery": recovery,
        "underwater_days": int(underwater),
    }


def calmar(nav: pd.Series) -> float:
    mdd = max_drawdown(nav)
    if not mdd or pd.isna(mdd) or mdd == 0:
        return float("nan")
    return float(cagr(nav) / abs(mdd))


def hit_rate(nav: pd.Series, freq: str = "ME") -> float:
    """Fraccion de periodos con retorno positivo."""
    resampled = nav.resample(freq).last()
    returns = resampled.pct_change().dropna()
    if returns.empty:
        return float("nan")
    return float((returns > 0).mean())


def beta_alpha(nav: pd.Series, benchmark_nav: pd.Series, risk_free: float = 0.0,
               periods_per_year: int = TRADING_DAYS) -> tuple[float, float]:
    """Beta y alfa anualizada frente al benchmark (regresion simple)."""
    joined = pd.concat([_to_returns(nav), _to_returns(benchmark_nav)], axis=1).dropna()
    if len(joined) < 10:
        return float("nan"), float("nan")
    joined.columns = ["portfolio", "benchmark"]
    rf_period = (1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0
    y = joined["portfolio"] - rf_period
    x = joined["benchmark"] - rf_period
    variance = x.var(ddof=1)
    if variance == 0 or pd.isna(variance):
        return float("nan"), float("nan")
    beta = float(x.cov(y) / variance)
    alpha_period = float(y.mean() - beta * x.mean())
    return beta, float((1.0 + alpha_period) ** periods_per_year - 1.0)


def tracking_error(nav: pd.Series, benchmark_nav: pd.Series,
                   periods_per_year: int = TRADING_DAYS) -> float:
    joined = pd.concat([_to_returns(nav), _to_returns(benchmark_nav)], axis=1).dropna()
    if len(joined) < 2:
        return float("nan")
    diff = joined.iloc[:, 0] - joined.iloc[:, 1]
    return float(diff.std(ddof=1) * math.sqrt(periods_per_year))


def information_ratio(nav: pd.Series, benchmark_nav: pd.Series,
                      periods_per_year: int = TRADING_DAYS) -> float:
    """Exceso sobre el benchmark por unidad de tracking error.

    Es la metrica honesta para una estrategia long-only: mide si la seleccion
    aporta algo mas alla de estar invertido en el mercado.
    """
    joined = pd.concat([_to_returns(nav), _to_returns(benchmark_nav)], axis=1).dropna()
    if len(joined) < 10:
        return float("nan")
    diff = joined.iloc[:, 0] - joined.iloc[:, 1]
    std = diff.std(ddof=1)
    if std == 0 or pd.isna(std):
        return float("nan")
    return float(diff.mean() / std * math.sqrt(periods_per_year))


@dataclass(frozen=True)
class Performance:
    """Ficha de rendimiento. Lo que va a la carta trimestral."""

    start: pd.Timestamp
    end: pd.Timestamp
    years: float
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    underwater_days: int | None
    calmar: float
    hit_rate_monthly: float
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float
    benchmark_cagr: float
    benchmark_max_drawdown: float
    excess_cagr: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(nav: pd.Series, benchmark_nav: pd.Series | None = None,
             risk_free: float = 0.0) -> Performance:
    """Ficha completa a partir de la serie diaria de NAV."""
    nav = nav.dropna().astype(float)
    if benchmark_nav is not None:
        benchmark_nav = benchmark_nav.reindex(nav.index).ffill().dropna()
        # Se reescala para que ambas series empiecen en el mismo punto: si no,
        # la comparacion de drawdowns es entre niveles distintos.
        if len(benchmark_nav) and benchmark_nav.iloc[0] != 0:
            benchmark_nav = benchmark_nav / benchmark_nav.iloc[0] * nav.iloc[0]

    beta = alpha = te = ir = bench_cagr = bench_mdd = float("nan")
    if benchmark_nav is not None and len(benchmark_nav) > 10:
        beta, alpha = beta_alpha(nav, benchmark_nav, risk_free)
        te = tracking_error(nav, benchmark_nav)
        ir = information_ratio(nav, benchmark_nav)
        bench_cagr = cagr(benchmark_nav)
        bench_mdd = max_drawdown(benchmark_nav)

    detail = drawdown_detail(nav)
    strategy_cagr = cagr(nav)

    return Performance(
        start=nav.index[0],
        end=nav.index[-1],
        years=years_elapsed(nav),
        total_return=total_return(nav),
        cagr=strategy_cagr,
        volatility=volatility(nav),
        sharpe=sharpe(nav, risk_free),
        sortino=sortino(nav, risk_free),
        max_drawdown=detail["max_drawdown"],
        underwater_days=detail["underwater_days"],
        calmar=calmar(nav),
        hit_rate_monthly=hit_rate(nav),
        beta=beta,
        alpha=alpha,
        tracking_error=te,
        information_ratio=ir,
        benchmark_cagr=bench_cagr,
        benchmark_max_drawdown=bench_mdd,
        excess_cagr=(strategy_cagr - bench_cagr) if not pd.isna(bench_cagr) else float("nan"),
    )


def yearly_returns(nav: pd.Series, benchmark_nav: pd.Series | None = None) -> pd.DataFrame:
    """Retorno ano a ano, frente al benchmark.

    Un CAGR de 12 anos esconde el ano en que se perdio el 30%. La tabla anual
    es lo que un LP mira antes que el CAGR.
    """
    annual = nav.resample("YE").last()
    first = pd.Series([nav.iloc[0]], index=[nav.index[0]])
    annual = pd.concat([first, annual]).pct_change().dropna()
    out = pd.DataFrame({"strategy": annual})
    if benchmark_nav is not None:
        bench = benchmark_nav.reindex(nav.index).ffill()
        bench_annual = bench.resample("YE").last()
        bench_first = pd.Series([bench.iloc[0]], index=[bench.index[0]])
        out["benchmark"] = pd.concat([bench_first, bench_annual]).pct_change().dropna()
        out["excess"] = out["strategy"] - out["benchmark"]
    out.index = out.index.year
    return out
