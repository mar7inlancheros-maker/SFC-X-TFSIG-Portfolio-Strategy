"""Metricas de un backtest long/short.

Reutiliza las de `sfc_tfsig.metrics` (CAGR, Sharpe, drawdown) y el VaR de
`sfc_tfsig.risk.var`, para que un Sharpe signifique lo mismo en los dos motores
del repo. Anade las de operativa.

**Tasa de acierto, dos definiciones, las dos reportadas:**

- por PERIODO: fraccion de periodos entre rebalanceos con retorno positivo;
- por POSICION: fraccion de posiciones-periodo cuya contribucion fue positiva.
  En long/short importa mas: dice si los cortos bajan y los largos suben, no
  solo si la cartera gano.

Factor de beneficio = suma de periodos ganadores / |suma de perdedores|.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sfc_tfsig import metrics as sm
from sfc_tfsig.risk.var import historical_es, historical_var

from .engine import BacktestResult

TD = 252


def summary(result: BacktestResult, benchmark_nav: pd.Series | None) -> dict[str, float]:
    nav = result.nav
    if len(nav) < 20:
        return {}
    rf = result.config.risk_free_rate
    r = nav.pct_change().dropna()
    years = sm.years_elapsed(nav)

    out: dict[str, float] = {
        "start": nav.index[0],
        "end": nav.index[-1],
        "years": years,
        "cumulative_return": sm.total_return(nav),
        "cagr": sm.cagr(nav),
        "volatility": sm.volatility(nav),
        "downside_deviation": float(math.sqrt((np.minimum(r, 0.0) ** 2).mean()) * math.sqrt(TD)),
        "max_drawdown": sm.max_drawdown(nav),
        "sharpe": sm.sharpe(nav, rf),
        "sortino": sm.sortino(nav, rf),
        "calmar": sm.calmar(nav),
        "var_95_1d": historical_var(r, 0.95),
        "cvar_95_1d": historical_es(r, 0.95),
        "var_99_1d": historical_var(r, 0.99),
        "cvar_99_1d": historical_es(r, 0.99),
    }
    if benchmark_nav is not None:
        bench = benchmark_nav.reindex(nav.index).ffill()
        out["information_ratio"] = sm.information_ratio(nav, bench)
        beta, alpha = sm.beta_alpha(nav, bench, rf)
        out["beta_realized"], out["alpha_annual"] = beta, alpha
        out["correlation_to_benchmark"] = float(r.corr(bench.pct_change().reindex(r.index)))

    reb = result.rebalances
    if not reb.empty:
        out["rebalances"] = float(len(reb))
        out["turnover_annual"] = float(reb["turnover"].sum() / years) if years else float("nan")
        out["cost_drag_annual"] = float(reb["cost_bps"].sum() / 1e4 / years) if years else float("nan")
        out["total_costs"] = float(reb["cost"].sum())
        out["avg_gross"] = float(reb["gross"].mean())
        out["avg_net"] = float(reb["net"].mean())
        out["avg_beta_ex_ante"] = float(reb["beta_ex_ante"].mean())
    out["n_trades"] = float(len(result.trades))

    periods = _period_returns(result)
    if len(periods):
        wins, losses = periods[periods > 0], periods[periods < 0]
        out["win_rate_periods"] = float((periods > 0).mean())
        out["avg_win"] = float(wins.mean()) if len(wins) else float("nan")
        out["avg_loss"] = float(losses.mean()) if len(losses) else float("nan")
        out["profit_factor"] = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf")
    c = result.contributions.drop(columns=["start", "end"], errors="ignore")
    if not c.empty:
        values = c.to_numpy(float).ravel()
        values = values[~np.isnan(values) & (values != 0)]
        out["win_rate_positions"] = float((values > 0).mean()) if len(values) else float("nan")
    return out


def _period_returns(result: BacktestResult) -> pd.Series:
    nav = result.nav
    if result.rebalances.empty:
        return pd.Series(dtype=float)
    marks = [d for d in result.rebalances["execution_date"] if d in nav.index] + [nav.index[-1]]
    marks = sorted(dict.fromkeys(marks))
    values = nav.loc[marks]
    return values.pct_change().dropna()


def annual_returns(nav: pd.Series, benchmark_nav: pd.Series | None = None) -> pd.DataFrame:
    return sm.yearly_returns(nav, benchmark_nav)


def monthly_returns(nav: pd.Series) -> pd.DataFrame:
    """Tabla ano x mes."""
    m = nav.resample("ME").last().pct_change().dropna()
    table = m.to_frame("r")
    table["year"], table["month"] = table.index.year, table.index.month
    return table.pivot(index="year", columns="month", values="r")
