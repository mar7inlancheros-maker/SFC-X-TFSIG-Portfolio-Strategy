"""Backtest long/short walk-forward. Un solo motor para los dos modos.

**MODO A -- instantanea.** La cesta LONG/SHORT de hoy se aplica a todo el
historico. Los PESOS se estiman walk-forward (solo con datos anteriores a cada
rebalanceo), pero la SELECCION de nombres es de hoy: el research ya sabia que
NVDA subio. El resultado es una trayectoria HIPOTETICA de "que habria pasado
con esta cesta", no un backtest del proceso de research. El reporte lo dice
en la cabecera de la seccion, no en una nota al pie.

**MODO B -- senales historicas.** Un CSV `date,ticker,signal` con las
recomendaciones tal como se emitieron. En cada rebalanceo se usa la ultima
senal de cada ticker con fecha <= la del rebalanceo. Esto SI es un backtest.

**Contabilidad en dolares, por dia:**

    posicion_i,t = posicion_i,t-1 x (1 + r_i,t)   (un corto es una posicion negativa)
    caja_t       = caja_t-1 x (1 + rf_d) - prestamo_d x |cortos|
    NAV_t        = caja_t + suma posiciones

Al vender en corto, el producto entra en caja y rinde rf; el coste de prestamo
se cobra aparte sobre el nocional corto. Una cartera 100/100 rinde asi la tasa
libre de riesgo mas el spread largo-corto menos el prestamo, que es como
funciona en la practica.

**Secuencia de rebalanceo:** senal y pesos con datos hasta el cierre de `d`;
ejecucion al cierre de la sesion siguiente. Coste = tc x nocional operado.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..data.cleaning import daily_rf
from ..factors.beta import rolling_beta
from ..inference.covariance import ewma, for_optimization
from ..portfolio.construction import build, shrink_means
from ..portfolio.constraints import ConstructionParams

log = logging.getLogger(__name__)

SignalSchedule = Callable[[pd.Timestamp], pd.Series | None]


@dataclass(frozen=True)
class BacktestConfig:
    method: str
    rebalance: str
    params: ConstructionParams
    capital: float
    transaction_cost: float
    borrow_cost: float
    risk_free_rate: float
    estimation_window_d: int = 252
    vol_estimator: str = "sample"   # sample | ewma
    ewma_lambda: float = 0.94
    min_estimation_obs: int = 126


@dataclass
class BacktestResult:
    nav: pd.Series
    returns: pd.Series
    rebalances: pd.DataFrame
    weights: pd.DataFrame           # fecha de ejecucion x ticker
    trades: pd.DataFrame
    contributions: pd.DataFrame     # periodo x ticker: P&L / NAV inicial del periodo
    config: BacktestConfig
    mode: str
    warnings: list[str] = field(default_factory=list)


def rebalance_dates(index: pd.DatetimeIndex, freq: str, start: pd.Timestamp) -> list[pd.Timestamp]:
    """Ultima sesion de cada mes o trimestre, desde `start`."""
    rule = "ME" if freq == "monthly" else "QE"
    s = pd.Series(index, index=index)
    last = s.groupby(pd.Grouper(freq=rule)).last().dropna()
    return [d for d in last if d >= start]


def constant_schedule(signs: pd.Series) -> SignalSchedule:
    return lambda _date: signs


def csv_schedule(path) -> SignalSchedule:
    """Senales historicas: LONG, SHORT, o FLAT/CLOSE para salir."""
    df = pd.read_csv(path)
    missing = {"date", "ticker", "signal"} - set(df.columns)
    if missing:
        raise ValueError(f"el CSV de senales necesita columnas date,ticker,signal (faltan {missing})")
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["signal"] = df["signal"].astype(str).str.upper().str.strip()
    bad = set(df["signal"]) - {"LONG", "SHORT", "FLAT", "CLOSE", "BUY", "SELL"}
    if bad:
        raise ValueError(f"senales no reconocidas: {sorted(bad)}")
    mapping = {"LONG": 1, "BUY": 1, "SHORT": -1, "SELL": -1, "FLAT": 0, "CLOSE": 0}
    df["sign"] = df["signal"].map(mapping)
    df = df.sort_values("date")

    def schedule(date: pd.Timestamp) -> pd.Series | None:
        visible = df[df["date"] <= date]
        if visible.empty:
            return None
        latest = visible.groupby("ticker")["sign"].last()
        latest = latest[latest != 0]
        return latest.astype(float) if len(latest) else None

    schedule.tickers = sorted(df["ticker"].unique())  # type: ignore[attr-defined]
    schedule.first_date = df["date"].min()            # type: ignore[attr-defined]
    return schedule


def _estimate(returns: pd.DataFrame, names: list[str], benchmark: str,
              cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    window = returns.iloc[-cfg.estimation_window_d:]
    usable = [t for t in names if window[t].notna().sum() >= cfg.min_estimation_obs]
    block = window[usable].dropna()
    if cfg.vol_estimator == "ewma":
        cov = ewma(block, cfg.ewma_lambda)
    else:
        cov, _ = for_optimization(block)
    vol = pd.Series(np.sqrt(np.diag(cov.to_numpy()) * 252), index=cov.index)
    mu = shrink_means(block)
    betas = pd.Series({
        t: float(rolling_beta(window[t], window[benchmark], len(window)).dropna().iloc[-1])
        if window[[t, benchmark]].dropna().shape[0] > 60 else 1.0
        for t in usable
    })
    return cov, vol, mu, betas, usable


def run(
    returns: pd.DataFrame,
    benchmark: str,
    schedule: SignalSchedule,
    cfg: BacktestConfig,
    *,
    start: pd.Timestamp,
    mode: str,
) -> BacktestResult:
    sessions = returns.index[returns.index >= start]
    signal_days = rebalance_dates(returns.index, cfg.rebalance, start)
    exec_day = {}
    for d in signal_days:
        after = returns.index[returns.index > d]
        if len(after):
            exec_day[after[0]] = d

    rf_d = daily_rf(cfg.risk_free_rate)
    borrow_d = daily_rf(cfg.borrow_cost)
    cash = cfg.capital
    pos = pd.Series(dtype=float)
    warnings: list[str] = []
    nav_points, rebal_rows, weight_rows, trade_rows, contrib_rows = [], [], {}, [], []
    period_start_nav, period_start_pos, period_start_date = None, None, None
    missing_returns = 0

    for t in sessions:
        # 1) retornos del dia sobre las posiciones que vienen de ayer
        if len(pos):
            r = returns.loc[t, pos.index]
            missing_returns += int(r.isna().sum())
            pos = pos * (1.0 + r.fillna(0.0))
        shorts = float(-pos[pos < 0].sum()) if len(pos) else 0.0
        cash = cash * (1.0 + rf_d) - borrow_d * shorts

        # 2) rebalanceo al cierre si hoy toca ejecutar
        if t in exec_day:
            d = exec_day[t]
            nav = cash + float(pos.sum())
            if period_start_nav is not None and period_start_pos is not None and len(period_start_pos):
                pnl = (pos.reindex(period_start_pos.index).fillna(0.0) - period_start_pos)
                contrib_rows.append({"start": period_start_date, "end": t,
                                     **(pnl / period_start_nav).to_dict()})

            signs = schedule(d)
            if signs is None or (signs > 0).sum() == 0 or (signs < 0).sum() == 0:
                warnings.append(f"{d.date()}: sin largos y cortos a la vez; se mantiene la cartera")
            else:
                hist = returns.loc[:d]
                names = [n for n in signs.index if n in returns.columns]
                cov, vol, mu, betas, usable = _estimate(hist, names, benchmark, cfg)
                dropped = sorted(set(names) - set(usable))
                if dropped:
                    warnings.append(f"{d.date()}: sin historia suficiente para estimar {dropped}")
                s = signs.reindex(usable).dropna()
                if (s > 0).sum() and (s < 0).sum():
                    try:
                        built = build(cfg.method, s, cov=cov, vol=vol, params=cfg.params, mu=mu, betas=betas)
                    except ValueError as exc:
                        built = None
                        warnings.append(f"{d.date()}: {exc}")
                    if built is not None and built.status == "ok":
                        target = built.weights * nav
                        all_names = pos.index.union(target.index)
                        current = pos.reindex(all_names).fillna(0.0)
                        target = target.reindex(all_names).fillna(0.0)
                        traded = (target - current).abs()
                        cost = float(traded.sum() * cfg.transaction_cost)
                        cash = cash - float((target - current).sum()) - cost
                        pos = target[target != 0.0]
                        rebal_rows.append({
                            "signal_date": d, "execution_date": t, "nav": nav,
                            "turnover": float(traded.sum() / nav / 2.0),
                            "cost": cost, "cost_bps": cost / nav * 1e4,
                            "gross": float(pos.abs().sum() / nav),
                            "net": float(pos.sum() / nav),
                            "beta_ex_ante": float((built.weights * betas.reindex(built.weights.index)).sum()),
                            "n_long": int((pos > 0).sum()), "n_short": int((pos < 0).sum()),
                        })
                        weight_rows[t] = built.weights
                        for name, delta in (target - current).items():
                            if abs(delta) > 1e-9:
                                trade_rows.append({"date": t, "ticker": name, "notional": float(delta),
                                                   "side": "BUY" if delta > 0 else "SELL"})
                    elif built is not None:
                        warnings.append(f"{d.date()}: {built.note}")
            period_start_nav = cash + float(pos.sum())
            period_start_pos = pos.copy()
            period_start_date = t

        nav_points.append((t, cash + float(pos.sum())))

    if period_start_pos is not None and len(period_start_pos):
        pnl = pos.reindex(period_start_pos.index).fillna(0.0) - period_start_pos
        contrib_rows.append({"start": period_start_date, "end": sessions[-1],
                             **(pnl / period_start_nav).to_dict()})
    if missing_returns:
        warnings.append(f"{missing_returns} retornos diarios faltantes se trataron como 0 en posiciones abiertas")

    nav = pd.Series(dict(nav_points)).sort_index()
    first_exec = min(exec_day) if exec_day else None
    if first_exec is not None:
        nav = nav.loc[nav.index >= first_exec]
    return BacktestResult(
        nav=nav,
        returns=nav.pct_change().dropna(),
        rebalances=pd.DataFrame(rebal_rows),
        weights=pd.DataFrame(weight_rows).T.fillna(0.0),
        trades=pd.DataFrame(trade_rows),
        contributions=pd.DataFrame(contrib_rows),
        config=cfg,
        mode=mode,
        warnings=warnings,
    )
