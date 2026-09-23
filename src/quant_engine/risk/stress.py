"""Pruebas de estres. Ningun resultado se inventa: todo sale de los datos.

**Por que no se reutiliza `sfc_tfsig.risk.stress`:** aquel corta el choque de un
nombre en -100%, correcto para un largo (no puede perder mas que lo invertido) y
FALSO para un corto, que pierde sin techo si el precio sube. Aqui los retornos
se aplican con signo y sin corte.

1. **Episodios historicos**: los peores drawdowns del BENCHMARK dentro de los
   datos, detectados por algoritmo (de pico a minimo), no una lista fija. Mas
   crisis con nombre (COVID, 2022...) solo si caen dentro del rango descargado.
   Pesos de hoy, precios de entonces: es "que le pasaria a esta cartera", no
   "que le paso".
2. **Choque de volatilidad**: todas las volatilidades x k, correlaciones iguales.
3. **Choque de correlacion**: (a) todo se correlaciona -- rho' = rho + 0,5 (1 - rho);
   (b) la cobertura se rompe -- correlacion entre largos y cortos a cero.
4. **Choque de mercado**: caida instantanea del benchmark x beta de cartera,
   con la beta de hoy y con una beta ADVERSA: cada largo con el percentil 95 de
   su beta movil de 252 dias y cada corto con el percentil 5 (un corto de beta
   baja cubre menos). Es un peor caso conjunto, no un escenario probable.
5. **Liquidez**: dias para deshacer cada posicion a la participacion del YAML, y
   coste de salida forzosa con el coste de transaccion multiplicado.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

NAMED_EPISODES = {
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
    "Q4 2018 selloff": ("2018-09-20", "2018-12-24"),
    "Aug 2024 vol spike": ("2024-07-16", "2024-08-05"),
    "Apr 2025 tariff shock": ("2025-02-19", "2025-04-08"),
}


def benchmark_episodes(bench_returns: pd.Series, top: int = 3, min_depth: float = 0.07) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    wealth = (1.0 + bench_returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    episodes = []
    underwater = dd < 0
    group = (~underwater).cumsum()
    for _, block in dd[underwater].groupby(group[underwater]):
        trough = block.idxmin()
        if block.min() > -min_depth:
            continue
        start = wealth.loc[:trough].idxmax()
        episodes.append((float(block.min()), start, trough))
    episodes.sort()
    return [(f"Benchmark drawdown {d:.0%}", s, t) for d, s, t in episodes[:top]]


def _hold_return(returns: pd.DataFrame, start, end) -> pd.Series:
    """Retorno de comprar y mantener cada activo en la ventana: prod(1 + r) - 1."""
    block = returns.loc[pd.Timestamp(start):pd.Timestamp(end)]
    return (1.0 + block.fillna(0.0)).prod() - 1.0 if len(block) else pd.Series(np.nan, index=returns.columns)


def historical_scenarios(weights: pd.Series, returns: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Pesos de hoy, MANTENIDOS sin rebalancear durante cada episodio.

    P&L de la cartera = suma de w_i x (prod(1 + r_i) - 1). Es exacto para una
    posicion que se mantiene: un corto con w < 0 gana lo que cae su activo. La
    version anterior componia los retornos diarios NEGADOS de la cesta corta,
    que es un ETF inverso rebalanceado a diario, no un corto: en 2022 daba una
    pata corta de +54,8% y una cartera que no cuadraba con sus patas.
    """
    names = list(weights.index)
    first, last = returns.index.min(), returns.index.max()
    windows = [(name, pd.Timestamp(s), pd.Timestamp(e)) for name, (s, e) in NAMED_EPISODES.items()
               if pd.Timestamp(s) >= first and pd.Timestamp(e) <= last]
    windows += benchmark_episodes(returns[benchmark])
    long_w, short_w = weights[weights > 0], weights[weights < 0]
    rows, seen = [], set()
    for name, s, e in windows:
        key = (s.date(), e.date())
        if key in seen:
            continue
        seen.add(key)
        held = _hold_return(returns[names + [benchmark]], s, e)
        rows.append({
            "scenario": name, "start": s.date(), "end": e.date(),
            "portfolio": float((weights * held[names]).sum()),
            "benchmark": float(held[benchmark]),
            # Contribucion de cada pata al P&L de la cartera, en fraccion del capital.
            "long_leg_pnl": float((long_w * held[long_w.index]).sum()),
            "short_leg_pnl": float((short_w * held[short_w.index]).sum()),
        })
    return pd.DataFrame(rows)


def _vol(w: np.ndarray, cov: np.ndarray) -> float:
    return math.sqrt(max(float(w @ cov @ w), 0.0) * 252)


def covariance_shocks(weights: pd.Series, cov: pd.DataFrame, signs: pd.Series,
                      multipliers: list[float]) -> pd.DataFrame:
    names = list(weights.index)
    w = weights.to_numpy(float)
    sigma = cov.loc[names, names].to_numpy(float)
    sd = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(sd, sd)
    base = _vol(w, sigma)
    rows = [{"shock": "base", "vol_annual": base, "var_95_1d_normal": 1.645 * base / math.sqrt(252)}]

    for k in multipliers:
        v = _vol(w, sigma * k * k)
        rows.append({"shock": f"volatility x{k}", "vol_annual": v, "var_95_1d_normal": 1.645 * v / math.sqrt(252)})

    stressed = corr + 0.5 * (1.0 - corr)
    np.fill_diagonal(stressed, 1.0)
    v = _vol(w, stressed * np.outer(sd, sd))
    rows.append({"shock": "correlation -> 1 (half way)", "vol_annual": v, "var_95_1d_normal": 1.645 * v / math.sqrt(252)})

    s = signs.reindex(names).to_numpy(float)
    broken = corr.copy()
    cross = np.outer(s > 0, s < 0) | np.outer(s < 0, s > 0)
    broken[cross] = 0.0
    v = _vol(w, broken * np.outer(sd, sd))
    rows.append({"shock": "hedge breakdown (long-short corr = 0)", "vol_annual": v, "var_95_1d_normal": 1.645 * v / math.sqrt(252)})
    return pd.DataFrame(rows)


def market_shocks(portfolio_beta: float, worst_beta: float, shocks: list[float]) -> pd.DataFrame:
    rows = []
    for shock in shocks:
        rows.append({"market_move": shock, "pnl_current_beta": portfolio_beta * shock,
                     "pnl_adverse_beta": worst_beta * shock})
    return pd.DataFrame(rows)


def liquidity_stress(weights: pd.Series, capital: float, adv: pd.Series,
                     participation: float, tc: float, spread_multiplier: float) -> tuple[pd.DataFrame, float]:
    rows = []
    for ticker, w in weights.items():
        position = abs(w) * capital
        a = adv.get(ticker, np.nan)
        days = position / (participation * a) if a and not np.isnan(a) else float("nan")
        rows.append({"ticker": ticker, "position_usd": position, "adv_usd": a, "days_to_exit": days})
    exit_cost = float(weights.abs().sum() * tc * spread_multiplier)
    return pd.DataFrame(rows).set_index("ticker"), exit_cost
