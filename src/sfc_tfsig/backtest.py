"""Motor de backtest: ejecuta la estrategia mes a mes y lleva la contabilidad.

**Contabilidad en acciones, no en pesos.** Muchos backtests multiplican pesos por
retornos y llaman a eso una cartera. Eso supone un rebalanceo continuo y gratuito
a los pesos objetivo, lo que regala un rebalancing bonus que no existe. Aqui se
compran acciones a un precio, se mantienen, y la deriva de los pesos entre
rebalanceos es real: si una posicion sube un 40%, su peso sube, y solo vuelve al
objetivo en el siguiente rebalanceo, pagando el coste de volver.

**Secuencia de cada rebalanceo, en orden estricto:**

1. Fecha de senal `d`: se calcula el score con datos visibles en `d`.
2. Fecha de ejecucion `d + execution_lag_d` sesiones: se opera al CIERRE de ese
   dia. Nunca al cierre de `d`, que es el precio que el propio dato de senal ya
   conoce.
3. Se cobran comision, spread y slippage sobre el nocional operado, en ambos
   sentidos.

**Que NO hace este motor, dicho antes de que lo pregunte el comite:**

- No modela el impacto de mercado por tamano de orden. Con capital simulado de
  seis cifras en acciones que mueven millones al dia, el impacto es
  despreciable; con cien millones, no lo seria.
- No modela dividendos por separado: van dentro del precio ajustado.
- No modela huecos de apertura ni ordenes limitadas. Todo es a cierre.
- Permite fracciones de accion. En paper trading con IBKR es realista; con un
  broker que no las admita, las posiciones pequenas redondean a la baja.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .portfolio import build_portfolio, turnover


@dataclass
class BacktestResult:
    """Todo lo que produce una corrida. Se guarda entero para poder auditarlo."""

    nav: pd.Series
    holdings: pd.DataFrame
    trades: pd.DataFrame
    rebalances: pd.DataFrame
    config_fingerprint: str
    warnings: list[str] = field(default_factory=list)

    @property
    def total_costs(self) -> float:
        return float(self.trades["cost"].sum()) if not self.trades.empty else 0.0

    @property
    def average_turnover(self) -> float:
        if self.rebalances.empty:
            return float("nan")
        return float(self.rebalances["turnover"].mean())


def _execution_date(available: pd.DatetimeIndex, signal_date: pd.Timestamp, lag: int) -> pd.Timestamp | None:
    """Sesion de ejecucion: `lag` sesiones DESPUES de la fecha de senal."""
    future = available[available > signal_date]
    if len(future) == 0:
        return None
    index = min(lag - 1, len(future) - 1) if lag > 0 else 0
    if lag == 0:
        # Ejecutar el mismo dia de la senal solo es valido si la senal se
        # construyo con el cierre anterior. Se permite, pero se avisa.
        candidates = available[available <= signal_date]
        return candidates[-1] if len(candidates) else None
    return future[index]


def run_backtest(
    scored_panel: pd.DataFrame,
    close_wide: pd.DataFrame,
    cfg: Config,
    *,
    progress: bool = True,
) -> BacktestResult:
    """Corre la estrategia sobre el panel ya puntuado.

    `scored_panel` debe traer la columna `score_composite` (la pone
    `factors.composite.build_scores`). `close_wide` son los cierres ajustados
    diarios, fechas x tickers.
    """
    capital = float(cfg.get("backtest.initial_capital"))
    lag = int(cfg.get("calendar.execution_lag_d"))
    cost_rate = cfg.total_cost_bps / 10_000.0

    sessions = close_wide.index
    signal_dates = sorted(scored_panel["date"].unique())

    cash = capital
    shares = pd.Series(dtype="float64")       # ticker -> numero de acciones
    last_price = pd.Series(dtype="float64")   # ultimo precio conocido por ticker

    nav_points: list[tuple[pd.Timestamp, float]] = []
    holdings_rows: list[dict] = []
    trade_rows: list[dict] = []
    rebalance_rows: list[dict] = []
    warnings: list[str] = []

    previous_weights = pd.Series(dtype="float64")
    previous_execution: pd.Timestamp | None = None

    for i, signal_date in enumerate(signal_dates, start=1):
        execution_date = _execution_date(sessions, pd.Timestamp(signal_date), lag)
        if execution_date is None:
            warnings.append(
                f"{pd.Timestamp(signal_date).date()}: sin sesion posterior para ejecutar; "
                "el ultimo rebalanceo del periodo se omite"
            )
            break

        prices_now = close_wide.loc[execution_date]

        # -- NAV diario del tramo anterior --------------------------------
        if previous_execution is not None and len(shares):
            segment = close_wide.loc[
                (close_wide.index > previous_execution) & (close_wide.index <= execution_date)
            ]
            held_prices = segment.reindex(columns=shares.index).ffill()
            # Un ticker que deja de cotizar mantiene su ultimo precio conocido
            # hasta que se liquida en el siguiente rebalanceo. Es optimista (una
            # exclusion suele ser mala noticia) y forma parte del sesgo de
            # supervivencia ya declarado.
            held_prices = held_prices.fillna(last_price.reindex(shares.index))
            values = held_prices.mul(shares, axis=1).sum(axis=1)
            for date, value in values.items():
                nav_points.append((date, float(value + cash)))

        # -- valoracion en el momento de operar ---------------------------
        current_prices = prices_now.reindex(shares.index)
        missing = current_prices.isna()
        if missing.any():
            current_prices[missing] = last_price.reindex(current_prices.index)[missing]
            warnings.append(
                f"{execution_date.date()}: sin precio para "
                f"{', '.join(map(str, current_prices.index[missing]))}; "
                "se liquidan al ultimo precio conocido"
            )
        position_value = float((shares * current_prices).sum()) if len(shares) else 0.0
        nav_before = cash + position_value

        if nav_before <= 0:
            warnings.append(f"{execution_date.date()}: capital agotado; se detiene el backtest")
            break

        # -- cartera objetivo ---------------------------------------------
        cross_section = scored_panel[scored_panel["date"] == signal_date]
        held = set(shares.index[shares > 0])
        target = build_portfolio(cross_section, cfg, held=held)
        if target.empty:
            warnings.append(f"{pd.Timestamp(signal_date).date()}: sin candidatos; se mantiene la cartera")
            previous_execution = execution_date
            continue

        target_weights = target.set_index("ticker")["weight"]
        # Solo se puede comprar lo que tiene precio el dia de ejecucion.
        tradable = target_weights.index.intersection(prices_now.dropna().index)
        dropped = set(target_weights.index) - set(tradable)
        if dropped:
            warnings.append(
                f"{execution_date.date()}: sin precio de ejecucion para {sorted(dropped)}; "
                "se excluyen del rebalanceo"
            )
            target_weights = target_weights.loc[tradable]
            if target_weights.empty:
                previous_execution = execution_date
                continue
            target_weights = target_weights / target_weights.sum() * (
                1.0 - float(cfg.get("portfolio.cash_buffer"))
            )

        # -- ordenes --------------------------------------------------------
        all_names = shares.index.union(target_weights.index)
        exec_prices = prices_now.reindex(all_names)
        exec_prices = exec_prices.fillna(last_price.reindex(all_names))

        target_value = target_weights.reindex(all_names).fillna(0.0) * nav_before
        target_shares = (target_value / exec_prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        current_shares = shares.reindex(all_names).fillna(0.0)

        delta_shares = target_shares - current_shares
        traded_notional = (delta_shares.abs() * exec_prices).fillna(0.0)
        costs = traded_notional * cost_rate

        for ticker in all_names:
            if abs(delta_shares[ticker]) < 1e-9:
                continue
            trade_rows.append(
                {
                    "date": execution_date,
                    "ticker": ticker,
                    "side": "BUY" if delta_shares[ticker] > 0 else "SELL",
                    "shares": float(abs(delta_shares[ticker])),
                    "price": float(exec_prices[ticker]),
                    "notional": float(traded_notional[ticker]),
                    "cost": float(costs[ticker]),
                }
            )

        total_cost = float(costs.sum())
        cash = nav_before - float((target_shares * exec_prices).sum()) - total_cost
        shares = target_shares[target_shares > 1e-9]
        last_price = exec_prices.reindex(shares.index).combine_first(last_price)

        nav_after = cash + float((shares * exec_prices.reindex(shares.index)).sum())
        nav_points.append((execution_date, nav_after))

        realized_weights = (shares * exec_prices.reindex(shares.index)) / nav_after

        # Rotacion = nocional operado / NAV / 2. Se mide sobre lo que REALMENTE
        # se opero, no comparando la cartera objetivo de este mes con la del mes
        # pasado: entre rebalanceos los pesos derivan con los precios, asi que
        # dos objetivos identicos pueden exigir operaciones grandes para volver
        # a ellos. Medirlo de la otra forma daba rotacion cero en meses en los
        # que si se pago comision, y la cifra que ve el comite tiene que cuadrar
        # con los costes de la misma fila.
        traded_turnover = float(traded_notional.sum() / nav_before / 2.0)

        rebalance_rows.append(
            {
                "signal_date": pd.Timestamp(signal_date),
                "execution_date": execution_date,
                "n_positions": int(len(shares)),
                "turnover": traded_turnover,
                "cost": total_cost,
                "cost_bps_of_nav": total_cost / nav_before * 10_000.0,
                "nav": nav_after,
                "cash": cash,
            }
        )
        for ticker, weight in realized_weights.items():
            row = target[target["ticker"] == ticker]
            holdings_rows.append(
                {
                    "date": execution_date,
                    "ticker": ticker,
                    "shares": float(shares[ticker]),
                    "price": float(exec_prices[ticker]),
                    "weight": float(weight),
                    "sector": row["sector"].iloc[0] if len(row) and "sector" in row else None,
                    "score": float(row["score_composite"].iloc[0])
                    if len(row) and "score_composite" in row else np.nan,
                }
            )

        previous_weights = realized_weights
        previous_execution = execution_date

        if progress and (i % 12 == 0 or i == len(signal_dates)):
            print(f"  backtest: {i}/{len(signal_dates)} rebalanceos, NAV {nav_after:,.0f}", flush=True)

    # -- cola: valorar hasta el ultimo dia disponible ----------------------
    if previous_execution is not None and len(shares):
        tail = close_wide.loc[close_wide.index > previous_execution]
        if len(tail):
            held_prices = tail.reindex(columns=shares.index).ffill().fillna(
                last_price.reindex(shares.index)
            )
            values = held_prices.mul(shares, axis=1).sum(axis=1)
            for date, value in values.items():
                nav_points.append((date, float(value + cash)))

    if not nav_points:
        raise RuntimeError(
            "el backtest no produjo ningun punto de NAV: revisa que el panel "
            "tenga scores y que las fechas coincidan con sesiones de mercado"
        )

    nav = pd.Series(dict(nav_points)).sort_index()
    nav.index = pd.DatetimeIndex(nav.index)
    nav = nav[~nav.index.duplicated(keep="last")]

    return BacktestResult(
        nav=nav,
        holdings=pd.DataFrame(holdings_rows),
        trades=pd.DataFrame(trade_rows),
        rebalances=pd.DataFrame(rebalance_rows),
        config_fingerprint=cfg.fingerprint,
        warnings=warnings,
    )


def benchmark_nav(close_wide: pd.DataFrame, ticker: str, nav_index: pd.DatetimeIndex,
                  initial_capital: float) -> pd.Series | None:
    """NAV de comprar y mantener el benchmark, alineado con la estrategia."""
    if ticker not in close_wide.columns:
        return None
    series = close_wide[ticker].reindex(nav_index).ffill().dropna()
    if series.empty:
        return None
    return series / series.iloc[0] * initial_capital
