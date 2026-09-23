"""Pruebas de estres: que le pasa a la cartera de HOY si se repite un mal episodio.

El VaR y el Monte Carlo describen la distribucion de dias normales y algo
peores. El estres pregunta por los dias que la muestra no tiene o tiene pocos:
2008 no esta en el backtest, y un bootstrap no puede inventarlo.

Tres familias:

- **Historicas.** La cartera actual con los precios REALES de cada episodio.
  Un nombre que no cotizaba entonces se aproxima con su beta frente al
  benchmark, y la tabla dice que fraccion del peso se aproximo: un escenario
  con el 60% del peso aproximado es un escenario de beta, no de nombres.
- **Hipoteticas.** Choque por nombre = beta x mercado + extra de su sector +
  choque de cada factor x z-score del nombre en ese factor. Cubren episodios
  anteriores a la historia de precios (1987, 2000) y los riesgos propios del
  modelo, como un crash de momentum.
- **Inversa.** Cuanto tiene que caer el mercado para que la cartera pierda un
  umbral dado. Lineal en la beta: es un orden de magnitud, no una prevision.

Un choque nunca se deja por debajo de -100%: una accion no puede valer menos
que cero, y una suma lineal de choques grandes lo permitiria.

Funciones puras: entran pesos, betas y precios; salen tablas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Un precio "del dia de inicio" que en realidad es de hace mas de una semana no
# es de ese episodio: el nombre no cotizaba y se aproxima por beta.
_STALE_DAYS = 7


def price_on_or_before(close_wide: pd.DataFrame, date, *, max_stale_days: int = _STALE_DAYS) -> pd.Series:
    """Ultimo cierre de cada columna en o antes de `date`, si no es viejo."""
    date = pd.Timestamp(date)
    visible = close_wide.loc[close_wide.index <= date]
    if visible.empty:
        return pd.Series(np.nan, index=close_wide.columns)
    last_valid = visible.apply(lambda s: s.last_valid_index())
    prices = visible.ffill().iloc[-1]
    stale = last_valid.isna() | ((date - pd.to_datetime(last_valid)).dt.days > max_stale_days)
    return prices.where(~stale)


def window_returns(close_wide: pd.DataFrame, start, end) -> pd.Series:
    """Retorno de cada columna entre el cierre de `start` y el de `end`."""
    first = price_on_or_before(close_wide, start)
    last = price_on_or_before(close_wide, end)
    return last / first - 1.0


def _nav_return(nav: pd.Series | None, start, end) -> float:
    """Retorno de una serie de NAV en la ventana, si la cubre entera."""
    if nav is None or nav.empty or nav.index[0] > pd.Timestamp(start):
        return float("nan")
    frame = nav.to_frame("nav")
    return float(window_returns(frame, start, end).iloc[0])


def historical_stress(
    weights: pd.Series,
    betas: pd.Series,
    close_wide: pd.DataFrame,
    benchmark: str,
    scenarios,
    *,
    strategy_nav: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repite cada episodio sobre la cartera actual. Devuelve (resumen, detalle).

    `strategy_nav` es opcional: si el backtest cubre el episodio, se anade lo
    que la estrategia perdio EN SU MOMENTO, con la cartera que tenia entonces.
    Las dos cifras responden a preguntas distintas y ambas importan.
    """
    w = weights.astype(float)
    names = list(w.index)
    rows, details = [], []
    for scenario in scenarios:
        name, start, end = scenario["name"], pd.Timestamp(scenario["start"]), pd.Timestamp(scenario["end"])
        base = {"scenario": name, "start": start.date(), "end": end.date()}

        if benchmark not in close_wide.columns:
            rows.append({**base, "status": f"sin precios de {benchmark}"})
            continue
        market = float(window_returns(close_wide[[benchmark]], start, end).iloc[0])
        if not np.isfinite(market):
            rows.append({**base, "status": f"sin precios de {benchmark} en el episodio"})
            continue

        observed = window_returns(close_wide.reindex(columns=names), start, end).reindex(names)
        proxied = observed.isna()
        shocks = observed.where(~proxied, betas.reindex(names) * market)
        # Sin beta ni precio, el nombre se mueve como el mercado: suponer que no
        # se mueve seria suponer que es caja.
        shocks = shocks.fillna(market).clip(lower=-1.0)
        pnl = w * shocks
        worst = pnl.idxmin() if len(pnl) else None

        rows.append({
            **base,
            "status": "ok",
            "portfolio": float(pnl.sum()),
            "benchmark": market,
            "relative": float(pnl.sum()) - market * float(w.sum()),
            "weight_proxied": float(w[proxied].sum()),
            "worst_name": worst,
            "worst_contribution": float(pnl.min()) if len(pnl) else float("nan"),
            "strategy_then": _nav_return(strategy_nav, start, end),
        })
        details.append(pd.DataFrame({
            "scenario": name,
            "ticker": names,
            "weight": w.to_numpy(),
            "shock": shocks.to_numpy(),
            "proxied": proxied.to_numpy(),
            "contribution": pnl.to_numpy(),
        }))
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return pd.DataFrame(rows), detail


def hypothetical_stress(
    weights: pd.Series,
    sectors: pd.Series,
    betas: pd.Series,
    scenarios,
    *,
    factor_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Escenarios declarados en `config/risk.toml`. Devuelve (resumen, detalle).

    `factor_scores` trae columnas `score_<factor>` por ticker. Un nombre sin
    score en un factor recibe choque de factor cero: no se le supone ni la
    exposicion media ni la extrema.
    """
    w = weights.astype(float)
    names = list(w.index)
    beta = betas.reindex(names).fillna(1.0)
    sector = sectors.reindex(names).fillna("Unknown")
    rows, details = [], []
    for scenario in scenarios:
        market = float(scenario["market"])
        sector_shock = sector.map(scenario.get("sectors", {})).fillna(0.0).astype(float)
        factor_shock = pd.Series(0.0, index=names)
        missing_factors = []
        for factor, shock in scenario.get("factors", {}).items():
            column = f"score_{factor}"
            if factor_scores is None or column not in factor_scores.columns:
                missing_factors.append(factor)
                continue
            z = pd.to_numeric(factor_scores[column], errors="coerce").reindex(names).fillna(0.0)
            factor_shock += float(shock) * z

        shocks = (beta * market + sector_shock + factor_shock).clip(lower=-1.0)
        pnl = w * shocks
        rows.append({
            "scenario": scenario["name"],
            "market": market,
            "portfolio": float(pnl.sum()),
            "relative": float(pnl.sum()) - market * float(w.sum()),
            "from_beta": float((w * beta * market).sum()),
            "from_sectors": float((w * sector_shock).sum()),
            "from_factors": float((w * factor_shock).sum()),
            "worst_name": pnl.idxmin() if len(pnl) else None,
            "worst_contribution": float(pnl.min()) if len(pnl) else float("nan"),
            "missing_factors": ", ".join(missing_factors),
        })
        details.append(pd.DataFrame({
            "scenario": scenario["name"],
            "ticker": names,
            "sector": sector.to_numpy(),
            "weight": w.to_numpy(),
            "shock": shocks.to_numpy(),
            "contribution": pnl.to_numpy(),
        }))
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return pd.DataFrame(rows), detail


def reverse_stress(portfolio_beta: float, invested: float, loss_thresholds) -> pd.DataFrame:
    """Caida del mercado que produce cada perdida, via la beta de la cartera.

    Lineal y sin riesgo especifico: con 30 nombres el idiosincratico no
    desaparece, asi que la perdida real a esa caida del mercado puede ser
    mayor o menor. Sirve para situar el umbral: "perder un 20% exige un
    mercado a -17%" se discute distinto que "exige un mercado a -40%".
    """
    rows = []
    for loss in loss_thresholds:
        exposure = portfolio_beta * invested
        needed = -float(loss) / exposure if exposure and exposure > 0 else float("nan")
        rows.append({
            "loss": float(loss),
            "market_move": needed,
            "plausible": bool(np.isfinite(needed) and needed > -1.0),
        })
    return pd.DataFrame(rows)


def worst_windows(nav: pd.Series, benchmark_nav: pd.Series | None, windows_d) -> pd.DataFrame:
    """Peor retorno de la estrategia en ventanas de `h` sesiones, y el del mercado."""
    nav = nav.dropna().astype(float)
    bench = benchmark_nav.reindex(nav.index).ffill() if benchmark_nav is not None else None
    rows = []
    for h in windows_d:
        if len(nav) <= h:
            continue
        rolling = nav / nav.shift(h) - 1.0
        end = rolling.idxmin()
        start = nav.index[nav.index.get_loc(end) - h]
        row = {"window_d": int(h), "start": start.date(), "end": end.date(),
               "strategy": float(rolling.min())}
        if bench is not None:
            row["benchmark"] = float(bench.loc[end] / bench.loc[start] - 1.0)
        rows.append(row)
    return pd.DataFrame(rows)
