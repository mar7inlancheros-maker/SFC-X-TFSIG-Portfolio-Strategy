"""Atribucion: de donde salio el resultado, nombre a nombre y sector a sector.

Un CAGR agregado no distingue dos historias que exigen decisiones opuestas:

- El modelo acerto de forma AMPLIA -- treinta posiciones, varios sectores, varios
  anos. Eso es un proceso, y un proceso se puede repetir con capital.
- El modelo acerto de forma CONCENTRADA -- dos o tres nombres en un ano. Eso es
  una posicion afortunada, y repetirla con capital de terceros es una apuesta
  disfrazada de sistema.

Este modulo separa las dos. Es la pregunta que bloquea todas las demas: mientras
no se sepa si el exceso fue amplio, no tiene sentido ajustar pesos, costes ni
rotacion.

**Como se calcula.** El backtest compra acciones y las mantiene hasta el
siguiente rebalanceo, asi que la contribucion de una posicion al retorno del
periodo es exacta, no una aproximacion:

    contribucion_i = acciones_i x (precio_final_i - precio_inicial_i) / NAV_inicial

La suma sobre todas las posiciones da el retorno BRUTO del periodo. La
diferencia contra el retorno neto del NAV son los costes, y se reporta aparte
en vez de repartirla entre los nombres: el coste es del proceso, no de la
accion.

Funciones puras: entran los artefactos que ya guarda el backtest y salen tablas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def position_contributions(
    holdings: pd.DataFrame,
    close_wide: pd.DataFrame,
    rebalances: pd.DataFrame,
) -> pd.DataFrame:
    """Contribucion de cada posicion al retorno de su periodo de tenencia.

    Columnas de salida: date, end_date, ticker, sector, weight, shares,
    price_start, price_end, position_return, contribution.

    `contribution` esta en unidades de retorno de cartera: 0.01 = un punto
    porcentual de NAV ganado en ese periodo por esa posicion.
    """
    if holdings.empty:
        return pd.DataFrame(
            columns=["date", "end_date", "ticker", "sector", "weight", "shares",
                     "price_start", "price_end", "position_return", "contribution"]
        )

    holdings = holdings.copy()
    holdings["date"] = pd.to_datetime(holdings["date"])

    nav_by_date = (
        rebalances.assign(execution_date=pd.to_datetime(rebalances["execution_date"]))
        .set_index("execution_date")["nav"]
        .astype(float)
    )

    dates = sorted(holdings["date"].unique())
    # El ultimo periodo se cierra con el ultimo cierre disponible, no se descarta:
    # si el modelo va ganando en el tramo abierto, ignorarlo sesga a la baja.
    next_date = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    last_session = close_wide.index.max()

    blocks: list[pd.DataFrame] = []
    for date in dates:
        block = holdings[holdings["date"] == date].copy()
        end = next_date.get(date, last_session)

        # NAV al INICIO del periodo: es el denominador correcto. Usar el NAV
        # final repartiria el propio resultado del periodo entre los pesos.
        nav_start = float(nav_by_date.get(pd.Timestamp(date), np.nan))
        if not np.isfinite(nav_start) or nav_start <= 0:
            continue

        prices_end = _price_at(close_wide, pd.Timestamp(end))
        block["end_date"] = pd.Timestamp(end)
        block["price_start"] = pd.to_numeric(block["price"], errors="coerce")
        block["price_end"] = block["ticker"].map(prices_end)
        # Sin precio final la posicion se valora a su precio de entrada: aporta
        # cero en vez de desaparecer del total, que es lo honesto cuando el dato
        # falta por cobertura y no porque la empresa valga cero.
        block["price_end"] = block["price_end"].fillna(block["price_start"])

        block["position_return"] = block["price_end"] / block["price_start"] - 1.0
        block["contribution"] = (
            block["shares"].astype(float)
            * (block["price_end"] - block["price_start"])
            / nav_start
        )
        blocks.append(block)

    if not blocks:
        return pd.DataFrame()

    out = pd.concat(blocks, ignore_index=True)
    columns = ["date", "end_date", "ticker", "sector", "weight", "shares",
               "price_start", "price_end", "position_return", "contribution"]
    return out[[c for c in columns if c in out.columns]]


def _price_at(close_wide: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Ultimo cierre conocido en o antes de `date`."""
    visible = close_wide.loc[close_wide.index <= date]
    if visible.empty:
        return pd.Series(dtype="float64")
    return visible.ffill().iloc[-1]


# ---------------------------------------------------------------------------
#  Agregaciones
# ---------------------------------------------------------------------------


def by_year(contributions: pd.DataFrame) -> pd.DataFrame:
    """Contribucion agregada por ano natural, con medidas de amplitud.

    `top5_share` es la fraccion del resultado bruto del ano explicada por las
    cinco mejores contribuciones. Es la medida que separa proceso de suerte:
    por encima del 60% con treinta posiciones, el ano lo hicieron unos pocos
    nombres.
    """
    if contributions.empty:
        return pd.DataFrame()

    df = contributions.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    rows = []
    for year, block in df.groupby("year"):
        total = float(block["contribution"].sum())
        by_name = block.groupby("ticker")["contribution"].sum().sort_values(ascending=False)
        winners = by_name[by_name > 0]
        rows.append(
            {
                "year": int(year),
                "gross_contribution": total,
                "n_positions": int(block["ticker"].nunique()),
                "n_winners": int((by_name > 0).sum()),
                "n_losers": int((by_name < 0).sum()),
                "top1": float(by_name.iloc[0]) if len(by_name) else np.nan,
                "top1_name": by_name.index[0] if len(by_name) else None,
                "top5_sum": float(by_name.head(5).sum()) if len(by_name) else np.nan,
                "top5_share": float(by_name.head(5).sum() / total) if total > 0 else np.nan,
                # Cuantos nombres hacen falta para explicar la mitad del total
                # positivo. Amplitud en una sola cifra.
                "names_for_half": _names_for_half(winners),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def _names_for_half(winners: pd.Series) -> float:
    if winners.empty:
        return np.nan
    cumulative = winners.cumsum() / winners.sum()
    return float((cumulative < 0.5).sum() + 1)


def by_sector(contributions: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """Contribucion y peso medio por sector."""
    if contributions.empty or "sector" not in contributions.columns:
        return pd.DataFrame()
    df = contributions.copy()
    if year is not None:
        df = df[pd.to_datetime(df["date"]).dt.year == year]
    out = df.groupby("sector").agg(
        contribution=("contribution", "sum"),
        avg_weight=("weight", "mean"),
        n_holdings=("ticker", "count"),
        hit_rate=("position_return", lambda s: float((s > 0).mean())),
    )
    return out.sort_values("contribution", ascending=False)


def top_names(contributions: pd.DataFrame, year: int | None = None, n: int = 15) -> pd.DataFrame:
    """Mejores y peores nombres por contribucion acumulada."""
    if contributions.empty:
        return pd.DataFrame()
    df = contributions.copy()
    if year is not None:
        df = df[pd.to_datetime(df["date"]).dt.year == year]
    grouped = df.groupby("ticker").agg(
        contribution=("contribution", "sum"),
        periods_held=("ticker", "count"),
        avg_weight=("weight", "mean"),
        sector=("sector", "first"),
    ).sort_values("contribution", ascending=False)
    return pd.concat([grouped.head(n), grouped.tail(n)])


def concentration_summary(contributions: pd.DataFrame) -> dict[str, float]:
    """Amplitud del resultado en todo el historico.

    Si quitar los cinco mejores nombres de toda la muestra elimina el resultado,
    el modelo no tiene un proceso: tuvo cinco aciertos.
    """
    if contributions.empty:
        return {}
    by_name = contributions.groupby("ticker")["contribution"].sum().sort_values(ascending=False)
    total = float(by_name.sum())
    return {
        "total_gross_contribution": total,
        "n_names_ever_held": int(len(by_name)),
        "pct_names_positive": float((by_name > 0).mean()),
        "top5_share": float(by_name.head(5).sum() / total) if total else np.nan,
        "top10_share": float(by_name.head(10).sum() / total) if total else np.nan,
        "total_without_top5": total - float(by_name.head(5).sum()),
        "names_for_half": _names_for_half(by_name[by_name > 0]),
    }


def excess_decomposition(
    contributions: pd.DataFrame,
    nav: pd.Series,
    benchmark_nav: pd.Series,
) -> pd.DataFrame:
    """Exceso anual contra el benchmark, junto a la amplitud de ese ano.

    Pone las dos cosas en la misma tabla a proposito: un ano de exceso grande
    con `top5_share` alto y `names_for_half` bajo es un ano que no se puede
    prometer que se repita.
    """
    yearly = by_year(contributions)
    if yearly.empty:
        return yearly

    strategy = nav.resample("YE").last().pct_change()
    bench = benchmark_nav.reindex(nav.index).ffill().resample("YE").last().pct_change()

    # El primer ano necesita el punto de partida, que `pct_change` deja en NaN.
    strategy.loc[strategy.index[0]] = nav.resample("YE").last().iloc[0] / nav.iloc[0] - 1.0
    bench_annual = benchmark_nav.reindex(nav.index).ffill().resample("YE").last()
    bench.loc[bench.index[0]] = bench_annual.iloc[0] / benchmark_nav.reindex(nav.index).ffill().iloc[0] - 1.0

    frame = pd.DataFrame({
        "strategy": strategy.to_numpy(),
        "benchmark": bench.to_numpy(),
    }, index=[d.year for d in strategy.index])
    frame["excess"] = frame["strategy"] - frame["benchmark"]

    joined = frame.join(yearly[["top5_share", "names_for_half", "n_winners", "n_losers"]])
    joined.index.name = "year"
    return joined
