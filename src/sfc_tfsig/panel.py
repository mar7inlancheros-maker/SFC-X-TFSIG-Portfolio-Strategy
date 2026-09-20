"""Ensamblaje del panel: la rejilla fecha x empresa sobre la que corre todo.

Este modulo es el unico sitio donde se juntan fundamentales y precios, y por
tanto el unico sitio donde se puede colar informacion del futuro. Las reglas que
lo impiden, todas verificables leyendo el codigo de abajo:

1. Los fundamentales entran por `sec.as_of(fecha)`, que filtra por FECHA DE
   PRESENTACION. Nunca por fecha de cierre contable.
2. El precio de una fecha es el ultimo cierre en o ANTES de esa fecha.
3. Momentum y volatilidad se calculan sobre la ventana que termina en esa fecha.
4. El retorno futuro (`forward_return`) se agrega aparte, con nombre explicito, y
   solo lo consume `validation.py`. El backtest no lo mira.
5. Los filtros de liquidez usan el precio y el volumen de esa fecha, no los de
   hoy. Una empresa que hoy es liquida pero en 2014 cotizaba a 3 dolares no
   entra en el universo de 2014.

Una empresa con fundamentales demasiado viejos se cae del panel: si la ultima
presentacion tiene mas de `max_fundamental_staleness_d` dias, o dejo de reportar
o hay un problema, y en ambos casos no se le pone capital detras.
"""

from __future__ import annotations

import pandas as pd

from . import financials, universe as universe_mod
from .config import Config
from .data import prices as prices_mod, sec

# Conceptos crudos que viajan al panel desde la SEC.
_CONCEPT_COLUMNS = tuple(sec.CONCEPTS.keys())

# Colchon de historia antes de la primera fecha de rebalanceo: hace falta para
# momentum (12 meses) y volatilidad (252 sesiones). Sin el, los primeros meses
# del backtest salen sin factores de precio y el modelo parece funcionar peor
# al principio por una razon puramente mecanica.
_WARMUP_MONTHS = 30


def rebalance_dates(cfg: Config, available: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Fechas de decision: fin de mes o de trimestre, dentro del rango."""
    start = pd.Timestamp(cfg.get("calendar.start"))
    end_raw = str(cfg.get("calendar.end") or "").strip()
    end = pd.Timestamp(end_raw) if end_raw else available.max()

    freq = "ME" if cfg.get("calendar.rebalance") == "M" else "QE"
    grid = pd.date_range(start=start, end=end, freq=freq)
    # Solo fechas para las que hay mercado: el ultimo dia natural del mes puede
    # ser domingo.
    return pd.DatetimeIndex([d for d in grid if (available <= d).any()])


def _as_of_price_frame(close_wide: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Ultimo cierre conocido en o antes de `date`."""
    visible = close_wide.loc[close_wide.index <= date]
    if visible.empty:
        return pd.Series(dtype="float64")
    return visible.iloc[-1]


def build_panel(cfg: Config, *, progress: bool = True, refresh: bool = False) -> pd.DataFrame:
    """Construye el panel completo. Es la parte lenta: descarga y cachea.

    Devuelve un marco largo con una fila por (fecha, ticker) y columnas de
    identidad, mercado, fundamentales crudos, ratios y factores de precio.
    """
    if progress:
        print("[1/6] universo estatico desde EDGAR...", flush=True)
    # Sin perfiles todavia: son una peticion por empresa y aun no sabemos
    # cuales sobreviven al filtro de liquidez.
    static = universe_mod.build_universe(cfg, with_profiles=False, progress=progress)
    if static.empty:
        raise RuntimeError("el universo estatico salio vacio: revisa [universe].exchanges")

    start = pd.Timestamp(cfg.get("calendar.start"))
    end_raw = str(cfg.get("calendar.end") or "").strip()
    end = pd.Timestamp(end_raw) if end_raw else pd.Timestamp.today().normalize()
    price_start = start - pd.DateOffset(months=_WARMUP_MONTHS)

    if progress:
        print(f"[2/6] precios de {len(static)} tickers desde {price_start.date()}...", flush=True)
    long_prices = prices_mod.get_prices(
        static["ticker"], price_start, end, refresh=refresh, progress=progress
    )
    close_wide = prices_mod.to_wide(long_prices, "close")
    dollar_volume_daily = prices_mod.to_wide(
        long_prices.assign(dv=long_prices["close"] * long_prices["volume"]), "dv"
    )

    # Prefiltro de coste: sin precio no hay nada que hacer, y sin liquidez
    # suficiente en NINGUN momento del historico la empresa no va a entrar en la
    # cartera ningun mes. Bajar su XBRL es gastar peticiones para nada.
    liquid = universe_mod.ever_liquid(
        dollar_volume_daily, float(cfg.get("universe.min_dollar_volume"))
    )
    tradable = static[
        static["ticker"].isin(close_wide.columns) & static["ticker"].isin(liquid)
    ].copy()
    if progress:
        print(f"      {len(tradable)} con precio y liquidez suficiente en algun momento",
              flush=True)
    if tradable.empty:
        raise RuntimeError(
            "ningun ticker supera el filtro de liquidez. Revisa "
            "[universe].min_dollar_volume o la cache de precios"
        )

    if progress:
        print(f"[3/6] perfiles (sector y pais) de {len(tradable)} empresas...", flush=True)
    tradable = universe_mod.attach_profiles(tradable, cfg, progress=progress)
    if progress:
        print(f"      {len(tradable)} tras excluir vehiculos de inversion y otros paises",
              flush=True)

    if progress:
        print("[4/6] fundamentales point-in-time (XBRL)...", flush=True)
    observations = sec.download_fundamentals(tradable["cik"], progress=progress)

    if progress:
        print("[5/6] factores de precio...", flush=True)
    momentum = prices_mod.momentum(
        close_wide,
        lookback_m=int(cfg.get("factors.momentum.lookback_m")),
        skip_m=int(cfg.get("factors.momentum.skip_m")),
    )
    volatility = prices_mod.realized_volatility(
        close_wide, window_d=int(cfg.get("factors.lowvol.lookback_d"))
    )
    dollar_volume = prices_mod.median_dollar_volume(long_prices)

    dates = rebalance_dates(cfg, close_wide.index)
    if progress:
        print(f"[6/6] ensamblando {len(dates)} fechas de rebalanceo...", flush=True)

    cik_to_ticker = dict(zip(tradable["cik"], tradable["ticker"]))
    max_stale = int(cfg.get("universe.max_fundamental_staleness_d", 550))

    blocks: list[pd.DataFrame] = []
    for date in dates:
        visible = sec.as_of(observations, date)
        if visible.empty:
            continue
        visible = visible.copy()
        visible["ticker"] = visible["cik"].map(cik_to_ticker)
        visible = visible.dropna(subset=["ticker"])

        staleness = (date - pd.to_datetime(visible["last_period_end"])).dt.days
        visible = visible[staleness <= max_stale]
        if visible.empty:
            continue

        price = _as_of_price_frame(close_wide, date)
        visible["price"] = visible["ticker"].map(price)
        if "shares" in visible.columns:
            visible["market_cap"] = visible["price"] * visible["shares"]
        else:
            visible["market_cap"] = pd.NA

        month = date.to_period("M")

        def _monthly(frame: pd.DataFrame) -> pd.Series:
            """Valor de la fila mensual correspondiente a `date`, o vacio."""
            if frame is None or frame.empty:
                return pd.Series(dtype="float64")
            rows = frame.loc[frame.index.to_period("M") == month]
            return rows.iloc[-1] if len(rows) else pd.Series(dtype="float64")

        visible["momentum"] = visible["ticker"].map(_monthly(momentum))
        visible["volatility"] = visible["ticker"].map(_monthly(volatility))
        visible["dollar_volume"] = visible["ticker"].map(_monthly(dollar_volume))

        eligible = universe_mod.apply_liquidity_filters(
            visible.merge(
                tradable[["cik", "name", "exchange", "sector", "country"]], on="cik", how="left"
            ),
            cfg,
            price=visible.set_index("ticker")["price"],
            dollar_volume=visible.set_index("ticker")["dollar_volume"],
            market_cap=visible.set_index("ticker")["market_cap"],
        )
        if eligible.empty:
            continue

        eligible.insert(0, "date", date)
        blocks.append(eligible)

    if not blocks:
        raise RuntimeError(
            "el panel salio vacio. Causas tipicas: rango de fechas anterior al "
            "XBRL de la SEC (2009), filtros de liquidez demasiado exigentes, o "
            "cache de precios incompleta"
        )

    panel = pd.concat(blocks, ignore_index=True)
    panel = _add_shares_growth(panel)
    panel = financials.compute_all(panel)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def _add_shares_growth(panel: pd.DataFrame) -> pd.DataFrame:
    """Dilucion a 12 meses: acciones de hoy frente a las de hace un ano.

    Se calcula sobre el propio panel, comparando cada fecha con la fila del
    mismo ticker 12 meses antes. Si la empresa no estaba en el universo hace un
    ano, queda NaN y el factor de calidad se promedia sin ella.
    """
    if "shares" not in panel.columns:
        panel["shares_growth"] = pd.NA
        return panel

    out = panel.copy()
    out["_month"] = out["date"].dt.to_period("M")
    lagged = out[["ticker", "_month", "shares"]].copy()
    lagged["_month"] = lagged["_month"] + 12
    lagged = lagged.rename(columns={"shares": "shares_prior"})

    out = out.merge(lagged, on=["ticker", "_month"], how="left")
    prior = pd.to_numeric(out["shares_prior"], errors="coerce")
    current = pd.to_numeric(out["shares"], errors="coerce")
    out["shares_growth"] = (current / prior.where(prior > 0)) - 1.0
    return out.drop(columns=["_month", "shares_prior"])


def add_forward_returns(panel: pd.DataFrame, close_wide: pd.DataFrame, horizon_m: int = 1) -> pd.DataFrame:
    """Agrega el retorno futuro. SOLO para validacion estadistica.

    Se separa del backtest a proposito: el backtest ejecuta operaciones y
    contabiliza costes; esta columna es para medir poder predictivo (IC,
    quintiles, Fama-MacBeth). Mezclar las dos cosas es como aparecen los
    resultados que no se pueden reproducir con dinero.
    """
    forward = prices_mod.forward_returns(close_wide, horizon_m=horizon_m)
    stacked = forward.stack(future_stack=True).rename("forward_return").reset_index()
    stacked.columns = ["date", "ticker", "forward_return"]
    stacked["_month"] = stacked["date"].dt.to_period("M")

    out = panel.copy()
    out["_month"] = out["date"].dt.to_period("M")
    out = out.merge(
        stacked[["ticker", "_month", "forward_return"]], on=["ticker", "_month"], how="left"
    )
    return out.drop(columns="_month")


def coverage_report(panel: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Cobertura de cada metrica: que fraccion del panel la tiene.

    Se mira ANTES de creerse un backtest. Una metrica con 40% de cobertura no
    esta midiendo lo que dice medir: esta midiendo "empresas que reportan esta
    etiqueta", que suele correlacionar con tamano y con sector.
    """
    metrics = metrics or [c for c in financials.METRIC_DIRECTION if c in panel.columns]
    rows = []
    for metric in metrics:
        series = pd.to_numeric(panel[metric], errors="coerce")
        rows.append(
            {
                "metric": metric,
                "coverage": float(series.notna().mean()),
                "median": float(series.median(skipna=True)),
                "p05": float(series.quantile(0.05)),
                "p95": float(series.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows).sort_values("coverage")
