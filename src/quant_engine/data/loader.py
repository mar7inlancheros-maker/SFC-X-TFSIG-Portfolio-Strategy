"""Descarga y cache de OHLCV, y sector de cada ticker.

Cache propia, separada de la del modelo multifactor: aquella guarda solo cierre
y volumen de miles de tickers; este motor necesita maximo y minimo (para el ATR)
de una decena. Reutiliza las primitivas de disco de `sfc_tfsig.data.cache`.

Nada se descarta en silencio. Un ticker que Yahoo no devuelve queda registrado
con su motivo y aparece en el reporte de calidad de datos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from sfc_tfsig.data import cache

log = logging.getLogger(__name__)

_FIELDS = ("Open", "High", "Low", "Close", "Volume")
_MAX_AGE_DAYS = 1.0  # los cierres de ayer cambian el analisis; los de hace un ano no


@dataclass
class MarketData:
    """OHLCV por ticker, mas lo que no se pudo cargar y por que."""

    ohlcv: dict[str, pd.DataFrame]
    failures: dict[str, str] = field(default_factory=dict)
    sectors: dict[str, str] = field(default_factory=dict)
    sector_source: dict[str, str] = field(default_factory=dict)


def _cache_name(ticker: str) -> str:
    return f"qe_ohlcv/{ticker.replace('^', '_')}"


def _download(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    import yfinance as yf  # perezoso: los tests no tocan la red

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
        actions=False,
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for ticker in tickers:
        try:
            block = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        block = block[[c for c in _FIELDS if c in block.columns]].dropna(how="all")
        if block.empty or block["Close"].dropna().empty:
            continue
        block.index = pd.to_datetime(block.index).tz_localize(None)
        block.index.name = "date"
        out[ticker] = block
    return out


_REQUESTED = "qe_ohlcv_requested"


def load_ohlcv(
    tickers: list[str] | tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    refresh: bool = False,
) -> MarketData:
    """OHLCV ajustado por splits y dividendos para cada ticker.

    La validez de la cache se decide por el REGISTRO de desde cuando se pidio
    cada ticker, no por su primer precio: una accion que salio a bolsa hace dos
    anos nunca tendra precios de hace cinco, y mirando solo la cache se volveria
    a descargar en cada corrida. Es el mismo error que tuvo `sfc_tfsig.data.prices`.
    """
    wanted = list(dict.fromkeys(t.upper() for t in tickers))
    result = MarketData(ohlcv={})
    requested = cache.read_json(_REQUESTED) or {}
    to_fetch: list[str] = []

    for ticker in wanted:
        cached = None if refresh else cache.read_frame(_cache_name(ticker), max_age_days=_MAX_AGE_DAYS)
        asked_from = requested.get(ticker)
        covered = asked_from is not None and pd.Timestamp(asked_from) <= start + pd.Timedelta(days=7)
        if cached is not None and not cached.empty and covered:
            frame = cached.set_index(pd.to_datetime(cached["date"])).drop(columns="date")
            result.ohlcv[ticker] = frame.loc[frame.index >= start]
            continue
        to_fetch.append(ticker)

    if to_fetch:
        log.info("descargando %d tickers de Yahoo", len(to_fetch))
        downloaded = _download(to_fetch, start, end)
        for ticker in to_fetch:
            frame = downloaded.get(ticker)
            if frame is None:
                result.failures[ticker] = "Yahoo no devolvio datos (ticker invalido, deslistado o sin historia)"
                continue
            cache.write_frame(_cache_name(ticker), frame.reset_index())
            requested[ticker] = start.strftime("%Y-%m-%d")
            result.ohlcv[ticker] = frame
        cache.write_json(_REQUESTED, requested)
    return result


def load_sectors(tickers: list[str] | tuple[str, ...]) -> tuple[dict[str, str], dict[str, str]]:
    """Sector por el SIC de la SEC, el mismo esquema de 11 sectores del modelo.

    Si la SEC no esta disponible (sin User-Agent, sin red) el sector queda
    "Unknown" y la fuente dice por que. No se inventa un sector.
    """
    sectors: dict[str, str] = {}
    source: dict[str, str] = {}
    try:
        from sfc_tfsig.data import sec
        from sfc_tfsig.universe import company_profile, sector_from_sic

        listing = sec.listed_companies().drop_duplicates("ticker").set_index("ticker")["cik"]
    except Exception as exc:  # noqa: BLE001 -- se reporta, no se traga
        reason = f"SEC no disponible: {type(exc).__name__}"
        for t in tickers:
            sectors[t], source[t] = "Unknown", reason
        return sectors, source

    for ticker in tickers:
        cik = listing.get(ticker)
        if cik is None or pd.isna(cik):
            sectors[ticker], source[ticker] = "Unknown", "no figura en la lista de la SEC (ETF o extranjero)"
            continue
        try:
            profile = company_profile(int(cik))
            sectors[ticker] = sector_from_sic(profile.get("sic"))
            source[ticker] = f"SEC SIC {profile.get('sic')}"
        except Exception as exc:  # noqa: BLE001
            sectors[ticker], source[ticker] = "Unknown", f"perfil SEC fallo: {type(exc).__name__}"
    return sectors, source
