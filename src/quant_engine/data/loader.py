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

from ..sessions import NYSECalendar

log = logging.getLogger(__name__)

_FIELDS = ("Open", "High", "Low", "Close", "Volume")
_MAX_AGE_DAYS = 1.0  # los cierres de ayer cambian el analisis; los de hace un ano no


def _today(today=None) -> pd.Timestamp:
    return (pd.Timestamp(today) if today is not None else pd.Timestamp.today()).normalize()


def last_settled_session(today=None) -> pd.Timestamp:
    """Ultima sesion de NYSE ya cerrada: la anterior a hoy, nunca hoy."""
    return _today(today) - pd.offsets.CustomBusinessDay(calendar=NYSECalendar())


def settled(frame: pd.DataFrame, today=None) -> pd.DataFrame:
    """Sin la barra de hoy: con el mercado abierto no es un cierre.

    El 10 de noviembre de 2026 una corrida de las 15:34 tomo el 63,01 de media
    sesion de NFLX como "cierre" para entrada, stop y objetivos, y lo guardo en
    la cache, donde al dia siguiente habria seguido pasando por cierre. Es la
    misma regla que `sfc_tfsig.data.prices` (`_drop_unsettled`).
    """
    return frame.loc[frame.index < _today(today)]


def written_at(name: str) -> pd.Timestamp | None:
    """Cuando se escribio la cache de un nombre, o None si no existe."""
    age = cache.age_days(name)
    return None if age is None else pd.Timestamp.now() - pd.Timedelta(days=age)


def cache_usable(frame: pd.DataFrame, written: pd.Timestamp | None, today=None) -> bool:
    """La cache de un ticker vale si trae la ultima sesion cerrada y se escribio
    DESPUES del dia de su ultima barra.

    Escrita el mismo dia que su ultima barra, esa barra pudo ser de media sesion
    (las caches anteriores a esta regla guardaban la de hoy). Sin la ultima sesion
    cerrada, el analisis saldria con el cierre de anteayer.
    """
    if frame.empty or written is None:
        return False
    last = pd.Timestamp(frame.index.max()).normalize()
    return pd.Timestamp(written).normalize() > last and last >= last_settled_session(today)


@dataclass
class MarketData:
    """OHLCV por ticker, mas lo que no se pudo cargar y por que."""

    ohlcv: dict[str, pd.DataFrame]
    failures: dict[str, str] = field(default_factory=dict)
    stale: dict[str, pd.Timestamp] = field(default_factory=dict)  # ticker -> ultimo dato de la cache vieja
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
_RETRY_BATCH = 25


def download_with_retry(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp,
                        download=None) -> dict[str, pd.DataFrame]:
    """Descarga en un lote y reintenta lo que falte en lotes de 25.

    En lotes grandes Yahoo corta a veces y yfinance da el ticker por "sin
    datos" aunque cotice: el 17 de octubre de 2026 faltaron 16 de los 50 pares
    de referencia (AMZN, JPM, V...) y un minuto despues bajaron todos. Un error
    de Yahoo no es "no hay datos".
    """
    download = download or _download
    out = download(tickers, start, end)
    missing = [t for t in tickers if t not in out]
    for i in range(0, len(missing), _RETRY_BATCH):
        out.update(download(missing[i:i + _RETRY_BATCH], start, end))
    return out

# Sufijo de bolsa de Yahoo -> (moneda de cotizacion, unidades de esa moneda por
# precio). Londres cotiza en peniques: 0.01 libras por unidad de precio. Yahoo
# usa "-" para las clases (BRK-B) y "." solo para la bolsa, asi que un punto
# en el ticker es siempre un sufijo de bolsa.
QUOTE_CURRENCY = {
    ".TO": ("CAD", 1.0), ".V": ("CAD", 1.0), ".NE": ("CAD", 1.0), ".CN": ("CAD", 1.0),
    ".L": ("GBP", 0.01), ".PA": ("EUR", 1.0), ".DE": ("EUR", 1.0), ".AS": ("EUR", 1.0),
    ".MI": ("EUR", 1.0), ".MC": ("EUR", 1.0), ".BR": ("EUR", 1.0), ".SW": ("CHF", 1.0),
    ".T": ("JPY", 1.0), ".HK": ("HKD", 1.0), ".AX": ("AUD", 1.0), ".MX": ("MXN", 1.0),
    ".SA": ("BRL", 1.0), ".KS": ("KRW", 1.0), ".ST": ("SEK", 1.0), ".OL": ("NOK", 1.0),
    ".CO": ("DKK", 1.0),
}
BASE_CURRENCY = "USD"


def quote_currency(ticker: str) -> tuple[str | None, float]:
    """Moneda en que cotiza un ticker de Yahoo y la escala de su precio.

    Sin sufijo: USD. Con un sufijo que no esta en la tabla: (None, 1.0), y quien
    llama avisa; suponer USD daria betas y tamanos equivocados sin decirlo.
    """
    t = ticker.upper()
    if t.startswith("^") or t.endswith("=X") or "." not in t:
        return BASE_CURRENCY, 1.0
    suffix = t[t.rindex("."):]
    return QUOTE_CURRENCY.get(suffix, (None, 1.0))


def fx_ticker(currency: str) -> str:
    """Ticker de Yahoo con los USD que vale una unidad de `currency`."""
    return f"{currency}{BASE_CURRENCY}=X"


def load_fx(currencies, start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Cierre diario de USD por unidad de cada moneda, y las que fallaron.

    Pasa por la misma cache que las acciones: el tipo de cambio de ayer no
    cambia y no se vuelve a pedir.
    """
    wanted = sorted({c for c in currencies if c and c != BASE_CURRENCY})
    if not wanted:
        return {}, {}
    market = load_ohlcv([fx_ticker(c) for c in wanted], start, end)
    rates, failures = {}, {}
    for c in wanted:
        frame = market.ohlcv.get(fx_ticker(c))
        close = None if frame is None else frame["Close"].where(frame["Close"] > 0).dropna()
        if close is None or close.empty:
            failures[c] = market.failures.get(fx_ticker(c), "sin datos")
        else:
            rates[c] = close[~close.index.duplicated(keep="last")].sort_index()
    return rates, failures


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
            if cache_usable(frame, written_at(_cache_name(ticker))):
                result.ohlcv[ticker] = settled(frame.loc[frame.index >= start])
                continue
        to_fetch.append(ticker)

    if to_fetch:
        log.info("descargando %d tickers de Yahoo", len(to_fetch))
        downloaded = download_with_retry(to_fetch, start, end)
        for ticker in to_fetch:
            frame = downloaded.get(ticker)
            if frame is None:
                # Si Yahoo falla y hay una cache de dias anteriores que cubre el
                # inicio, se usa y se avisa: perder el nombre entero por un corte
                # de Yahoo es peor que analizarlo con precios de hace unos dias.
                old = cache.read_frame(_cache_name(ticker))
                asked_from = requested.get(ticker)
                if (old is not None and not old.empty and asked_from is not None
                        and pd.Timestamp(asked_from) <= start + pd.Timedelta(days=7)):
                    frame = old.set_index(pd.to_datetime(old["date"])).drop(columns="date")
                    # La barra del dia en que se escribio pudo ser de media sesion.
                    when = written_at(_cache_name(ticker))
                    if when is not None:
                        frame = frame.loc[frame.index < when.normalize()]
                    frame = settled(frame)
                    result.ohlcv[ticker] = frame.loc[frame.index >= start]
                    result.stale[ticker] = frame.index.max()
                    continue
                result.failures[ticker] = "Yahoo returned no data (invalid ticker, delisted or no history)"
                continue
            frame = settled(frame)
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


_EARNINGS = "qe_earnings_dates"


def _next_earnings(ticker: str) -> pd.Timestamp | None:
    import yfinance as yf  # perezoso: los tests no tocan la red

    cal = yf.Ticker(ticker).calendar
    dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
    if not dates:
        return None
    stamps = sorted(pd.Timestamp(d) for d in (dates if isinstance(dates, (list, tuple)) else [dates]))
    return stamps[0]


def _next_ex_dividend(ticker: str) -> pd.Timestamp | None:
    import yfinance as yf  # perezoso: los tests no tocan la red

    cal = yf.Ticker(ticker).calendar
    d = cal.get("Ex-Dividend Date") if isinstance(cal, dict) else None
    return pd.Timestamp(d) if d is not None else None


def load_ex_dividend_dates(tickers, today: pd.Timestamp, timeout: float = 20.0,
                           fetch=None) -> dict[str, pd.Timestamp]:
    """Proxima fecha ex-dividendo de Yahoo. Un corto abierto ese dia paga el dividendo."""
    return load_earnings_dates(tickers, today, timeout, fetch or _next_ex_dividend, cache_name=_EX_DIVIDEND)


_EX_DIVIDEND = "qe_ex_dividend_dates"


def load_earnings_dates(tickers, today: pd.Timestamp, timeout: float = 20.0,
                        fetch=None, cache_name: str = "") -> dict[str, pd.Timestamp]:
    """Proxima fecha de resultados de cada ticker, la que da Yahoo.

    Solo para la ejecucion: un resultado antes de la salida por tiempo puede
    saltar el stop con un gap. Se cachea un dia. Lo que no llega en `timeout`
    segundos queda sin fecha y la ficha dice que hay que comprobarla: nunca se
    inventa una fecha ni se bloquea la corrida por ella.
    """
    import concurrent.futures as cf  # noqa: PLC0415

    fetch = fetch or _next_earnings
    cache_name = cache_name or _EARNINGS
    day = today.strftime("%Y-%m-%d")
    cached = cache.read_json(cache_name) or {}
    known = cached.get(day, {})
    out = {t: pd.Timestamp(v) for t, v in known.items() if v}
    todo = [t for t in tickers if t not in known]
    if todo:
        pool = cf.ThreadPoolExecutor(max_workers=min(8, len(todo)))
        futures = {pool.submit(fetch, t): t for t in todo}
        done, _ = cf.wait(futures, timeout=timeout)
        for f in done:
            t = futures[f]
            try:
                d = f.result()
            except Exception as exc:  # noqa: BLE001 -- sin fecha, la ficha lo dice
                log.info("sin fecha de resultados para %s: %s", t, exc)
                d = None
            known[t] = d.strftime("%Y-%m-%d") if d is not None and pd.notna(d) else None
            if d is not None and pd.notna(d):
                out[t] = pd.Timestamp(d)
        pool.shutdown(wait=False, cancel_futures=True)
        cache.write_json(cache_name, {day: known})
    # Una fecha ya pasada (Yahoo tarda en actualizar) no es la proxima.
    return {t: d for t, d in out.items() if d >= today.normalize()}

