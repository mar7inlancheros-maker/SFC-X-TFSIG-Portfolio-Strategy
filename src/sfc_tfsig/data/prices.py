"""Precios diarios ajustados y las series que se derivan de ellos.

Fuente: Yahoo via `yfinance`, con `auto_adjust=True` -- el cierre ya viene
ajustado por splits y dividendos, que es lo unico con lo que se puede calcular
un retorno total honesto.

**Sesgo de supervivencia, declarado.** Yahoo solo devuelve tickers que existen
HOY. Las empresas que quebraron, se fusionaron o fueron excluidas del mercado no
estan. Un universo construido con esta fuente esta, por construccion, limpio de
los desastres -- y un backtest sobre ese universo sobreestima el retorno. La
magnitud tipica en renta variable estadounidense es de 1 a 2 puntos de CAGR al
ano. No se corrige con codigo: se declara, se descuenta al interpretar, y se
compara siempre contra un benchmark que sufre el mismo sesgo en menor medida.
Ver README, "Limitaciones declaradas".

**Cache incremental.** Se guarda un unico marco largo maestro; cada corrida
descarga solo los tickers o los tramos de fecha que falten. Una actualizacion
mensual baja un mes, no quince anos.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from . import cache

_MASTER = "prices_master"
_FAILURES = "prices_failures"
_BATCH = 150  # tickers por peticion; por encima, Yahoo empieza a devolver huecos

# Cuantos dias se respeta un fallo de descarga antes de volver a intentarlo. En
# cada corrida, alrededor del 10% de los tickers de la lista de la SEC no
# devuelve nada: nombres retirados, tickers reasignados, o simplemente huecos de
# Yahoo. Sin memoria de esos fallos, cada corrida vuelve a pedirlos, tarda lo
# mismo la decima vez que la primera, y el log se llena de ruido que tapa los
# fallos nuevos. Con memoria, se reintentan una vez por semana por si vuelven.
_FAILURE_TTL_DAYS = 7

# Un ticker que vuelve sin datos puede no tener datos (retirado, reasignado) o
# haber chocado con un error de Yahoo: en lotes de 150, Yahoo a veces corta y
# yfinance anota "TypeError: 'NoneType' object is not subscriptable" para
# empresas vivas y liquidas (VRT, TPG, OZK en octubre de 2026). Lo primero es
# un hecho; lo segundo, ruido que se reintenta. Estos mensajes son "no hay".
_NO_DATA = re.compile(r"delisted|no price data|no data found|no timezone|not found", re.IGNORECASE)
_RETRY_BATCH = 25
_RETRY_PAUSE_S = 3.0

# Un ticker sin ningun precio en los 30 dias previos a su ultima descarga se da
# por muerto (retirado, comprado, reasignado) y su cola se pide una vez al mes,
# no en cada corrida. Sin esto, desde que la cola se pide en cuanto falta un
# dia habil, cada corrida diaria volvia a pedir a Yahoo todo el universo,
# muertos de 2018 incluidos. Un valor suspendido mas de un mes se actualiza con
# hasta un mes de retraso: es el precio de no pedir miles de colas vacias.
_DEAD_AFTER = pd.Timedelta(days=30)
_DEAD_RECHECK = pd.Timedelta(days=31)


class PriceError(RuntimeError):
    """No se pudieron obtener precios utilizables."""


def _import_yfinance():
    """Importacion perezosa: el nucleo del modelo no depende de yfinance."""
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise PriceError(
            "falta yfinance. Instala con: pip install yfinance"
        ) from exc
    return yf


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "ticker": pd.Series(dtype="object"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )


def _tidy(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Respuesta de yfinance (columnas MultiIndex) -> marco largo."""
    if raw is None or raw.empty:
        return _empty_long()

    if not isinstance(raw.columns, pd.MultiIndex):
        # Un solo ticker: yfinance aplana las columnas.
        raw = pd.concat({tickers[0]: raw}, axis=1).swaplevel(axis=1)

    frames = []
    for field, column in (("close", "Close"), ("volume", "Volume")):
        if column not in raw.columns.get_level_values(0):
            continue
        block = raw[column].copy()
        block.index.name = "date"
        long = block.reset_index().melt(id_vars="date", var_name="ticker", value_name=field)
        frames.append(long.set_index(["date", "ticker"]))

    if not frames:
        return _empty_long()

    out = pd.concat(frames, axis=1).reset_index()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out = out.dropna(subset=["close"])
    if "volume" not in out.columns:
        out["volume"] = np.nan
    return out[["date", "ticker", "close", "volume"]].sort_values(["ticker", "date"])


def _download_batch(tickers: Sequence[str], start, end) -> tuple[pd.DataFrame, dict[str, str]]:
    """Precios de un lote y el error de cada ticker que fallo, segun yfinance."""
    yf = _import_yfinance()
    raw = yf.download(
        list(tickers),
        start=pd.Timestamp(start).strftime("%Y-%m-%d"),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
        actions=False,
    )
    # yfinance vacia este registro al empezar cada descarga: hay que leerlo ya.
    errors = getattr(getattr(yf, "shared", None), "_ERRORS", None) or {}
    return _tidy(raw, list(tickers)), {str(k).upper(): str(v) for k, v in errors.items()}


def _drop_unsettled(frame: pd.DataFrame, today=None) -> pd.DataFrame:
    """Quita la barra de hoy: con el mercado abierto no es un cierre."""
    if frame.empty:
        return frame
    today = (pd.Timestamp(today) if today is not None else pd.Timestamp.today()).normalize()
    return frame[pd.to_datetime(frame["date"]) < today]


def transient_failures(missing: Iterable[str], errors: dict[str, str]) -> list[str]:
    """De los tickers que no trajeron datos, los que fallaron por un error de
    Yahoo y no porque no haya datos. Funcion pura."""
    return [t for t in missing if t in errors and not _NO_DATA.search(errors[t])]


_LEDGER = "prices_fetched"
_TOLERANCE = pd.Timedelta(days=5)  # fines de semana y festivos no son huecos


def settled_end(end, today=None) -> pd.Timestamp:
    """Ultimo dia con cierre definitivo dentro de lo pedido: nunca hoy.

    Hoy, con el mercado abierto, Yahoo devuelve una barra a medias. El 30 de
    septiembre de 2026 una descarga de las 10:40 guardo como cierre del dia un
    precio de media sesion, y el registro dio el dia por bajado.
    """
    today = (pd.Timestamp(today) if today is not None else pd.Timestamp.today()).normalize()
    return min(pd.Timestamp(end).normalize(), today - pd.Timedelta(days=1))


def plan_downloads(
    wanted: Sequence[str],
    ledger: dict[str, dict[str, str]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp | None = None,
    last_seen: dict[str, pd.Timestamp] | None = None,
) -> list[tuple[list[str], pd.Timestamp, pd.Timestamp]]:
    """Decide que pedir a Yahoo y con que rango. Funcion pura.

    Devuelve trabajos `(tickers, desde, hasta)`. Se decide con el REGISTRO de lo
    ya solicitado, no con las fechas que hay en la cache. La diferencia importa:

    - Una empresa que salio a bolsa en 2015 no tiene precios de 2009, por mucho
      que se pidan. Mirando la cache, su primer precio (2015) queda despues del
      inicio pedido (2009) y parece que "falta historia": se volvia a pedir la
      serie completa EN CADA CORRIDA, para siempre. Unos 2.400 tickers del
      universo estaban en ese caso.
    - Una empresa que dejo de cotizar en 2018 tampoco tendra precios nuevos.
      Mirando la cache, su ultimo precio queda antes del final pedido y se
      volvia a pedir en cada corrida.

    Con el registro, cada tramo se pide una vez. Ademas, lo que falta por la
    derecha se pide SOLO desde donde se quedo la ultima descarga: una
    actualizacion mensual baja un mes, no quince anos del universo entero.
    """
    full: list[str] = []
    tails: dict[pd.Period, list[str]] = {}
    tail_from: dict[pd.Period, pd.Timestamp] = {}

    settled = settled_end(end, today)
    last_session = pd.bdate_range(end=settled, periods=1)[0]

    for ticker in wanted:
        entry = ledger.get(ticker)
        if entry is None:
            full.append(ticker)
            continue
        fetched_from = pd.Timestamp(entry["from"])
        fetched_to = pd.Timestamp(entry["to"])
        if fetched_from > start + _TOLERANCE:
            # Nunca se pidio desde tan atras: serie completa.
            full.append(ticker)
            continue
        # Por la derecha basta UN dia habil sin pedir. La tolerancia de 5 dias
        # dejaba sin pedir los ultimos dias: con lo bajado el 30 de septiembre,
        # una corrida del 4 de octubre no traia ni el 1 ni el 2, y el backtest
        # no tenia precio para ejecutar el rebalanceo de fin de mes. Un festivo
        # cuesta una peticion vacia, una vez.
        if fetched_to < last_session:
            seen = last_seen.get(ticker) if last_seen is not None else fetched_to
            dead = seen is None or pd.Timestamp(seen) < fetched_to - _DEAD_AFTER
            if dead and fetched_to >= settled - _DEAD_RECHECK:
                continue
            # Se agrupa por mes de la ultima descarga para no arrastrar a todos
            # los tickers hasta la fecha del mas atrasado.
            bucket = fetched_to.to_period("M")
            tails.setdefault(bucket, []).append(ticker)
            tail_from[bucket] = min(tail_from.get(bucket, fetched_to), fetched_to)

    jobs: list[tuple[list[str], pd.Timestamp, pd.Timestamp]] = []
    if full:
        jobs.append((sorted(full), start, end))
    for bucket in sorted(tails):
        jobs.append((sorted(tails[bucket]), tail_from[bucket] - _TOLERANCE, end))
    return jobs


def _load_ledger(master: pd.DataFrame, coverage: pd.DataFrame | None = None) -> dict[str, dict[str, str]]:
    """Registro de lo solicitado. Si no existe, se inicializa desde la cache.

    La inicializacion es conservadora: toma las fechas reales de la cache como
    lo solicitado. Eso provoca UNA descarga extra de los tickers con historia
    corta; a partir de ahi el registro ya sabe que se pidieron desde el inicio.
    """
    ledger = cache.read_json(_LEDGER) or {}
    if not master.empty:
        if coverage is None:
            coverage = master.groupby("ticker")["date"].agg(["min", "max"])
        for ticker, row in coverage.iterrows():
            if ticker not in ledger:
                ledger[ticker] = {
                    "from": pd.Timestamp(row["min"]).strftime("%Y-%m-%d"),
                    "to": pd.Timestamp(row["max"]).strftime("%Y-%m-%d"),
                }
    return ledger


def _update_ledger(ledger: dict, tickers: Sequence[str], start, end) -> None:
    """Anota que estos tickers se pidieron en [start, end], vinieran datos o no."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for ticker in tickers:
        entry = ledger.get(ticker)
        if entry is None:
            ledger[ticker] = {"from": s.strftime("%Y-%m-%d"), "to": e.strftime("%Y-%m-%d")}
        else:
            entry["from"] = min(pd.Timestamp(entry["from"]), s).strftime("%Y-%m-%d")
            entry["to"] = max(pd.Timestamp(entry["to"]), e).strftime("%Y-%m-%d")


def _recent_failures() -> set[str]:
    """Tickers que fallaron hace menos de `_FAILURE_TTL_DAYS` dias."""
    ledger = cache.read_json(_FAILURES) or {}
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=_FAILURE_TTL_DAYS)
    return {
        ticker for ticker, stamp in ledger.items()
        if pd.Timestamp(stamp) >= cutoff
    }


def _record_failures(tickers: list[str]) -> None:
    """Anota que estos tickers no devolvieron datos, con la fecha de hoy."""
    if not tickers:
        return
    ledger = cache.read_json(_FAILURES) or {}
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    ledger.update({ticker: today for ticker in tickers})
    cache.write_json(_FAILURES, ledger)


def get_prices(
    tickers: Iterable[str],
    start,
    end,
    *,
    refresh: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Precios diarios ajustados en formato largo: date, ticker, close, volume.

    Descarga solo lo que falta respecto a la cache maestra.
    """
    wanted = sorted({str(t).upper().strip() for t in tickers if str(t).strip()})
    if not wanted:
        return _empty_long()

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # Ojo: `cache.read_frame(...) or _empty_long()` NO vale, el valor de verdad
    # de un DataFrame es ambiguo y lanza ValueError.
    master = None if refresh else cache.read_frame(_MASTER)
    if master is None:
        master = _empty_long()
    if not master.empty:
        master["date"] = pd.to_datetime(master["date"])

    coverage = (master.groupby("ticker")["date"].agg(["min", "max"]) if not master.empty
                else pd.DataFrame(columns=["min", "max"]))
    ledger = _load_ledger(master, coverage)
    jobs = [] if refresh else plan_downloads(wanted, ledger, start, end,
                                              last_seen=coverage["max"].to_dict())
    if refresh:
        jobs = [(wanted, start, end)]

    # Los que ya fallaron hace poco no se vuelven a pedir.
    if jobs and not refresh:
        recent_failures = _recent_failures()
        skipped = sorted({t for tickers, _, _ in jobs for t in tickers if t in recent_failures})
        if skipped:
            jobs = [([t for t in tickers if t not in recent_failures], s, e) for tickers, s, e in jobs]
            jobs = [job for job in jobs if job[0]]
            if progress:
                print(f"  precios: {len(skipped)} tickers omitidos por fallo reciente "
                      f"(se reintentan tras {_FAILURE_TTL_DAYS} dias)", flush=True)

    if jobs:
        downloaded: list[pd.DataFrame] = []
        failures: list[str] = []
        unresolved: list[str] = []
        for job_tickers, job_start, job_end in jobs:
            batches = [job_tickers[i : i + _BATCH] for i in range(0, len(job_tickers), _BATCH)]
            obtained: set[str] = set()
            pending: list[str] = []
            for i, batch in enumerate(batches, start=1):
                if progress:
                    print(f"  precios: {job_start.date()}..{job_end.date()} "
                          f"lote {i}/{len(batches)} ({len(batch)} tickers)", flush=True)
                frame, errors = _download_batch(batch, job_start, job_end)
                frame = _drop_unsettled(frame)
                downloaded.append(frame)
                got = set(frame["ticker"]) if not frame.empty else set()
                obtained |= got
                pending.extend(transient_failures([t for t in batch if t not in got], errors))

            # Un error de Yahoo se reintenta en lotes pequenos. Lo que siga
            # fallando NO se anota como pedido: si se anotara, el registro daria
            # por bajado un tramo que no llego y ese hueco no se volveria a pedir
            # nunca. Tampoco cuenta como fallo de 7 dias: no es que no haya datos.
            if pending:
                if progress:
                    print(f"  precios: {len(pending)} tickers con error de Yahoo; "
                          f"reintento en lotes de {_RETRY_BATCH}", flush=True)
                time.sleep(_RETRY_PAUSE_S)
                still: list[str] = []
                for i in range(0, len(pending), _RETRY_BATCH):
                    chunk = pending[i : i + _RETRY_BATCH]
                    frame, errors = _download_batch(chunk, job_start, job_end)
                    frame = _drop_unsettled(frame)
                    downloaded.append(frame)
                    got = set(frame["ticker"]) if not frame.empty else set()
                    obtained |= got
                    still.extend(t for t in chunk if t not in got)
                pending = still
                unresolved.extend(pending)

            done = [t for t in job_tickers if t not in set(pending)]
            # Solo es fallo si se pidio la serie completa y no vino nada.
            # Una cola vacia es normal: la empresa dejo de cotizar.
            if job_start <= start + _TOLERANCE:
                failures.extend(t for t in done if t not in obtained)
            _update_ledger(ledger, done, job_start, settled_end(job_end))

        _record_failures(failures)
        cache.write_json(_LEDGER, ledger)
        if unresolved and progress:
            print(f"  precios: {len(unresolved)} tickers siguen con error de Yahoo y se piden en la "
                  f"proxima corrida: {', '.join(sorted(unresolved)[:15])}"
                  + (" ..." if len(unresolved) > 15 else ""), flush=True)

        new = pd.concat(downloaded, ignore_index=True) if downloaded else _empty_long()
        if not new.empty:
            master = pd.concat([master, new], ignore_index=True)
            master = (
                master.sort_values(["ticker", "date"])
                .drop_duplicates(subset=["ticker", "date"], keep="last")
                .reset_index(drop=True)
            )
            cache.write_frame(_MASTER, master)

    if master.empty:
        raise PriceError(
            "no se obtuvo ningun precio. Revisa la conexion y que los tickers "
            "sean validos en Yahoo (los canadienses de TSX llevan sufijo .TO)"
        )

    mask = master["ticker"].isin(wanted) & master["date"].between(start, end)
    return master.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
#  Transformaciones (puras: se prueban con series sinteticas)
# ---------------------------------------------------------------------------


def to_wide(long: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Marco largo -> matriz fechas x tickers."""
    if long.empty:
        return pd.DataFrame()
    return long.pivot_table(index="date", columns="ticker", values=field, aggfunc="last").sort_index()


def month_end(wide: pd.DataFrame) -> pd.DataFrame:
    """Ultimo valor observado de cada mes natural."""
    if wide.empty:
        return wide
    return wide.resample("ME").last()


def monthly_returns(close_wide: pd.DataFrame) -> pd.DataFrame:
    """Retorno total mensual a partir del cierre ajustado."""
    monthly = month_end(close_wide)
    return monthly.pct_change(fill_method=None)


def momentum(close_wide: pd.DataFrame, lookback_m: int = 12, skip_m: int = 1) -> pd.DataFrame:
    """Momentum 12-1: retorno de `lookback_m` meses saltando los ultimos `skip_m`.

    El salto no es decoracion: el ultimo mes revierte con fuerza suficiente para
    anular la senal si se incluye (Jegadeesh 1990). Es la construccion estandar
    desde Jegadeesh-Titman (1993) y la que usa Fama-French para UMD.
    """
    monthly = month_end(close_wide)
    if monthly.empty:
        return monthly
    numerator = monthly.shift(skip_m)
    denominator = monthly.shift(lookback_m)
    return (numerator / denominator) - 1.0


def realized_volatility(close_wide: pd.DataFrame, window_d: int = 252) -> pd.DataFrame:
    """Volatilidad anualizada de los retornos diarios, muestreada a fin de mes."""
    if close_wide.empty:
        return close_wide
    daily = close_wide.pct_change(fill_method=None)
    vol = daily.rolling(window_d, min_periods=max(60, window_d // 3)).std() * np.sqrt(252.0)
    return month_end(vol)


def median_dollar_volume(long: pd.DataFrame, window_d: int = 63) -> pd.DataFrame:
    """Mediana movil del volumen en dolares, a fin de mes.

    Mediana y no media: un solo dia de volumen extraordinario (una inclusion en
    indice, un resultado) no debe hacer pasar por liquida a una accion que no lo
    es el resto del tiempo.
    """
    if long.empty:
        return pd.DataFrame()
    df = long.copy()
    df["dollar_volume"] = df["close"] * df["volume"]
    wide = df.pivot_table(index="date", columns="ticker", values="dollar_volume", aggfunc="last")
    rolling = wide.rolling(window_d, min_periods=max(20, window_d // 3)).median()
    return month_end(rolling)


def forward_returns(close_wide: pd.DataFrame, horizon_m: int = 1) -> pd.DataFrame:
    """Retorno FUTURO a `horizon_m` meses, alineado con la fecha de la senal.

    Solo para validacion (IC, quantiles, Fama-MacBeth). El backtest NO usa esto:
    ejecuta y contabiliza operacion a operacion. Mezclarlos es como se cuelan los
    look-ahead que nadie ve hasta que el dinero es real.
    """
    monthly = month_end(close_wide)
    return (monthly.shift(-horizon_m) / monthly) - 1.0
