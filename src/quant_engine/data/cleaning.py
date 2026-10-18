"""Alineacion de calendarios y retornos. Funciones puras.

Reglas, todas contadas y reportadas en la calidad de datos:

1. **Calendario = sesiones del benchmark.** Es la referencia de "dia de mercado".
2. **Duplicados**: se conserva la ultima observacion de cada fecha.
3. **Precios no positivos** se anulan: un cierre de 0 o negativo es un error de
   datos, no un precio.
4. **Relleno hacia delante limitado a `max_ffill` sesiones**, y solo dentro del
   rango en que el ticker cotiza. Cubre festivos que no coinciden entre
   mercados. Rellenar huecos largos inventaria retornos de cero seguidos de un
   salto, que aplanan la volatilidad y rompen las correlaciones.
5. **Retornos simples** diarios. Se agregan en el tiempo componiendo (1+r), y
   entre activos sumando pesos x retornos. Mezclar esto con retornos
   logaritmicos es el error clasico: los log-retornos no se suman entre activos.
6. **Todo en USD.** Un nombre que cotiza en otra moneda se convierte con el
   tipo de cambio del dia: su retorno, su beta frente a SPY y su liquidez son
   los de un inversor en dolares. Los precios locales se guardan aparte
   (`local_prices`): las ordenes, el stop y los objetivos se ponen en la
   moneda en que cotiza.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .loader import MarketData

EXTREME_RETURN = 0.50  # |r| diario por encima de esto se marca (no se borra)
STALE_RUN = 5          # sesiones seguidas con cierre identico


@dataclass
class CleanData:
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    returns: pd.DataFrame
    benchmark: str
    stats: dict[str, dict[str, float]] = field(default_factory=dict)
    fx: pd.DataFrame = field(default_factory=pd.DataFrame)  # USD por unidad de precio local
    currency: dict[str, str] = field(default_factory=dict)

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self.close.index

    def fx_rate(self, ticker: str) -> pd.Series:
        """USD por unidad de precio local en cada sesion (1 para los de USD)."""
        if ticker in self.fx.columns:
            return self.fx[ticker]
        return pd.Series(1.0, index=self.close.index)

    def local_prices(self, ticker: str) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Cierre, maximo y minimo en la moneda en que cotiza: para niveles y ordenes."""
        rate = self.fx_rate(ticker)
        return tuple(frame[ticker] / rate for frame in (self.close, self.high, self.low))  # type: ignore[return-value]


def _longest_stale_run(series: pd.Series) -> int:
    s = series.dropna()
    if s.empty:
        return 0
    same = s.diff().eq(0)
    groups = (~same).cumsum()
    return int(same.groupby(groups).sum().max() + 1) if same.any() else 1


def _ffill(series: pd.Series, limit: int) -> pd.Series:
    """Relleno hacia delante con tope. `limit = 0` significa no rellenar nada;
    pandas rechaza `ffill(limit=0)`, asi que se trata aparte."""
    return series.ffill(limit=limit) if limit > 0 else series


def fx_on_calendar(rate: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Tipo de cambio en cada sesion: el ultimo cierre conocido, nunca uno futuro.

    Divisas y bolsas no comparten festivos: se arrastra el ultimo cierre sin
    tope. Antes del primer dato no hay cambio y el precio queda sin convertir
    (NaN), no convertido con un cambio inventado.
    """
    full = rate.reindex(rate.index.union(calendar)).ffill()
    return full.reindex(calendar)


def align(market: MarketData, benchmark: str, *, max_ffill: int = 2,
          fx: dict[str, pd.Series] | None = None, currency: dict[str, str] | None = None) -> CleanData:
    """`fx`: ticker -> USD por unidad de su precio local (con la escala, p. ej.
    peniques). Los tickers que no estan se suponen ya en USD."""
    if benchmark not in market.ohlcv:
        raise ValueError(
            f"no hay datos del benchmark {benchmark}: sin el no hay calendario ni beta. "
            f"Motivo: {market.failures.get(benchmark, 'desconocido')}"
        )

    calendar = market.ohlcv[benchmark].index.drop_duplicates().sort_values()
    fx = fx or {}
    rates: dict[str, pd.Series] = {}
    fields: dict[str, dict[str, pd.Series]] = {f: {} for f in ("Close", "High", "Low", "Volume")}
    stats: dict[str, dict[str, float]] = {}

    for ticker, raw in market.ohlcv.items():
        n_raw = len(raw)
        frame = raw[~raw.index.duplicated(keep="last")].sort_index()
        duplicates = n_raw - len(frame)

        close = frame["Close"].where(frame["Close"] > 0)
        non_positive = int(frame["Close"].le(0).sum())

        first, last = close.first_valid_index(), close.last_valid_index()
        active = calendar[(calendar >= first) & (calendar <= last)] if first is not None else calendar[:0]
        reindexed = close.reindex(active)
        missing_before = int(reindexed.isna().sum())
        filled = _ffill(reindexed, max_ffill)
        missing_after = int(filled.isna().sum())

        rate = fx_on_calendar(fx[ticker], active) if ticker in fx else None
        if rate is not None:
            rates[ticker] = rate
        fields["Close"][ticker] = filled if rate is None else filled * rate
        for f in ("High", "Low", "Volume"):
            if f in frame.columns:
                col = frame[f].reindex(active)
                if f == "Volume":
                    fields[f][ticker] = col
                else:
                    col = _ffill(col, max_ffill)
                    fields[f][ticker] = col if rate is None else col * rate

        rets = filled.pct_change(fill_method=None)
        stats[ticker] = {
            "raw_obs": float(n_raw),
            "active_sessions": float(len(active)),
            "duplicates": float(duplicates),
            "non_positive": float(non_positive),
            "filled": float(missing_before - missing_after),
            "missing": float(missing_after),
            "missing_pct": float(missing_after / len(active)) if len(active) else 1.0,
            "extreme_returns": float((rets.abs() > EXTREME_RETURN).sum()),
            "stale_run": float(_longest_stale_run(filled)),
            "start": first,
            "end": last,
        }

    close = pd.DataFrame(fields["Close"]).reindex(calendar)
    returns = close.pct_change(fill_method=None)
    return CleanData(
        close=close,
        high=pd.DataFrame(fields["High"]).reindex(calendar),
        low=pd.DataFrame(fields["Low"]).reindex(calendar),
        volume=pd.DataFrame(fields["Volume"]).reindex(calendar),
        returns=returns,
        benchmark=benchmark,
        stats=stats,
        fx=pd.DataFrame(rates).reindex(calendar) if rates else pd.DataFrame(index=calendar),
        currency=dict(currency or {}),
    )


def compound(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Retorno acumulado componiendo: prod(1+r) - 1. Ignora NaN."""
    return (1.0 + returns.fillna(0.0)).prod() - 1.0


def annualize_return(total: float, years: float) -> float:
    if years <= 0 or total <= -1:
        return float("nan")
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def daily_rf(annual: float, periods: int = 252) -> float:
    """Tasa diaria equivalente por composicion, no annual/252."""
    return (1.0 + annual) ** (1.0 / periods) - 1.0


def log_prices(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close)
