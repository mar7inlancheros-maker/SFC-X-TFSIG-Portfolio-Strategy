"""Momentum y tendencia. Funciones puras sobre precios de un activo.

Indicadores clasicos con su definicion exacta, porque "RSI" o "MACD" sin
parametros no son reproducibles:

- RSI(14) con suavizado de Wilder (media exponencial con alpha = 1/14), no con
  media simple. Son numeros distintos y el de Wilder es el estandar.
- MACD(12, 26, 9) sobre el cierre. Se reporta el histograma DIVIDIDO por el
  precio: en dolares, el MACD de una accion de 800$ y el de una de 20$ no se
  pueden comparar.
- ADX(14), de Wilder: fuerza de la tendencia, sin direccion. Por encima de ~25
  hay tendencia; por debajo de ~20, no.

El score de momentum NO se calcula aqui: es un z-score de seccion cruzada y
necesita a todos los activos a la vez (ver `signals.composite`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SMA_WINDOWS = (20, 50, 100, 200)


def _wilder(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0.0), n)
    loss = _wilder((-delta).clip(lower=0.0), n)
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # Sin perdidas en la ventana el RSI es 100 por definicion, no NaN: una
    # accion con 14 sesiones seguidas al alza perdia su RSI justo cuando es mas
    # extremo. Sin movimiento alguno (ganancia y perdida cero) se deja en 50.
    out = out.where(~((loss == 0) & (gain > 0)), 100.0)
    return out.where(~((loss == 0) & (gain == 0)), 50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, n)
    plus_di = 100.0 * _wilder(plus_dm, n) / atr
    minus_di = 100.0 * _wilder(minus_dm, n) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return _wilder(dx, n)


def momentum_features(
    close: pd.Series, high: pd.Series | None = None, low: pd.Series | None = None
) -> dict[str, float]:
    s = close.dropna()
    out: dict[str, float] = {}
    if len(s) < 30:
        return out
    last = s.iloc[-1]

    def ret(n: int, skip: int = 0) -> float:
        if len(s) <= n:
            return float("nan")
        return float(s.iloc[-1 - skip] / s.iloc[-1 - n] - 1.0)

    out["mom_1m"] = ret(21)
    out["mom_3m"] = ret(63)
    out["mom_6m"] = ret(126)
    out["mom_12m"] = ret(252)
    # 12-1: se salta el ultimo mes por la reversion de corto plazo
    # (Jegadeesh 1990); es la construccion de Jegadeesh-Titman y de UMD.
    out["mom_12_1"] = ret(252, skip=21)

    for w in SMA_WINDOWS:
        sma = s.rolling(w).mean().iloc[-1] if len(s) >= w else np.nan
        out[f"sma_{w}"] = float(sma)
        out[f"price_to_sma_{w}"] = float(last / sma - 1.0) if sma and not np.isnan(sma) else float("nan")

    out["rsi_14"] = float(rsi(s).iloc[-1])
    m = macd(s)
    out["macd_hist_pct"] = float(m["hist"].iloc[-1] / last)
    if high is not None and low is not None:
        a = adx(high.reindex(s.index), low.reindex(s.index), s).dropna()
        out["adx_14"] = float(a.iloc[-1]) if len(a) else float("nan")
    else:
        out["adx_14"] = float("nan")
    return out


def trend_quality(close: pd.Series, window: int = 252) -> float:
    """signo(pendiente) x R2 del log-precio contra el tiempo, en `window` sesiones.

    +1: subida perfectamente lineal. -1: bajada perfectamente lineal. Cerca de
    0: sin tendencia o con una muy ruidosa. Recoge la idea de "informacion
    continua" de Da, Gurun y Warachka (2014): el momentum construido a base de
    muchos dias pequenos persiste mas que el hecho de pocos saltos grandes.
    """
    s = np.log(close.dropna().iloc[-window:])
    if len(s) < 60:
        return float("nan")
    t = np.arange(len(s), dtype=float)
    slope, intercept = np.polyfit(t, s.to_numpy(), 1)
    fitted = slope * t + intercept
    ss_res = float(((s.to_numpy() - fitted) ** 2).sum())
    ss_tot = float(((s.to_numpy() - s.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(np.sign(slope) * r2)
