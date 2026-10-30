"""Lectura tecnica de un nombre para cronometrar la entrada. Funciones puras.

El Quant Score decide la direccion; esto decide CUANDO y DONDE. Cada
indicador con su definicion exacta (los de Wilder, como en `momentum`):

- Tendencia: precio frente a SMA50 y SMA200, cruce SMA50/SMA200 y pendiente
  de la SMA200 en 20 sesiones.
- Fuerza: ADX(14) con +DI y -DI. ADX >= 25 hay tendencia; < 20 no.
- Sobrecompra / sobreventa: RSI(14) de Wilder; %B de Bollinger(20, 2).
- Compresion: ancho de Bollinger en su percentil de un ano (bajo = la
  volatilidad suele expandirse despues).
- Soportes y resistencias: pivotes confirmados (maximo o minimo de 5 sesiones
  a cada lado) en los ultimos 6 meses; el mas cercano por debajo y por encima.
- Volumen: media de 20 sesiones sobre la de 60.

Los consejos son de ejecucion, no de direccion: "esperar un rebote antes de
vender en corto un nombre sobrevendido" no cambia la senal, cambia el precio de
entrada. No hay evidencia de que estos indicadores predigan por si solos en
acciones grandes; por eso no entran en el score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .momentum import _wilder, macd, rsi


def earnings_moves(returns: pd.Series, report_dates=None, years: int = 3) -> dict[str, float] | None:
    """Tamano tipico del movimiento del dia de resultados.

    Con fechas de publicacion reales (`rdq` de Compustat): el mayor |retorno|
    entre ese dia y el siguiente (muchas empresas publican despues del cierre).
    Sin ellas, aproximacion: los cuatro dias de mayor |retorno| de cada ano, que
    en una accion suelen ser los de resultados. Se dice cual de las dos es.
    """
    r = returns.dropna()
    if len(r) < 252:
        return None
    r = r[r.index >= r.index[-1] - pd.DateOffset(years=years)]
    moves, source = [], "proxy: the 4 largest daily moves of each year"
    if report_dates is not None and len(report_dates):
        for d in pd.to_datetime(pd.Series(report_dates)).dropna().unique():
            pos = r.index.searchsorted(pd.Timestamp(d))
            window = r.iloc[pos:pos + 2]
            if len(window) and window.index[0] - pd.Timestamp(d) <= pd.Timedelta(days=4):
                moves.append(float(window.abs().max()))
        source = "actual report dates (Compustat)"
    if len(moves) < 4:
        moves = [float(x) for _, g in r.abs().groupby(r.index.year) for x in g.nlargest(4)]
        source = "proxy: the 4 largest daily moves of each year"
    if not moves:
        return None
    m = pd.Series(moves)
    return {"median": float(m.median()), "p75": float(m.quantile(0.75)), "n": len(m), "source": source}


def directional(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    """+DI, -DI y ADX de Wilder."""
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, n)
    plus_di = 100.0 * _wilder(plus_dm, n) / atr
    minus_di = 100.0 * _wilder(minus_dm, n) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": _wilder(dx, n), "atr": atr})


def pivots(high: pd.Series, low: pd.Series, lookback: int = 126, window: int = 5) -> tuple[list[float], list[float]]:
    """Maximos y minimos locales confirmados: (resistencias, soportes).

    Un pivote necesita `window` sesiones a cada lado: las ultimas `window` no
    pueden confirmar uno todavia.
    """
    h, l = high.iloc[-lookback:], low.iloc[-lookback:]
    hv, lv = h.to_numpy(float), l.to_numpy(float)
    res, sup = [], []
    for i in range(window, len(hv) - window):
        seg_h, seg_l = hv[i - window:i + window + 1], lv[i - window:i + window + 1]
        if np.isfinite(hv[i]) and hv[i] == np.nanmax(seg_h):
            res.append(float(hv[i]))
        if np.isfinite(lv[i]) and lv[i] == np.nanmin(seg_l):
            sup.append(float(lv[i]))
    return res, sup


def nearest_levels(price: float, resistances: list[float], supports: list[float]) -> tuple[float, float]:
    """Soporte mas cercano por debajo y resistencia mas cercana por encima del precio."""
    below = [x for x in supports + resistances if x < price]
    above = [x for x in resistances + supports if x > price]
    return (max(below) if below else np.nan), (min(above) if above else np.nan)


def technical_view(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series | None = None) -> dict:
    """Todos los indicadores de la ultima sesion, en la moneda en que cotiza."""
    c = close.dropna()
    if len(c) < 60:
        return {"available": False, "reason": "fewer than 60 sessions of prices"}
    h = high.reindex(c.index).fillna(c)
    lo = low.reindex(c.index).fillna(c)
    last = float(c.iloc[-1])
    sma = {w: float(c.rolling(w).mean().iloc[-1]) if len(c) >= w else np.nan for w in (20, 50, 200)}
    sma200 = c.rolling(200).mean()
    slope200 = float(sma200.iloc[-1] / sma200.iloc[-21] - 1) if len(sma200.dropna()) > 21 else np.nan
    di = directional(h, lo, c).iloc[-1]
    m = macd(c)["hist"]
    cross = ""
    signs = np.sign(m.iloc[-6:].to_numpy())
    flips = np.nonzero(np.diff(signs))[0]
    if len(flips):
        ago = len(signs) - 2 - flips[-1]
        cross = (f"{'bullish' if signs[-1] > 0 else 'bearish'} MACD cross "
                 f"{'today' if ago == 0 else f'{ago} session(s) ago'}")
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    upper, lower = mid + 2 * sd, mid - 2 * sd
    pct_b = float((last - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])) if sd.iloc[-1] > 0 else np.nan
    width = ((upper - lower) / mid).dropna()
    width_pct = float((width.iloc[-252:] < width.iloc[-1]).mean()) if len(width) > 20 else np.nan
    year = c.iloc[-252:]
    res, sup = pivots(h, lo)
    support, resistance = nearest_levels(last, res, sup)
    vol_ratio = np.nan
    if volume is not None:
        v = volume.reindex(c.index).dropna()
        if len(v) >= 60 and v.iloc[-60:].mean() > 0:
            vol_ratio = float(v.iloc[-20:].mean() / v.iloc[-60:].mean())
    above200 = np.isfinite(sma[200]) and last > sma[200]
    if np.isfinite(sma[200]) and above200 and sma[50] > sma[200] and slope200 > 0:
        trend = "UP"
    elif np.isfinite(sma[200]) and not above200 and sma[50] < sma[200] and slope200 < 0:
        trend = "DOWN"
    else:
        trend = "SIDEWAYS"
    r = float(rsi(c).iloc[-1])
    score = {"UP": 1.0, "DOWN": -1.0, "SIDEWAYS": 0.0}[trend] + (0.5 if m.iloc[-1] > 0 else -0.5) + \
        (0.5 if r > 50 else -0.5)
    stance = "BULLISH" if score >= 1.5 else "BEARISH" if score <= -1.5 else "NEUTRAL"
    return {
        "available": True, "price": last, "trend": trend, "stance": stance, "sma20": sma[20], "sma50": sma[50],
        "sma200": sma[200], "sma200_slope": slope200, "adx": float(di["adx"]), "plus_di": float(di["plus_di"]),
        "minus_di": float(di["minus_di"]), "atr": float(di["atr"]), "rsi": r, "macd_hist": float(m.iloc[-1]),
        "macd_cross": cross, "pct_b": pct_b, "band_width_pctile": width_pct,
        "high_52w": float(year.max()), "low_52w": float(year.min()),
        "from_high": last / float(year.max()) - 1, "from_low": last / float(year.min()) - 1,
        "support": support, "resistance": resistance, "volume_ratio": vol_ratio,
    }


def describe(t: dict) -> list[str]:
    """La lectura en frases cortas."""
    if not t.get("available"):
        return [f"Technical view unavailable: {t.get('reason')}"]
    strength = "strong" if t["adx"] >= 25 else "weak" if t["adx"] < 20 else "moderate"
    lines = [f"Trend {t['trend']} ({strength}, ADX {t['adx']:.0f}, +DI {t['plus_di']:.0f} / -DI {t['minus_di']:.0f}); "
             f"price vs SMA50 {t['price'] / t['sma50'] - 1:+.1%}"
             + (f", vs SMA200 {t['price'] / t['sma200'] - 1:+.1%}" if np.isfinite(t["sma200"]) else "")
             + f". RSI {t['rsi']:.0f}, Bollinger %B {t['pct_b']:.2f}"
             + (f", {t['macd_cross']}" if t["macd_cross"] else "") + "."]
    lvl = []
    if np.isfinite(t["support"]):
        lvl.append(f"support {t['support']:,.2f}")
    if np.isfinite(t["resistance"]):
        lvl.append(f"resistance {t['resistance']:,.2f}")
    lines.append((", ".join(lvl) + "; " if lvl else "")
                 + f"{t['from_high']:+.0%} from the 52-week high, {t['from_low']:+.0%} from the low"
                 + (f"; volume {t['volume_ratio']:.1f}x its 3-month average" if np.isfinite(t["volume_ratio"]) else "")
                 + (f"; volatility compressed (band width in its {t['band_width_pctile']:.0%} percentile)"
                    if np.isfinite(t["band_width_pctile"]) and t["band_width_pctile"] < 0.15 else "") + ".")
    return lines


def timing_advice(t: dict, side: int, entry: float, stop: float, tp1: float) -> list[str]:
    """Consejos de ejecucion para el lado que se va a operar (+1 largo, -1 corto)."""
    if not t.get("available") or side == 0:
        return []
    out = []
    atr = t["atr"]
    strong = t["adx"] >= 25
    if side > 0:
        if t["rsi"] > 70 or t["pct_b"] > 1:
            out.append(f"Extended (RSI {t['rsi']:.0f}, %B {t['pct_b']:.2f}): buying here chases the move; a pullback "
                       f"toward the SMA20 ({t['sma20']:,.2f}) gives a better entry and a smaller R.")
        if t["trend"] == "DOWN" and strong and t["minus_di"] > t["plus_di"]:
            out.append(f"Buying against a strong downtrend (ADX {t['adx']:.0f}): the stop is more likely to be hit; "
                       f"consider waiting for a close back above the SMA50 ({t['sma50']:,.2f}).")
        elif t["trend"] == "UP":
            out.append("The trend supports the long.")
        sup = t["support"]
        if np.isfinite(sup) and stop < sup and sup - stop < 0.5 * atr:
            out.append(f"Stop {stop:,.2f} sits just under support {sup:,.2f}: good placement.")
        elif np.isfinite(sup) and stop >= sup - 0.25 * atr and sup < entry:
            out.append(f"Support {sup:,.2f} is at or below the stop {stop:,.2f}: a test of support would stop you "
                       f"out; a stop under it ({sup - 0.25 * atr:,.2f}) avoids that but widens R.")
        res = t["resistance"]
        if np.isfinite(res) and entry < res < tp1:
            out.append(f"Resistance {res:,.2f} before TP1 {tp1:,.2f}: expect the move to stall there.")
    else:
        if t["rsi"] < 30 or t["pct_b"] < 0:
            out.append(f"Oversold (RSI {t['rsi']:.0f}, %B {t['pct_b']:.2f}): shorting here sells the low; a bounce "
                       f"toward the SMA20 ({t['sma20']:,.2f}) gives a better entry and a smaller R.")
        if t["trend"] == "UP" and strong and t["plus_di"] > t["minus_di"]:
            out.append(f"Shorting into a strong uptrend (ADX {t['adx']:.0f}): the stop is more likely to be hit; "
                       f"consider waiting for a close below the SMA50 ({t['sma50']:,.2f}).")
        elif t["trend"] == "DOWN":
            out.append("The trend supports the short.")
        res = t["resistance"]
        if np.isfinite(res) and stop > res and stop - res < 0.5 * atr:
            out.append(f"Stop {stop:,.2f} sits just above resistance {res:,.2f}: good placement.")
        elif np.isfinite(res) and stop <= res + 0.25 * atr and res > entry:
            out.append(f"Resistance {res:,.2f} is at or above the stop {stop:,.2f}: a test of resistance would stop "
                       f"you out; a stop above it ({res + 0.25 * atr:,.2f}) avoids that but widens R.")
        sup = t["support"]
        if np.isfinite(sup) and tp1 < sup < entry:
            out.append(f"Support {sup:,.2f} before TP1 {tp1:,.2f}: expect the move to stall there.")
    return out
