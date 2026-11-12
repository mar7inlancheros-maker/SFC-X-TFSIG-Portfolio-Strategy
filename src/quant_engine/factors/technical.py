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

Los avisos son de ejecucion, no de direccion, y no entran en el score. Los
tres que afirmaban algo del resultado (extendido, contra una tendencia fuerte,
a favor de la tendencia larga) se midieron con la regla del plan en nombres
que no son los 50 pares (`tools/qe_rule_studies.py timing`): ninguno movio el
E(R). La ficha da la condicion y la cifra medida (`TIMING_EVIDENCE`), no un
consejo de esperar.
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


def indicator_series(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """Los indicadores de los consejos de entrada, sesion a sesion.

    Todo es causal (medias moviles y suavizados de Wilder): el valor de la
    sesion `d` solo usa precios hasta `d`. `technical_view` toma la ultima fila
    y `tools/qe_rule_studies.py timing` toma la de cada entrada: la ficha y el
    estudio que la valida miden lo mismo.
    """
    c = close.dropna()
    h = high.reindex(c.index).fillna(c)
    lo = low.reindex(c.index).fillna(c)
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    slope200 = sma200 / sma200.shift(20) - 1
    up = (c > sma200) & (sma50 > sma200) & (slope200 > 0)
    down = (c <= sma200) & (sma50 < sma200) & (slope200 < 0)
    trend = pd.Series(np.where(up, "UP", np.where(down, "DOWN", "SIDEWAYS")), index=c.index)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    band = 4 * sd
    pct_b = ((c - (mid - 2 * sd)) / band).where(band > 0)
    di = directional(h, lo, c)
    out = pd.DataFrame({"close": c, "sma50": sma50, "sma200": sma200, "sma200_slope": slope200, "trend": trend,
                        "rsi": rsi(c), "pct_b": pct_b, "adx": di["adx"], "plus_di": di["plus_di"],
                        "minus_di": di["minus_di"], "atr": di["atr"]})
    flags = timing_flags(out)
    for side, names in flags.items():
        for name, value in names.items():
            out[f"{'long' if side > 0 else 'short'}_{name}"] = value
    return out


STRONG_ADX = 25


def timing_flags(t) -> dict[int, dict[str, object]]:
    """Las condiciones de los consejos de entrada, por lado (+1 largo, -1 corto).

    `t` es la lectura de una sesion (dict) o la tabla de `indicator_series`:
    con escalares devuelve booleanos y con columnas, series. Una sola
    definicion para la ficha y para el estudio que la valida.
    """
    strong = t["adx"] >= STRONG_ADX
    return {
        1: {"extended": (t["rsi"] > 70) | (t["pct_b"] > 1),
            "against_trend": (t["trend"] == "DOWN") & strong & (t["minus_di"] > t["plus_di"]),
            "with_trend": t["trend"] == "UP"},
        -1: {"extended": (t["rsi"] < 30) | (t["pct_b"] < 0),
             "against_trend": (t["trend"] == "UP") & strong & (t["plus_di"] > t["minus_di"]),
             "with_trend": t["trend"] == "DOWN"},
    }


def technical_view(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series | None = None) -> dict:
    """Todos los indicadores de la ultima sesion, en la moneda en que cotiza."""
    c = close.dropna()
    if len(c) < 60:
        return {"available": False, "reason": "fewer than 60 sessions of prices"}
    h = high.reindex(c.index).fillna(c)
    lo = low.reindex(c.index).fillna(c)
    last = float(c.iloc[-1])
    sma = {w: float(c.rolling(w).mean().iloc[-1]) if len(c) >= w else np.nan for w in (20, 50, 200)}
    ind = indicator_series(c, h, lo).iloc[-1]
    slope200 = float(ind["sma200_slope"])
    di = ind
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
    pct_b = float(ind["pct_b"])
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
    trend = str(ind["trend"])
    r = float(ind["rsi"])
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
    # Dos horizontes: la tendencia sale de las medias de 50 y 200 dias; ADX y DI,
    # del movimiento de 14 dias. Juntos en una frase ("DOWN (strong, +DI > -DI)")
    # parecian contradecirse cuando el rebote corto va contra la tendencia larga.
    short_dir = "up" if t["plus_di"] > t["minus_di"] else "down"
    lines = [f"Long-term trend {t['trend']} (50/200-day averages); 14-day direction {short_dir}, {strength} "
             f"(ADX {t['adx']:.0f}, +DI {t['plus_di']:.0f} / -DI {t['minus_di']:.0f}); "
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


# `python tools/qe_rule_studies.py timing --end 2026-10-30`: la regla del plan
# (niveles de cada lado, sin stop por tiempo) abierta la primera sesion tras
# cada cierre de mes, jul-2023 a jun-2026, en los 600 nombres mas liquidos de
# cada mes sin los 50 pares (870 distintos, 21.550 entradas por lado).
# `diff` es la media mensual de E(R) con la condicion menos sin ella, t de
# Newey-West (retardo 3) y p ajustado por Holm sobre las seis afirmaciones;
# `stopped` es la fraccion de entradas que acaban en el stop (con, sin).
# Si cambia la regla del plan o una condicion de `timing_flags`, se vuelve a
# correr y se actualiza.
# Yahoo reajusta la historia con cada dividendo: volver a correr mueve las cifras
# en el segundo decimal (la de largos contra tendencia paso de -0,091 a -0,080).
# `ci` es el intervalo del 95% de `diff`: lo que cabe aun sin efecto medible.
TIMING_SAMPLE = "870 liquid names excluding the 50 reference pairs, 36 month-ends to Jun-2026"
TIMING_EVIDENCE: dict[tuple[int, str], dict[str, object]] = {
    (1, "extended"): {"diff": 0.070, "ci": (-0.034, 0.175), "t": 1.31, "p_holm": 1.0, "verdict": "NOT SUPPORTED",
                      "stopped": (0.53, 0.53)},
    (1, "against_trend"): {"diff": -0.080, "ci": (-0.256, 0.096), "t": -0.89, "p_holm": 1.0,
                           "verdict": "NOT SUPPORTED", "stopped": (0.51, 0.53)},
    (1, "with_trend"): {"diff": 0.072, "ci": (-0.058, 0.203), "t": 1.08, "p_holm": 1.0, "verdict": "NOT SUPPORTED",
                        "stopped": (0.54, 0.52)},
    (-1, "extended"): {"diff": -0.049, "ci": (-0.171, 0.073), "t": -0.78, "p_holm": 1.0, "verdict": "NOT SUPPORTED",
                       "stopped": (0.57, 0.53)},
    (-1, "against_trend"): {"diff": -0.036, "ci": (-0.096, 0.023), "t": -1.19, "p_holm": 1.0,
                            "verdict": "NOT SUPPORTED", "stopped": (0.51, 0.53)},
    (-1, "with_trend"): {"diff": 0.008, "ci": (-0.076, 0.092), "t": 0.19, "p_holm": 1.0, "verdict": "NOT SUPPORTED",
                         "stopped": (0.55, 0.52)},
    # `python tools/qe_rule_studies.py levels --end 2026-10-30`: mismas entradas,
    # soportes y resistencias de `pivots` con precios hasta la entrada; Holm
    # sobre estas seis. `tp1`: fraccion que llega al TP1 (con, sin); `share`:
    # fraccion de entradas con la condicion.
    (1, "stop_behind_level"): {"diff": 0.015, "ci": (-0.111, 0.142), "t": 0.24, "p_holm": 0.962,
                               "verdict": "NOT SUPPORTED", "stopped": (0.52, 0.53), "tp1": (0.31, 0.32),
                               "share": 0.031},
    (1, "level_inside_stop"): {"diff": 0.043, "ci": (-0.072, 0.157), "t": 0.73, "p_holm": 0.962,
                               "verdict": "NOT SUPPORTED", "stopped": (0.53, 0.53), "tp1": (0.34, 0.32),
                               "share": 0.069},
    (1, "level_before_tp1"): {"diff": -0.078, "ci": (-0.232, 0.076), "t": -0.99, "p_holm": 0.962,
                              "verdict": "NOT SUPPORTED", "stopped": (0.53, 0.55), "tp1": (0.32, 0.33),
                              "share": 0.871},
    (-1, "stop_behind_level"): {"diff": 0.036, "ci": (-0.024, 0.095), "t": 1.18, "p_holm": 0.950,
                                "verdict": "NOT SUPPORTED", "stopped": (0.52, 0.53), "tp1": (0.48, 0.47),
                                "share": 0.071},
    (-1, "level_inside_stop"): {"diff": -0.032, "ci": (-0.069, 0.004), "t": -1.72, "p_holm": 0.426,
                                "verdict": "NOT SUPPORTED", "stopped": (0.56, 0.52), "tp1": (0.44, 0.47),
                                "share": 0.106},
    (-1, "level_before_tp1"): {"diff": 0.046, "ci": (0.009, 0.082), "t": 2.47, "p_holm": 0.082,
                               "verdict": "NOT SUPPORTED", "stopped": (0.53, 0.53), "tp1": (0.47, 0.46),
                               "share": 0.801},
}
_VERDICT_TEXT = {
    "SUPPORTED": "the measurement backs this condition",
    "CONTRADICTED": "the measurement points the other way",
    "NOT SUPPORTED": "no measurable effect on the rule's E(R), so it is no reason to change the order",
}


def evidence_note(side: int, flag: str) -> str:
    """Lo que se midio de una condicion de la ficha, en una frase."""
    e = TIMING_EVIDENCE.get((side, flag))
    if not e:
        return ""
    stopped, tp1, share = e.get("stopped"), e.get("tp1"), e.get("share")
    stop_txt = f", stopped out {stopped[0]:.0%} vs {stopped[1]:.0%}" if stopped else ""
    tp1_txt = f", TP1 reached {tp1[0]:.0%} vs {tp1[1]:.0%}" if tp1 else ""
    # Una condicion que se cumple en casi todas las entradas no distingue nada.
    common = f" It holds on {share:.0%} of entries, so it says little." if share and share >= 0.5 else ""
    # El intervalo dice cuanto efecto cabe aun: "sin efecto medible" no es "efecto cero".
    ci = e.get("ci")
    size = (f" (95% CI {ci[0]:+.2f} to {ci[1]:+.2f}R)" if ci else f", t {e['t']:+.1f}")
    verdict = _VERDICT_TEXT[str(e["verdict"])]
    if ci and e["verdict"] == "NOT SUPPORTED" and (ci[0] > 0 or ci[1] < 0):
        # El intervalo crudo excluye el cero pero la familia de seis pruebas no:
        # decirlo, o el lector ve una contradiccion entre la cifra y el veredicto.
        verdict = (f"the interval excludes zero on its own, but not after Holm's correction for the six claims "
                   f"tested (p {e['p_holm']:.2f}), so it is no reason to change the order")
    return (f" Measured ({TIMING_SAMPLE}): {e['diff']:+.2f}R vs other entries the same month{size}"
            f"{stop_txt}{tp1_txt}; {verdict}.{common}")


def level_flags(t, side: int, entry: float, stop: float, tp1: float) -> dict[str, bool]:
    """Donde caen el stop y el TP1 frente a soportes y resistencias.

    - `stop_behind_level`: el stop queda justo detras del nivel (a menos de
      0,5 ATR): un test del nivel no lo toca.
    - `level_inside_stop`: el nivel esta en el stop o del lado de la entrada
      a menos de 0,25 ATR: un test del nivel toca el stop.
    - `level_before_tp1`: hay un nivel entre la entrada y el TP1.

    `t` necesita `atr`, `support` y `resistance` (de `technical_view` o de
    `nearest_levels` en la sesion de entrada). La ficha y
    `tools/qe_rule_studies.py levels` usan esta funcion.
    """
    atr = t["atr"]
    behind, front = (t["support"], t["resistance"]) if side > 0 else (t["resistance"], t["support"])
    if side > 0:
        good = bool(np.isfinite(behind) and stop < behind and behind - stop < 0.5 * atr)
        inside = bool(not good and np.isfinite(behind) and stop >= behind - 0.25 * atr and behind < entry)
        before = bool(np.isfinite(front) and entry < front < tp1)
    else:
        good = bool(np.isfinite(behind) and stop > behind and stop - behind < 0.5 * atr)
        inside = bool(not good and np.isfinite(behind) and stop <= behind + 0.25 * atr and behind > entry)
        before = bool(np.isfinite(front) and tp1 < front < entry)
    return {"stop_behind_level": good, "level_inside_stop": inside, "level_before_tp1": before}


def timing_advice(t: dict, side: int, entry: float, stop: float, tp1: float) -> list[str]:
    """Avisos de ejecucion para el lado que se va a operar (+1 largo, -1 corto)."""
    if not t.get("available") or side == 0:
        return []
    out = []
    atr = t["atr"]
    flag = timing_flags(t)[side]
    lv = level_flags(t, side, entry, stop, tp1)
    if side > 0:
        if flag["extended"]:
            out.append(f"Extended (RSI {t['rsi']:.0f}, %B {t['pct_b']:.2f}): price stretched above its 20-day "
                       f"band (SMA20 {t['sma20']:,.2f})." + evidence_note(1, "extended"))
        if flag["against_trend"]:
            out.append(f"Buying against a strong downtrend (ADX {t['adx']:.0f}, -DI {t['minus_di']:.0f} over "
                       f"+DI {t['plus_di']:.0f})." + evidence_note(1, "against_trend"))
        elif flag["with_trend"]:
            out.append("Long-term trend UP, on the side of the long." + evidence_note(1, "with_trend"))
        sup, res = t["support"], t["resistance"]
        if lv["stop_behind_level"]:
            out.append(f"Stop {stop:,.2f} sits just under support {sup:,.2f}." + evidence_note(1, "stop_behind_level"))
        elif lv["level_inside_stop"]:
            out.append(f"Support {sup:,.2f} is at or below the stop {stop:,.2f}: a test of support reaches the stop; "
                       f"a stop under it would be {sup - 0.25 * atr:,.2f}." + evidence_note(1, "level_inside_stop"))
        if lv["level_before_tp1"]:
            out.append(f"Resistance {res:,.2f} before TP1 {tp1:,.2f}." + evidence_note(1, "level_before_tp1"))
    else:
        if flag["extended"]:
            out.append(f"Oversold (RSI {t['rsi']:.0f}, %B {t['pct_b']:.2f}): price stretched below its 20-day "
                       f"band (SMA20 {t['sma20']:,.2f})." + evidence_note(-1, "extended"))
        if flag["against_trend"]:
            out.append(f"Shorting into a strong uptrend (ADX {t['adx']:.0f}, +DI {t['plus_di']:.0f} over "
                       f"-DI {t['minus_di']:.0f})." + evidence_note(-1, "against_trend"))
        elif flag["with_trend"]:
            out.append("Long-term trend DOWN, on the side of the short." + evidence_note(-1, "with_trend"))
        res, sup = t["resistance"], t["support"]
        if lv["stop_behind_level"]:
            out.append(f"Stop {stop:,.2f} sits just above resistance {res:,.2f}."
                       + evidence_note(-1, "stop_behind_level"))
        elif lv["level_inside_stop"]:
            out.append(f"Resistance {res:,.2f} is at or above the stop {stop:,.2f}: a test of resistance reaches the "
                       f"stop; a stop above it would be {res + 0.25 * atr:,.2f}." + evidence_note(-1, "level_inside_stop"))
        if lv["level_before_tp1"]:
            out.append(f"Support {sup:,.2f} before TP1 {tp1:,.2f}." + evidence_note(-1, "level_before_tp1"))
    return out
