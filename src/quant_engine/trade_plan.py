"""Plan de operacion: direccion corregida, entrada, stop, take profits y tamano.

**Correccion de la senal de research.** Se usa el Quant Score ALINEADO (score
x +1 si research dice LONG, x -1 si dice SHORT) y los umbrales del YAML:

    alineado >= weak                 CONFIRM      se opera como dijo research
    conflict < alineado < weak       REDUCE       se opera a `reduce_size` del tamano
    flip < alineado <= conflict      NO TRADE     el dato contradice; revisar la tesis
    alineado <= flip                 FLIP         el modelo sugiere el lado contrario

**Un FLIP sin evidencia va a medio tamano.** Ir contra research es la decision
mas fuerte del plan. Con evidencia de que el Quant Score predice en este
universo (t del IC de `signals.score_validation` >= `flip_min_ic_t`) va a
tamano completo. Sin ella, `unproven_flip` decide: "reduce" (por defecto) lo
opera a `reduce_size` con conviccion LOW; "no_trade" lo deja en NO TRADE.

**Senal clara por nombre.** Cada fila lleva `signal` (LONG, SHORT, NO TRADE),
si va con research o contra, y una conviccion HIGH / MEDIUM / LOW que combina
la fuerza del score, la evidencia de la validacion y la t del backtest de la
regla. Sin cartera long/short (una sola pata), cada nombre lleva ademas un
tamano propio por riesgo (`solo_*`): direccional, no un libro.

**Los numeros se comprueban antes de salir** (`check_plan`): precios
positivos, stop y objetivos del lado correcto, acciones = nocional / precio
convertido, riesgo por operacion bajo el tope, ordenes que cuadran con la
posicion y cobertura que cuadra con la beta. Un fallo sale en rojo.

**Niveles, por volatilidad y no por porcentajes fijos.** Un stop del 5% es
ruido en una accion de 60% de volatilidad y es enorme en una de 15%:

    stop        = entrada -/+ k x ATR(14)          (largo: debajo; corto: encima)
    R           = |entrada - stop|
    TP1, TP2    = entrada +/- m1 x R, m2 x R       (50% de la posicion en cada uno)
    tras TP1    : el stop pasa a la entrada (break-even)

**Tamano.** El peso de la cartera reconstruida con las direcciones corregidas,
recortado para que la perdida si salta el stop no supere `max_risk_per_trade`
del capital. Lo que manda es el menor de los dos.

**Monedas.** Capital, nocional, riesgo y P&L van en USD; entrada, stop y
objetivos, en la moneda en que cotiza el nombre (es donde se ponen las
ordenes). Las acciones salen de dividir el nocional en USD por el precio
convertido: dividir USD por un precio en CAD compraba un 30% de menos, y en
Londres (peniques) unas 76 veces menos.

**Cobertura de beta.** Los recortes por riesgo y los NO TRADE cambian el libro:
quitar un corto de beta alta lo deja largo de mercado sin que nadie lo decida.
Si la beta del libro final pasa de `hedge_threshold`, el plan anade una
operacion en el benchmark que la devuelve a cero.

**Ordenes contra la posicion actual.** Si se dan las posiciones que ya hay
(acciones con signo), el plan pasa de "que cartera quiero" a "que hago hoy":
BUY / SELL / SELL SHORT / BUY TO COVER / FLIP / HOLD. HOLD cuando se esta del
mismo lado y la diferencia no llega a `hold_band` del capital: ajustar un 0,2%
cuesta comision y spread y no cambia el riesgo.

**Probabilidad historica de TP antes que stop.** Para cada dia de los ultimos
anos se coloca la operacion con estos mismos multiplos de ATR y se mira que se
toca primero en `horizon_d` sesiones. Es una TASA BASE de la propia accion con
esos niveles, no un pronostico: incluye la deriva del pasado (NVDA subiendo)
y no sabe nada de lo que viene.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import multifactor
from .inference import statistics as stats_mod
from .sessions import session_date
from .portfolio import construction
from .portfolio.constraints import ConstructionParams
from .signals import score_validation

VERDICTS = ("CONFIRM", "REDUCE", "NO TRADE", "FLIP")


@dataclass(frozen=True)
class PlanParams:
    # Regla de los largos: elegida en la primera mitad de 6 anos de los 50 pares
    # entre 9 combinaciones y medida en la segunda: +0,084R frente a 2,5 ATR y
    # 1,5R / 3R, t 5,9, mejor en el 76% de los nombres (`tools/qe_rule_studies.py
    # long-rule`). En largo la deriva ayuda: dejar correr rinde mas. Es el borde
    # de la rejilla: mas ancho no se probo.
    stop_atr: float = 3.0
    tp_r: tuple[float, float] = (2.0, 4.0)
    max_risk_per_trade: float = 0.01
    reduce_size: float = 0.5
    weak: float = 0.15
    conflict: float = -0.15
    flip: float = -0.5
    horizon_d: int = 63
    history_d: int = 756
    hedge_beta: bool = True
    hedge_threshold: float = 0.05
    hold_band: float = 0.005
    flip_min_ic_t: float = 2.0
    unproven_flip: str = "reduce"
    short_time_stop: bool = True
    max_position: float = 0.35
    # Regla propia de los cortos: stop mas corto y objetivos mas cerca. Elegida
    # en la primera mitad de 6 anos de los 50 pares y medida en la segunda:
    # +0,082R frente a 2,5 ATR y 1,5R / 3R, t 7,8, mejor en el 84% de los
    # nombres (`tools/qe_rule_studies.py short-rule`). En corto el tiempo juega
    # en contra (la deriva del mercado): cobrar antes rinde mas.
    short_stop_atr: float | None = 2.0
    short_tp_r: tuple[float, float] | None = (1.0, 2.0)


def rule_text(p: PlanParams) -> str:
    """La regla de cada lado en una frase, para los reportes."""
    q = for_side(p, -1)
    return (f"longs: stop {p.stop_atr:g} x ATR(14), TP1 {p.tp_r[0]:g}R, TP2 {p.tp_r[1]:g}R; "
            f"shorts: stop {q.stop_atr:g} x ATR(14), TP1 {q.tp_r[0]:g}R, TP2 {q.tp_r[1]:g}R"
            + (", plus the time stop" if p.short_time_stop else ""))


def for_side(p: PlanParams, side: int) -> PlanParams:
    """Parametros de la regla del lado: los cortos llevan su stop y objetivos.

    Idempotente: aplicarlo dos veces da lo mismo. `levels`, `hit_rate` y
    `simulate_trades` lo aplican dentro, asi ningun camino usa la regla del
    otro lado; `holding_window` no sabe el lado y lo recibe ya aplicado.
    """
    if side < 0 and p.short_stop_atr is not None and p.short_tp_r is not None:
        return dataclasses.replace(p, stop_atr=float(p.short_stop_atr), tp_r=tuple(p.short_tp_r))
    return p


def verdict(aligned: float, p: PlanParams) -> str:
    if pd.isna(aligned):
        return "REDUCE"
    if aligned >= p.weak:
        return "CONFIRM"
    if aligned <= p.flip:
        return "FLIP"
    if aligned <= p.conflict:
        return "NO TRADE"
    return "REDUCE"


def final_signs(research: dict[str, int], aligned: pd.Series, p: PlanParams) -> tuple[pd.Series, pd.Series]:
    """Signo con el que se opera cada nombre (0 = no se opera) y su veredicto."""
    signs, verdicts = {}, {}
    for t, s in research.items():
        v = verdict(aligned.get(t, np.nan), p)
        verdicts[t] = v
        signs[t] = {"CONFIRM": s, "REDUCE": s, "NO TRADE": 0, "FLIP": -s}[v]
    return pd.Series(signs, dtype=float), pd.Series(verdicts)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def levels(entry: float, atr_value: float, side: int, p: PlanParams) -> dict[str, float]:
    """Stop y take profits. side = +1 largo, -1 corto."""
    p = for_side(p, side)
    r = p.stop_atr * atr_value
    return {
        "entry": entry,
        "stop": entry - side * r,
        "tp1": entry + side * p.tp_r[0] * r,
        "tp2": entry + side * p.tp_r[1] * r,
        "risk_per_share": r,
        "stop_pct": r / entry,
    }


def hit_rate(close: pd.Series, high: pd.Series, low: pd.Series, atr_series: pd.Series,
             side: int, p: PlanParams, tp_multiple: float) -> float:
    """Fraccion de entradas historicas que tocaron el TP antes que el stop.

    Con maximos y minimos diarios: si en un mismo dia se tocan los dos, cuenta
    como stop (supuesto conservador: no se sabe el orden dentro del dia).
    Las entradas que no tocan ninguno en el horizonte no cuentan como acierto.
    """
    p = for_side(p, side)
    c, h, l, a = (s.to_numpy(float) for s in (close, high, low, atr_series))
    n = len(c)
    start = max(0, n - p.history_d - p.horizon_d)
    wins = total = 0
    for i in range(start, n - p.horizon_d):
        if not np.isfinite(a[i]) or not np.isfinite(c[i]):
            continue
        r = p.stop_atr * a[i]
        stop = c[i] - side * r
        tp = c[i] + side * tp_multiple * r
        hh, ll = h[i + 1:i + 1 + p.horizon_d], l[i + 1:i + 1 + p.horizon_d]
        if side > 0:
            hit_stop, hit_tp = ll <= stop, hh >= tp
        else:
            hit_stop, hit_tp = hh >= stop, ll <= tp
        first_stop = np.argmax(hit_stop) if hit_stop.any() else math.inf
        first_tp = np.argmax(hit_tp) if hit_tp.any() else math.inf
        total += 1
        wins += first_tp < first_stop
    return wins / total if total else float("nan")


def size_position(weight: float, verdict_: str, entry: float, stop_pct: float,
                  capital: float, p: PlanParams, fx: float = 1.0, mult: float = 1.0) -> dict[str, float | bool]:
    """Nocional (USD) y acciones: el menor entre el peso de cartera y el tope de riesgo.

    `entry` en moneda local; `fx` = USD por unidad de esa moneda. `mult` recorta
    aparte (un FLIP sin evidencia va a `reduce_size`).
    """
    if verdict_ == "NO TRADE" or weight == 0 or entry <= 0 or not fx > 0:
        return {"notional": 0.0, "shares": 0, "size_capped_by_risk": False}
    w = abs(weight) * (p.reduce_size if verdict_ == "REDUCE" else 1.0) * mult
    notional = w * capital
    cap = p.max_risk_per_trade * capital / stop_pct if stop_pct > 0 else notional
    final = min(notional, cap)
    return {"notional": final, "shares": math.floor(final / (entry * fx)), "size_capped_by_risk": cap < notional}


def beta_hedge(weights: pd.Series, betas: pd.Series, price: float, capital: float,
               threshold: float) -> dict[str, object] | None:
    """Cobertura con el benchmark para devolver la beta del libro a cero.

    Los recortes por riesgo y los NO TRADE cambian el libro que se construyo:
    quitar un corto de beta alta deja al libro largo de mercado sin que nadie
    lo haya decidido. Por debajo de `threshold` no se cubre: operar para
    corregir 0,02 de beta cuesta mas de lo que protege.
    """
    b = betas.reindex(weights.index).fillna(1.0)
    beta_book = float((weights * b).sum())
    if abs(beta_book) < threshold or price <= 0:
        return None
    notional = beta_book * capital
    # Redondeo al entero mas cercano: en la cobertura importa acercarse a beta cero, no quedarse corto.
    shares = round(abs(notional) / price)
    return {"action": "SELL SHORT" if beta_book > 0 else "BUY", "shares": shares,
            "notional": shares * price, "price": price, "beta_before": beta_book,
            "beta_after": beta_book - math.copysign(shares * price / capital, beta_book)}


def drivers(components: pd.Series, weights: dict[str, float], research_sign: int,
            n: int = 2) -> tuple[list[str], list[str]]:
    """Componentes que mas empujan a favor y en contra de la direccion de research.

    Contribucion = peso x z-score x signo de research, renormalizada sobre los
    componentes disponibles, igual que el Quant Score. Es el "por que" de cada
    veredicto: un NO TRADE sin motivo visible no se puede discutir.
    """
    w = pd.Series(weights, dtype=float)
    comp = pd.to_numeric(components.reindex(w.index), errors="coerce")
    avail = w[comp.notna()]
    if avail.sum() <= 0:
        return [], []
    contrib = (comp[avail.index] * avail / avail.sum() * research_sign).sort_values()
    pro = [f"{k} {v:+.2f}" for k, v in contrib[contrib > 0].iloc[::-1].head(n).items()]
    con = [f"{k} {v:+.2f}" for k, v in contrib[contrib < 0].head(n).items()]
    return pro, con


def block_flips(signs: pd.Series, verdicts: pd.Series) -> tuple[pd.Series, pd.Series, list[str]]:
    """Cada FLIP pasa a NO TRADE: no se abre el lado contrario a research.

    Devuelve tambien los nombres afectados, para decirlo en el plan.
    """
    flipped = list(verdicts.index[verdicts == "FLIP"])
    signs, verdicts = signs.copy(), verdicts.copy()
    signs.loc[flipped] = 0.0
    verdicts.loc[flipped] = "NO TRADE"
    return signs, verdicts, flipped


def long_only_signs(signs: pd.Series, verdicts: pd.Series) -> tuple[pd.Series, pd.Series, list[str]]:
    """En una canasta solo LONG un FLIP no abre un corto: queda NO TRADE.

    Con un mandato solo LONG, "el dato contradice tanto la compra que el modelo
    venderia en corto" significa en la practica "no se compra". Abrir un corto
    convertiria la canasta en un long/short que nadie pidio.
    """
    return block_flips(signs, verdicts)


CONVICTION = ("LOW", "MEDIUM", "HIGH")


def conviction(verdict_: str, aligned: float, evidence: str, unproven: bool, mf_agree: int = 0) -> str:
    """Cuanto respaldo tiene la senal de un nombre que se opera.

    Fuerza del score alineado (|score| >= 0,5 fuerte, >= 0,15 moderada), con
    techo MEDIUM si la validacion no muestra que el score predice y LOW si
    ordena al reves o si es un FLIP sin evidencia.

    El backtest de la regla en el nombre NO entra: en los 50 pares, el E(R) de
    la primera mitad de las entradas no predijo el de la segunda dentro del
    mismo lado (rangos: cortos +0,10, t 0,7; largos -0,11). Lo que si persiste
    es el lado: los cortos perdieron de media -0,15R fuera de muestra, la deriva del
    mercado. Antes un t <= -2 bajaba un escalon, y eso castigaba al 40% de los
    cortos por la deriva, no por el nombre.
    """
    if verdict_ == "NO TRADE":
        return ""
    strength = abs(aligned) if pd.notna(aligned) else 0.0
    level = 2 if strength >= 0.5 else 1 if strength >= 0.15 else 0
    if verdict_ == "REDUCE":
        level = 0
    if evidence != "predictive":
        level = min(level, 1)
    if evidence == "inverse" or unproven:
        level = 0
    # El multifactor validado (IC con t >= 2 en su backtest) en su quintil
    # extremo: un escalon arriba si respalda la direccion, uno abajo si la
    # contradice. Va despues del techo por falta de evidencia del Quant Score:
    # es evidencia de otro modelo, y si la hay.
    level = min(max(level + int(mf_agree), 0), 2)
    return CONVICTION[level]


def solo_size(entry: float, risk_per_share: float, fx: float, capital: float, p: PlanParams,
              mult: float = 1.0) -> dict[str, float]:
    """Tamano de un nombre por si solo: que el stop cueste `max_risk_per_trade`.

    Sirve cuando no hay cartera long/short que reparta pesos. Nunca mas de
    `max_position` del capital en nocional: con un stop muy cerca, el tope de
    riesgo solo daria una posicion enorme.
    """
    unit_risk, unit = risk_per_share * fx, entry * fx
    if not (unit_risk > 0 and unit > 0):
        return {"shares": 0, "notional": 0.0, "risk_pct": 0.0}
    shares = math.floor(min(p.max_risk_per_trade * capital * mult / unit_risk,
                            p.max_position * capital * mult / unit))
    return {"shares": shares, "notional": shares * unit, "risk_pct": shares * unit_risk / capital}


def check_plan(plan: pd.DataFrame, orders: pd.DataFrame, p: PlanParams, capital: float,
               risk_at_stops: float, hedge: dict | None) -> list[str]:
    """Que los numeros del plan cuadren entre si. Devuelve los fallos (vacio = todo bien).

    No recalcula el plan con otra formula: comprueba identidades que tienen que
    cumplirse con cualquier dato. Un fallo aqui es un error de codigo o de
    datos, y el plan lo dice en rojo en vez de mandar ordenes con el.
    """
    errors: list[str] = []
    for t, r in plan.iterrows():
        prices = [r.get(k) for k in ("entry", "stop", "tp1", "tp2", "fx")]
        if not all(v is not None and np.isfinite(v) and v > 0 for v in prices):
            errors.append(f"{t}: invalid entry, stop, targets or exchange rate {prices}")
            continue
        side = 1 if r["stop"] < r["entry"] else -1
        expected = {"BUY": 1, "SELL SHORT": -1}.get(r["action"])
        if expected is not None and expected != side:
            errors.append(f"{t}: {r['action']} with the stop on the side of a {'short' if side < 0 else 'long'}")
        if not side * r["entry"] < side * r["tp1"] < side * r["tp2"]:
            errors.append(f"{t}: targets out of order ({r['tp1']:.2f}, {r['tp2']:.2f})")
        unit = r["entry"] * r["fx"]
        if r["shares"] < 0 or r["shares"] * unit > r["notional"] + 1e-6 or \
                (r["notional"] > 0 and r["notional"] - r["shares"] * unit >= unit + 1e-6):
            errors.append(f"{t}: {r['shares']} shares do not match the notional {r['notional']:,.0f} "
                          f"at {unit:,.4f} USD per share")
        risk = r["shares"] * abs(r["entry"] - r["stop"]) * r["fx"] / capital
        if abs(risk - r["risk_capital_pct"]) > 1e-9 or risk > p.max_risk_per_trade + 1e-9:
            errors.append(f"{t}: risk at the stop {risk:.4%} (cap {p.max_risk_per_trade:.2%})")
    for t, o in orders.iterrows():
        if o.get("verdict") == "NOT ANALYZED":
            continue
        moved = int(o["target"]) - int(o["current"])
        if o["signal"] == "HOLD":
            continue
        if o["signal"] != "NO TRADE" and int(o["qty"]) != abs(moved):
            errors.append(f"{t}: order of {int(o['qty'])} to go from {int(o['current'])} to {int(o['target'])}")
    total = float(orders.get("risk_capital_pct", pd.Series(dtype=float)).fillna(0.0).sum())
    if abs(total - risk_at_stops) > 1e-9:
        errors.append(f"total risk {risk_at_stops:.4%} is not the sum of the orders {total:.4%}")
    if hedge:
        moved = math.copysign(hedge["shares"] * hedge["price"] / capital, hedge["beta_before"])
        if abs(hedge["beta_before"] - moved - hedge["beta_after"]) > 1e-9:
            errors.append("the beta after the hedge does not match the benchmark shares")
    return errors


def flip_allowed(validation: dict | None, min_t: float) -> bool:
    """Si el plan puede ir contra research. Sin validacion configurada (`{}`),
    se respeta la decision de quien la apago; validada sin evidencia, no."""
    if min_t <= 0 or not validation:
        return True
    return score_validation.evidence(validation, min_t) == "predictive"


def target_shares(verdict_: str, research_sign: int, planned: int, current: int) -> int:
    """Posicion objetivo con signo. `planned` ya lleva la direccion corregida.

    NO TRADE con una posicion CONTRA research: el dato tampoco respalda a
    research, asi que no hay motivo para cerrarla. Se mantiene.
    """
    if verdict_ == "NO TRADE" and current * research_sign < 0:
        return current
    return planned


def planned_without_book(verdict_: str, research_sign: int, current: int, solo_shares: int = 0) -> int:
    """Objetivo cuando tras las correcciones solo queda una pata.

    Sin cartera long/short, el lado que el modelo apoya (CONFIRM o REDUCE en la
    direccion de research, FLIP en la contraria) se dimensiona por su cuenta
    (`solo_size`). Las ordenes llevan ese tamano: antes la senal final decia
    NO TRADE mientras la ficha del mismo nombre mandaba abrir 1.022 acciones.
    Sin tamano propio, una posicion del lado apoyado se mantiene y lo demas va
    a cero (antes todos los objetivos eran cero y se vendian hasta los CONFIRM).
    """
    supported = {"CONFIRM": research_sign, "REDUCE": research_sign, "FLIP": -research_sign}.get(verdict_)
    if supported is None:
        return 0
    if solo_shares > 0:
        return supported * int(solo_shares)
    return current if current * supported > 0 else 0


def order_for(current: int, target: int, price: float, capital: float, p: PlanParams) -> tuple[str, int]:
    """Orden para pasar de `current` a `target` acciones (con signo): (accion, cantidad)."""
    delta = target - current
    if current == 0 and target == 0:
        return "NO TRADE", 0
    if delta == 0:
        return "HOLD", 0
    if current != 0 and target != 0 and (current > 0) == (target > 0):
        if abs(delta) * price < p.hold_band * capital:
            return "HOLD", 0
        if current > 0:
            return ("BUY" if delta > 0 else "SELL"), abs(delta)
        return ("SELL SHORT" if delta < 0 else "BUY TO COVER"), abs(delta)
    if target == 0:
        return ("SELL" if current > 0 else "BUY TO COVER"), abs(current)
    if current == 0:
        return ("BUY" if target > 0 else "SELL SHORT"), abs(target)
    return ("FLIP TO LONG" if target > 0 else "FLIP TO SHORT"), abs(delta)


def time_range(ticker: str, side: int, levels_date: pd.Timestamp, horizon_d: int,
               time_stop_day: float | None) -> dict[str, object]:
    """Rango de tiempo de una posicion: apertura, stop por tiempo (cortos) y
    salida obligatoria, en sesiones de su bolsa (`sessions.session_date`). Las
    fichas usan la misma funcion: las fechas de la tabla y de la ficha cuadran.
    """
    if not side:
        return {"open_date": None, "time_stop_date": None, "exit_date": None}
    stop = (session_date(ticker, levels_date, int(time_stop_day))
            if side < 0 and time_stop_day is not None and pd.notna(time_stop_day) else None)
    return {"open_date": session_date(ticker, levels_date, 0), "time_stop_date": stop,
            "exit_date": session_date(ticker, levels_date, horizon_d)}


def _final(current: int, target: int, action: str) -> int:
    return current if action == "HOLD" else target


WRDS_COMPONENTS = ("value", "quality", "analyst", "short_interest")


def missing_wrds_weight(weights: dict[str, float]) -> float:
    """Fraccion del Quant Score que depende de WRDS (0 si todo es de precio)."""
    total = sum(float(v) for v in weights.values() if float(v) > 0)
    if total <= 0:
        return 0.0
    return sum(float(weights.get(k, 0.0)) for k in WRDS_COMPONENTS) / total


EXITS = ("stop", "tp1_be", "tp1_tp2", "tp1_time", "time")


def simulate_trades(close: pd.Series, high: pd.Series, low: pd.Series, atr_series: pd.Series,
                    side: int, p: PlanParams, cost: float = 0.0, borrow: float = 0.0,
                    time_stop: int | None = None) -> pd.DataFrame:
    """La regla exacta del plan, abierta cada dia de los ultimos `history_d`.

    Entrada al cierre; stop a `stop_atr` ATR; en TP1 sale la mitad y el stop
    de la otra mitad pasa a la entrada; esa mitad sale en TP2, en la entrada o
    al cierre de la sesion `horizon_d`. El resultado va en multiplos de R.
    Con `time_stop`, sin TP1 en esa sesion se cierra todo: la regla de los
    cortos.

    Supuestos conservadores, porque solo hay maximo, minimo y cierre diarios:
    si un dia toca stop y objetivo, gana el stop; el dia de TP1, si tambien
    toca la entrada, la otra mitad sale en la entrada; un gap que salta el stop
    se cobra al peor precio del dia en vez de al stop. Costes: `cost` por lado
    sobre el nocional (ida y vuelta) y, en cortos, `borrow` anual por dia.
    """
    p = for_side(p, side)
    c, h, l, a = (x.to_numpy(float) for x in (close, high, low, atr_series))
    n, horizon = len(c), p.horizon_d
    m1, m2 = p.tp_r
    rows, dates = [], []
    for i in range(max(0, n - p.history_d - horizon), n - horizon):
        entry, r = c[i], p.stop_atr * a[i]
        if not (np.isfinite(entry) and np.isfinite(r)) or r <= 0:
            continue
        stop, tp1, tp2 = entry - side * r, entry + side * m1 * r, entry + side * m2 * r
        half, realized, outcome, tp1_day = False, 0.0, None, np.nan
        for j in range(1, horizon + 1):
            hi, lo = h[i + j], l[i + j]
            if side > 0:
                hit_stop, hit_tp1, hit_be, hit_tp2 = lo <= stop, hi >= tp1, lo <= entry, hi >= tp2
                worst_stop, worst_be = min(stop, hi), min(entry, hi)
            else:
                hit_stop, hit_tp1, hit_be, hit_tp2 = hi >= stop, lo <= tp1, hi >= entry, lo <= tp2
                worst_stop, worst_be = max(stop, lo), max(entry, lo)
            if not half:
                if hit_stop:
                    outcome = (side * (worst_stop - entry) / r, "stop", j)
                    break
                if hit_tp1:
                    half, realized, tp1_day = True, 0.5 * m1, j
                    if hit_be:
                        outcome = (realized, "tp1_be", j)
                        break
                    continue
                # Stop por tiempo: sin TP1 a esta altura, se cierra al cierre del
                # dia. Es la regla en los cortos (`rule_time_stop`); en los largos
                # solo se usa para comparar.
                if time_stop is not None and j >= time_stop:
                    outcome = (side * (c[i + j] - entry) / r, "time_stop", j)
                    break
                continue
            if hit_be:
                outcome = (realized + 0.5 * side * (worst_be - entry) / r, "tp1_be", j)
                break
            if hit_tp2:
                outcome = (realized + 0.5 * m2, "tp1_tp2", j)
                break
        if outcome is None:
            last = side * (c[i + horizon] - entry) / r
            outcome = (realized + 0.5 * last, "tp1_time", horizon) if half else (last, "time", horizon)
        r_mult, exit_type, days = outcome
        r_mult -= 2 * cost * entry / r
        if side < 0:
            r_mult -= borrow * days / 252 * entry / r
        rows.append((r_mult, exit_type, days, tp1_day))
        dates.append(close.index[i])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=["r_mult", "exit", "days", "tp1_day"])


def time_stop_out_of_sample(trades: pd.DataFrame, timed: pd.DataFrame, p: PlanParams,
                            min_trades: int = 200) -> dict[str, float] | None:
    """El stop por tiempo probado fuera de muestra.

    El dia de revision se elige con la PRIMERA mitad de las entradas
    historicas y se mide en la SEGUNDA: E(R) sin el stop frente a E(R) con el.
    `timed` es `simulate_trades(..., time_stop=dia elegido en la primera mitad)`
    sobre las mismas fechas. Dentro de muestra el stop siempre parece ayudar
    (el dia sale de esas operaciones); esta es la prueba que cuenta.
    """
    if trades is None or len(trades) < min_trades:
        return None
    split = trades.index[len(trades) // 2]
    first = trades[trades.index < split]
    hw = holding_window(first, p)
    if not hw.get("review_day"):
        return None
    second = trades[trades.index >= split]
    later = timed[timed.index >= split]
    return {"review_day": hw["review_day"], "split": split, "first": first.index.min(),
            "last": trades.index.max(), "n": len(second), "cut_in_sample": bool(hw.get("cut_at_review")),
            "exp_r_plain": float(second["r_mult"].mean()), "exp_r_time_stop": float(later["r_mult"].mean()),
            "improvement": float(later["r_mult"].mean() - second["r_mult"].mean())}


def rule_time_stop(side: int, window: dict, p: PlanParams) -> int | None:
    """Dia del stop por tiempo si la regla lo aplica: solo en cortos.

    Fuera de muestra en los 50 pares (dia de revision de la primera mitad de
    cada historia, medido en la segunda): en cortos mejoro la E(R) en el 68% de
    los nombres, +0,04R de media con t 4,3 (con su regla propia; con la de los
    largos era 80%, +0,05R, t 3,7); en largos la empeoro (38%, -0,04R,
    t -2,5). Un corto que no avanza pelea contra la deriva al alza y paga
    prestamo; a un largo cortarlo le quita el recorrido de los que suben.
    """
    if side < 0 and p.short_time_stop and window.get("review_day"):
        return int(window["review_day"])
    return None


def holding_window(trades: pd.DataFrame, p: PlanParams) -> dict[str, float]:
    """Cuanto dura la operacion con esta regla en este valor, y cuando revisarla.

    Sale de las mismas entradas historicas del backtest de la regla:

    - Ventana tipica: percentiles 25-75 de los dias hasta la salida.
    - TP1: en que dia suele tocarse (mediana y percentil 75).
    - Revision (stop por tiempo): el percentil 75 del dia del TP1. De las
      operaciones que a esa altura seguian vivas sin TP1 (ni stop), que
      fraccion lo toco despues. Si es menor que el break-even del TP1
      (1 / (1 + TP1 en R)), seguir esperando tiene expectativa negativa: se
      cierra en esa fecha en vez de agotar el horizonte. Mantener un corto
      cuesta prestamo cada dia, y el tiempo sin avance es informacion.
    """
    if trades is None or trades.empty:
        return {"n": 0}
    days = trades["days"].astype(float)
    out: dict[str, float] = {"n": len(trades), "p25": float(days.quantile(0.25)),
                             "median": float(days.median()), "p75": float(days.quantile(0.75)),
                             "max": float(p.horizon_d)}
    tp1 = trades["tp1_day"].dropna() if "tp1_day" in trades else pd.Series(dtype=float)
    out["tp1_share"] = float(len(tp1) / len(trades))
    if len(tp1) >= 10:
        out["tp1_median"] = float(tp1.median())
        k = int(np.ceil(tp1.quantile(0.75)))
        stopped_by_k = (trades["exit"] == "stop") & (trades["days"] <= k)
        tp1_by_k = trades["tp1_day"] <= k
        alive = trades[~stopped_by_k & ~tp1_by_k.fillna(False)]
        out["review_day"] = float(k)
        out["alive_at_review"] = float(len(alive) / len(trades))
        out["late_tp1_rate"] = float(alive["tp1_day"].notna().mean()) if len(alive) else float("nan")
        out["breakeven_tp1"] = 1.0 / (1.0 + p.tp_r[0])
        out["cut_at_review"] = bool(len(alive) >= 10 and out["late_tp1_rate"] < out["breakeven_tp1"])
    return out


def hac_t_stat(x: np.ndarray, lag: int) -> float:
    """t de la media con error estandar de Newey-West (nucleo de Bartlett).

    Las operaciones de la regla se abren cada dia y duran hasta `horizon_d`
    sesiones: dos entradas seguidas comparten casi todo el camino. Con el error
    estandar ingenuo, 756 entradas parecerian 756 observaciones independientes
    y cualquier R medio saldria "significativo"; en realidad hay unas pocas
    decenas de operaciones que no se solapan. El error estandar es el mismo del
    modelo multifactor (`sfc_tfsig.validation.newey_west_se`).
    """
    from sfc_tfsig.validation import newey_west_se  # noqa: PLC0415

    values = pd.Series(np.asarray(x, dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
    se = newey_west_se(values, lag)
    return float(values.mean() / se) if se and np.isfinite(se) and se > 0 else float("nan")


def trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    """Expectativa de la regla: aciertos, R medio y su t, profit factor, dias."""
    if trades.empty:
        return {"n": 0}
    r = trades["r_mult"]
    wins, losses = r[r > 0], r[r <= 0]
    loss_sum = -losses.sum()
    return {
        "n": len(r),
        "win_rate": float((r > 0).mean()),
        "exp_r": float(r.mean()),
        # Retardo = la operacion mas larga: mas alla no hay solape.
        "t_stat": hac_t_stat(r.to_numpy(), int(trades["days"].max())),
        "avg_win_r": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss_r": float(losses.mean()) if len(losses) else float("nan"),
        "profit_factor": float(wins.sum() / loss_sum) if loss_sum > 0 else float("inf"),
        "avg_days": float(trades["days"].mean()),
        "p05_r": float(r.quantile(0.05)),
        "p95_r": float(r.quantile(0.95)),
        **{f"share_{e}": float((trades["exit"] == e).mean()) for e in EXITS},
    }


def plan_backtest(trades: dict[str, pd.DataFrame], dollars_per_r: dict[str, float],
                  capital: float) -> dict[str, float] | None:
    """El plan completo, abierto a la vez en cada fecha historica comun.

    Usar las MISMAS fechas para todos los nombres conserva la correlacion entre
    ellos: si el plan se hubiera puesto el dia de una caida, todos los largos lo
    sufren juntos. Sumar distribuciones independientes subestimaria la cola.
    """
    live = {t: df["r_mult"] for t, df in trades.items() if dollars_per_r.get(t) and not df.empty}
    if not live:
        return None
    joint = pd.DataFrame(live).dropna()
    if joint.empty:
        return None
    pnl = sum(joint[t] * dollars_per_r[t] for t in joint.columns)
    return {
        "n_dates": len(pnl), "first": joint.index.min(), "last": joint.index.max(),
        "mean": float(pnl.mean()), "median": float(pnl.median()),
        "p05": float(pnl.quantile(0.05)), "p95": float(pnl.quantile(0.95)),
        "worst": float(pnl.min()), "best": float(pnl.max()),
        "prob_loss": float((pnl < 0).mean()),
        "mean_pct": float(pnl.mean() / capital), "p05_pct": float(pnl.quantile(0.05) / capital),
        "p95_pct": float(pnl.quantile(0.95) / capital),
        "pnl": pnl,
    }


def portfolio_expectations(result) -> dict[str, float]:
    """La canasta como cartera (direcciones de research) y su Monte Carlo."""
    out: dict[str, float] = {}
    primary = result.primary or {}
    sm = primary.get("summary") or {}
    bt = primary.get("backtest")
    out.update({k: sm.get(k) for k in ("cagr", "sharpe", "max_drawdown", "volatility", "years")})
    out["method"] = primary.get("method")
    if bt is not None and len(bt.nav) > 1:
        bench = result.clean.returns[result.clean.benchmark].reindex(bt.nav.index).fillna(0.0)
        growth = float((1.0 + bench.iloc[1:]).prod())
        years = len(bench) / 252
        out["bench_cagr"] = growth ** (1 / years) - 1 if years > 0 else float("nan")
    mc = (result.montecarlo or {}).get("bootstrap_neutral") or {}
    out.update({f"mc_{k}": v for k, v in mc.items()})
    return out


def build_plan(result, p: PlanParams) -> dict[str, object]:
    """Plan completo a partir del resultado del analisis."""
    s = result.settings
    research = {**{t: 1 for t in result.longs}, **{t: -1 for t in result.shorts}}
    aligned = result.agreement["aligned_score"]
    signs, verdicts = final_signs(research, aligned, p)
    score_verdicts = verdicts.copy()
    long_only = not result.shorts
    flipped: list[str] = []
    if long_only:
        signs, verdicts, flipped = long_only_signs(signs, verdicts)
    validation = getattr(result, "score_validation", None) or {}
    evidence = score_validation.evidence(validation, p.flip_min_ic_t or 2.0) if validation else "unknown"
    unproven: list[str] = []
    mf = getattr(result, "multifactor", None) or {}
    mf_ok = multifactor.usable(mf)
    mf_rows = mf.get("rows", {}) if mf_ok else {}
    mf_backed: list[str] = []
    if not flip_allowed(validation, p.flip_min_ic_t):
        flips = list(verdicts.index[verdicts == "FLIP"])
        # Un FLIP que el multifactor validado respalda en su quintil extremo no
        # es "sin evidencia": va a tamano completo. Solo hacia LARGO: con la
        # regla del plan, los cortos en Q5 no ganaron a los neutrales
        # (`multifactor.conviction_step`).
        mf_backed = [t for t in flips if multifactor.conviction_step(int(np.sign(signs[t])),
                                                                      mf_rows.get(t, {}).get("percentile")) > 0]
        weak = [t for t in flips if t not in mf_backed]
        if p.unproven_flip == "no_trade" and weak:
            blocked_signs, blocked_verdicts, unproven = block_flips(signs[weak], verdicts[weak])
            signs.loc[weak], verdicts.loc[weak] = blocked_signs, blocked_verdicts
        else:
            unproven = weak

    traded = signs[signs != 0]
    method = result.primary["method"]
    weights = pd.Series(0.0, index=signs.index)
    note = ""
    cov = result.primary["cov"]
    # Solo LONG: la cartera es la pata larga. Long/short: hacen falta las dos.
    has_book = bool((traded > 0).any() and (long_only or (traded < 0).any()))
    gross, net = s.exposure
    if has_book:
        vol = pd.Series(np.sqrt(np.diag(cov.loc[traded.index, traded.index]) * 252), index=traded.index)
        leg, fewest = ((gross, (traded > 0).sum()) if long_only
                       else (gross / 2, min((traded > 0).sum(), (traded < 0).sum())))
        params = ConstructionParams(
            gross=gross, net=net,
            max_position=max(float(s.get("portfolio.max_position", 0.35)), leg / max(1, fewest)))
        built = construction.build(method, traded, cov=cov, vol=vol, params=params,
                                   mu=pd.Series(0.0, index=traded.index), betas=result.primary.get("betas"))
        if built.status == "ok":
            weights.loc[built.weights.index] = built.weights
        else:
            note = f"{method} failed with the corrected directions ({built.note}); equal_weight is used instead"
            weights.loc[traded.index] = construction.equal_weight(traded, params)
    elif long_only:
        note = ("after the corrections no LONG is left to trade: each name the data supports is sized on its "
                "own, with its stop costing the per-trade risk cap.")
    else:
        note = ("after the corrections there are not both longs and shorts left: no long/short book. Each name "
                "the data supports is sized on its own, with its stop costing the per-trade risk cap.")
    if flipped:
        flip_note = f"long-only: {', '.join(flipped)} scored as FLIP and stays NO TRADE, no short is opened"
        note = "; ".join(n for n in (note, flip_note) if n)
    if unproven:
        got = (f"IC t {validation['ic_t']:+.1f}" if validation.get("available")
               else f"no validation: {validation.get('reason')}")
        outcome = ("stays NO TRADE" if p.unproven_flip == "no_trade"
                   else f"is traded at {p.reduce_size:.0%} size with LOW conviction")
        gate_note = (f"{', '.join(unproven)} scored as FLIP and {outcome}: the Quant Score has not shown it "
                     f"predicts on this universe ({got}; full size needs t >= {p.flip_min_ic_t:.1f})")
        note = "; ".join(n for n in (note, gate_note) if n)

    capital = s.initial_capital
    rows = []
    sims: dict[str, pd.DataFrame] = {}
    for t in signs.index:
        # Niveles en la moneda en que cotiza: es donde se ponen las ordenes.
        close_l, high_l, low_l = result.clean.local_prices(t)
        close = close_l.dropna()
        high = high_l.reindex(close.index).fillna(close)
        low = low_l.reindex(close.index).fillna(close)
        a = atr(high, low, close)
        side = int(signs[t]) if signs[t] != 0 else research[t]
        q = for_side(p, side)
        lv = levels(float(close.iloc[-1]), float(a.iloc[-1]), side, q)
        fx = result.clean.fx_rate(t).reindex(close.index).ffill()
        fx_now = float(fx.iloc[-1]) if len(fx) and pd.notna(fx.iloc[-1]) else 1.0

        is_unproven = t in unproven and verdicts[t] == "FLIP"
        mult = p.reduce_size if is_unproven else 1.0
        size = size_position(float(weights.get(t, 0.0)), verdicts[t], lv["entry"], lv["stop_pct"], capital, p,
                             fx=fx_now, mult=mult)
        solo = (solo_size(lv["entry"], lv["risk_per_share"], fx_now, capital, p,
                          mult * (p.reduce_size if verdicts[t] == "REDUCE" else 1.0))
                if signs[t] != 0 else {"shares": 0, "notional": 0.0, "risk_pct": 0.0})

        rows.append({
            "ticker": t,
            "research": "LONG" if research[t] > 0 else "SHORT",
            "verdict": verdicts[t],
            "score_verdict": score_verdicts[t],
            "action": ("BUY" if signs[t] > 0 else "SELL SHORT") if signs[t] != 0 else "NO TRADE",
            "aligned_score": float(aligned.get(t, np.nan)),
            **lv,
            "atr": float(a.iloc[-1]),
            "currency": result.clean.currency.get(t, "USD"),
            "fx": fx_now,
            "weight": math.copysign(size["notional"] / capital, side) if signs[t] != 0 else 0.0,
            "notional": size["notional"],
            "shares": size["shares"],
            "risk_capital_pct": size["shares"] * lv["risk_per_share"] * fx_now / capital,
            "size_capped_by_risk": size["size_capped_by_risk"],
            "signal": ("LONG" if signs[t] > 0 else "SHORT") if signs[t] != 0 else "NO TRADE",
            "against_research": bool(signs[t] != 0 and np.sign(signs[t]) != research[t]),
            "unproven_flip": is_unproven,
            "solo_shares": solo["shares"], "solo_notional": solo["notional"], "solo_risk_pct": solo["risk_pct"],
            "p_tp1_before_stop": hit_rate(close, high, low, a, side, q, q.tp_r[0]),
            "p_tp2_before_stop": hit_rate(close, high, low, a, side, q, q.tp_r[1]),
        })
        sims[t] = simulate_trades(close, high, low, a, side, q, cost=s.transaction_cost,
                                  borrow=s.borrow_cost)
        st = trade_stats(sims[t])
        hw = holding_window(sims[t], q)
        if hw.get("cut_at_review"):
            # Lo que habria dado la regla cerrando en la revision (dentro de
            # muestra: el dia de revision sale de estas mismas operaciones).
            ts = trade_stats(simulate_trades(close, high, low, a, side, q, cost=s.transaction_cost,
                                             borrow=s.borrow_cost, time_stop=int(hw["review_day"])))
            hw["exp_r_time_stop"] = ts.get("exp_r")
        # Prueba fuera de muestra: dia de revision de la primera mitad, medido en la segunda.
        half = sims[t].iloc[: len(sims[t]) // 2]
        hw_first = holding_window(half, q)
        oos = None
        if hw_first.get("review_day"):
            timed = simulate_trades(close, high, low, a, side, q, cost=s.transaction_cost, borrow=s.borrow_cost,
                                    time_stop=int(hw_first["review_day"]))
            oos = time_stop_out_of_sample(sims[t], timed, q)
        # La regla de los cortos incluye el stop por tiempo: el backtest de la
        # regla, la ventana y el backtest del plan miden ESA regla, no la de
        # sin stop (que queda para comparar).
        plain_exp_r = st.get("exp_r")
        stop_day = rule_time_stop(side, hw, q)
        hold = hw
        if stop_day:
            sims[t] = simulate_trades(close, high, low, a, side, q, cost=s.transaction_cost, borrow=s.borrow_cost,
                                      time_stop=stop_day)
            st = trade_stats(sims[t])
            hold = holding_window(sims[t], q)
        # Intervalo del E(R): bootstrap por bloques del horizonte, porque las
        # entradas diarias se solapan hasta `horizon_d` sesiones.
        ci = stats_mod.block_bootstrap_mean(sims[t]["r_mult"], block=p.horizon_d) if len(sims[t]) else {}
        rows[-1].update({
            "bt_exp_r_lo": ci.get("lo"), "bt_exp_r_hi": ci.get("hi"), "bt_p_positive": ci.get("p_positive"),
            "time_stop_day": stop_day, "bt_exp_r_no_time_stop": plain_exp_r,
            "bt_trades": st.get("n", 0), "bt_win_rate": st.get("win_rate"), "bt_exp_r": st.get("exp_r"),
            "bt_t_stat": st.get("t_stat"),
            "bt_avg_win_r": st.get("avg_win_r"), "bt_avg_loss_r": st.get("avg_loss_r"),
            "bt_profit_factor": st.get("profit_factor"), "bt_avg_days": st.get("avg_days"),
            "hold_p25": hold.get("p25"), "hold_median": hold.get("median"), "hold_p75": hold.get("p75"),
            "tp1_median_days": hw.get("tp1_median"), "review_day": hw.get("review_day"),
            "late_tp1_rate": hw.get("late_tp1_rate"), "cut_at_review": hw.get("cut_at_review", False),
            "bt_exp_r_time_stop": hw.get("exp_r_time_stop"),
            "oos_ts_review_day": (oos or {}).get("review_day"), "oos_ts_split": (oos or {}).get("split"),
            "oos_ts_exp_r_plain": (oos or {}).get("exp_r_plain"),
            "oos_ts_exp_r_time_stop": (oos or {}).get("exp_r_time_stop"),
            "oos_ts_improvement": (oos or {}).get("improvement"),
            "exp_return_pct": (st.get("exp_r") or 0.0) * lv["stop_pct"] if st.get("n") else float("nan"),
        })
    plan = pd.DataFrame(rows).set_index("ticker")
    plan["mf_percentile"] = [mf_rows.get(t, {}).get("percentile", np.nan) for t in plan.index]
    plan["mf_agreement"] = [multifactor.agreement({"LONG": 1, "SHORT": -1}.get(plan.loc[t, "signal"], 0),
                                                  plan.loc[t, "mf_percentile"]) for t in plan.index]
    plan["mf_step"] = [multifactor.conviction_step({"LONG": 1, "SHORT": -1}.get(plan.loc[t, "signal"], 0),
                                                   plan.loc[t, "mf_percentile"]) for t in plan.index]
    plan["flip_backed_by_multifactor"] = [t in mf_backed for t in plan.index]
    plan["conviction"] = [conviction(plan.loc[t, "verdict"], plan.loc[t, "aligned_score"], evidence,
                                     bool(plan.loc[t, "unproven_flip"]), int(plan.loc[t, "mf_step"]))
                          for t in plan.index]
    try:
        from .signals.composite import normalized_weights  # noqa: PLC0415

        sw = normalized_weights(s.get("signal_weights", {}) or {})
    except ValueError:
        sw = {}
    for t in plan.index:
        pro, con = drivers(result.scores.loc[t], sw, research[t]) if t in result.scores.index and sw else ([], [])
        plan.loc[t, "drivers_for"] = ", ".join(pro)
        plan.loc[t, "drivers_against"] = ", ".join(con)
    corrections = plan[plan["verdict"] != "CONFIRM"]

    positions = dict(getattr(s, "positions", ()) or ())
    orders = []
    final = {}
    for t, row in plan.iterrows():
        current = int(positions.get(t, 0))
        solo = int(row.get("solo_shares", 0) or 0)
        if has_book:
            planned = -int(row["shares"]) if row["action"] == "SELL SHORT" else int(row["shares"])
        else:
            planned = planned_without_book(row["verdict"], research[t], current, solo if row["action"] != "NO TRADE"
                                           else 0)
        target = target_shares(row["verdict"], research[t], planned, current)
        action, qty = order_for(current, target, row["entry"] * row["fx"], capital, p)
        final[t] = _final(current, target, action)
        # Niveles del lado de la posicion que QUEDA: una posicion mantenida contra
        # research es larga aunque research dijera SHORT. Sin posicion, sin niveles.
        side = int(np.sign(final[t]))
        lv = levels(row["entry"], row["atr"], side, p) if side else {}
        # El stop por tiempo es de la regla del lado del plan; una posicion que
        # queda del otro lado no lo lleva.
        plan_side = {"BUY": 1, "SELL SHORT": -1}.get(row["action"], 0)
        when = time_range(t, side, result.clean.close.index[-1], p.horizon_d,
                          row.get("time_stop_day") if side == plan_side else None)
        orders.append({"ticker": t, "signal": action, "qty": qty, "current": current, "target": final[t], **when,
                       "price": row["entry"], "currency": row["currency"], "stop": lv.get("stop"),
                       "tp1": lv.get("tp1"), "tp2": lv.get("tp2"), "verdict": row["verdict"],
                       "risk_capital_pct": (abs(final[t]) * abs(row["entry"] - lv["stop"]) * row["fx"] / capital
                                            if side else 0.0),
                       "sized_by": "book" if has_book else ("standalone" if final[t] and planned == target else ""),
                       "hold_reason": ("no long/short book: kept as is" if not has_book and action == "HOLD"
                                       and not (solo and planned == target)
                                       else "kept against research" if target != planned
                                       else "within hold band" if action == "HOLD" else "")})

    betas = result.primary.get("betas", pd.Series(1.0, index=plan.index))
    betas = betas.reindex(plan.index).fillna(1.0)
    final_w = pd.Series(final, dtype=float) * plan["entry"] * plan["fx"] / capital
    # Dolares por R de cada posicion FINAL: lo que se gana o pierde por cada
    # multiplo del riesgo por accion de hoy.
    dollars_per_r = {t: abs(final[t]) * float(plan.loc[t, "risk_per_share"] * plan.loc[t, "fx"])
                     for t in plan.index}
    plan["exp_pnl"] = [dollars_per_r[t] * (plan.loc[t, "bt_exp_r"] if pd.notna(plan.loc[t, "bt_exp_r"]) else 0.0)
                       for t in plan.index]
    plan_bt = plan_backtest(sims, dollars_per_r, capital)
    book_beta = float((final_w * betas).sum())

    bench = result.clean.benchmark
    hedge = None
    # Solo LONG: la beta de mercado ES la exposicion buscada. Cubrirla
    # convertiria la canasta en un libro neutral al mercado que nadie pidio.
    if p.hedge_beta and not long_only:
        bench_price = float(result.clean.close[bench].dropna().iloc[-1])
        hedge = beta_hedge(final_w, betas, bench_price, capital, p.hedge_threshold)
        if hedge:
            hedge["ticker"] = bench
        current = int(positions.get(bench, 0))
        target = 0 if hedge is None else (-hedge["shares"] if hedge["action"] == "SELL SHORT" else hedge["shares"])
        action, qty = order_for(current, target, bench_price, capital, p)
        if action != "NO TRADE":
            orders.append({"ticker": bench, "signal": action, "qty": qty, "current": current,
                           "target": _final(current, target, action), "price": bench_price,
                           "currency": result.clean.currency.get(bench, "USD"), "verdict": "HEDGE"})

    handled = set(plan.index) | ({bench} if p.hedge_beta and not long_only else set())
    for t, current in positions.items():
        if t not in handled:
            orders.append({"ticker": t, "signal": "NOT ANALYZED", "qty": 0, "current": int(current),
                           "target": int(current), "verdict": "NOT ANALYZED"})

    orders_df = pd.DataFrame(orders).set_index("ticker")
    for col in ("open_date", "time_stop_date", "exit_date"):
        plan[col] = orders_df[col].reindex(plan.index) if col in orders_df else None
    risk_at_stops = float(pd.DataFrame(orders)["risk_capital_pct"].fillna(0.0).sum())
    return {"plan": plan, "corrections": corrections, "note": note, "params": p, "hedge": hedge,
            "orders": orders_df, "has_positions": bool(positions),
            "long_only": long_only, "has_book": has_book, "unproven_flips": unproven, "evidence": evidence,
            "multifactor": mf, "multifactor_backed_flips": mf_backed,
            "beta": book_beta, "gross": float(final_w.abs().sum()), "net": float(final_w.sum()),
            "risk_at_stops": risk_at_stops,
            "integrity_errors": check_plan(plan, orders_df, p, capital, risk_at_stops, hedge),
            "levels_date": result.clean.close.index[-1],
            "plan_backtest": plan_bt, "portfolio": portfolio_expectations(result),
            "expected_pnl": float(plan["exp_pnl"].sum()),
            "wrds_missing": (None if (result.wrds is not None and result.wrds.available)
                             else {"weight": missing_wrds_weight(s.get("signal_weights", {}) or {}),
                                   "reason": result.wrds.reason if result.wrds is not None else "not requested"})}


def params_from_settings(settings) -> PlanParams:
    tp = settings.get("trade_plan", {}) or {}
    ag = settings.get("agreement", {}) or {}
    return PlanParams(
        stop_atr=float(tp.get("stop_atr", 3.0)),
        tp_r=tuple(float(x) for x in tp.get("take_profit_r", [2.0, 4.0]))[:2],
        short_stop_atr=float(tp.get("short_stop_atr", 2.0)),
        short_tp_r=tuple(float(x) for x in tp.get("short_take_profit_r", [1.0, 2.0]))[:2],
        max_risk_per_trade=float(tp.get("max_risk_per_trade", 0.01)),
        reduce_size=float(tp.get("reduce_size", 0.5)),
        weak=float(ag.get("weak", 0.15)),
        conflict=float(ag.get("conflict", -0.15)),
        flip=float(tp.get("flip_threshold", -0.5)),
        horizon_d=int(tp.get("horizon_d", 63)),
        hedge_beta=bool(tp.get("hedge_beta", True)),
        hedge_threshold=float(tp.get("hedge_threshold", 0.05)),
        hold_band=float(tp.get("hold_band", 0.005)),
        flip_min_ic_t=float(tp.get("flip_min_ic_t", 2.0)),
        unproven_flip=str(tp.get("unproven_flip", "reduce")),
        short_time_stop=bool(tp.get("short_time_stop", True)),
        max_position=float(settings.get("portfolio.max_position", 0.35)),
    )
