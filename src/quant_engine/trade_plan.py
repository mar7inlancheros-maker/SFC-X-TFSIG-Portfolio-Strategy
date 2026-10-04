"""Plan de operacion: direccion corregida, entrada, stop, take profits y tamano.

**Correccion de la senal de research.** Se usa el Quant Score ALINEADO (score
x +1 si research dice LONG, x -1 si dice SHORT) y los umbrales del YAML:

    alineado >= weak                 CONFIRM      se opera como dijo research
    conflict < alineado < weak       REDUCE       se opera a `reduce_size` del tamano
    flip < alineado <= conflict      NO TRADE     el dato contradice; revisar la tesis
    alineado <= flip                 FLIP         el modelo sugiere el lado contrario

**Niveles, por volatilidad y no por porcentajes fijos.** Un stop del 5% es
ruido en una accion de 60% de volatilidad y es enorme en una de 15%:

    stop        = entrada -/+ k x ATR(14)          (largo: debajo; corto: encima)
    R           = |entrada - stop|
    TP1, TP2    = entrada +/- m1 x R, m2 x R       (50% de la posicion en cada uno)
    tras TP1    : el stop pasa a la entrada (break-even)

**Tamano.** El peso de la cartera reconstruida con las direcciones corregidas,
recortado para que la perdida si salta el stop no supere `max_risk_per_trade`
del capital. Lo que manda es el menor de los dos.

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

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import construction
from .portfolio.constraints import ConstructionParams

VERDICTS = ("CONFIRM", "REDUCE", "NO TRADE", "FLIP")


@dataclass(frozen=True)
class PlanParams:
    stop_atr: float = 2.5
    tp_r: tuple[float, float] = (1.5, 3.0)
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
                  capital: float, p: PlanParams) -> dict[str, float | bool]:
    """Nocional y acciones: el menor entre el peso de cartera y el tope de riesgo."""
    if verdict_ == "NO TRADE" or weight == 0 or entry <= 0:
        return {"notional": 0.0, "shares": 0, "size_capped_by_risk": False}
    w = abs(weight) * (p.reduce_size if verdict_ == "REDUCE" else 1.0)
    notional = w * capital
    cap = p.max_risk_per_trade * capital / stop_pct if stop_pct > 0 else notional
    final = min(notional, cap)
    return {"notional": final, "shares": math.floor(final / entry), "size_capped_by_risk": cap < notional}


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


def target_shares(verdict_: str, research_sign: int, planned: int, current: int) -> int:
    """Posicion objetivo con signo. `planned` ya lleva la direccion corregida.

    NO TRADE con una posicion CONTRA research: el dato tampoco respalda a
    research, asi que no hay motivo para cerrarla. Se mantiene.
    """
    if verdict_ == "NO TRADE" and current * research_sign < 0:
        return current
    return planned


def planned_without_book(verdict_: str, research_sign: int, current: int) -> int:
    """Objetivo cuando tras las correcciones solo queda una pata.

    Sin cartera long/short no hay tamano: lo que no se puede dimensionar no se
    abre, pero tampoco se liquida. Una posicion del lado que el modelo sigue
    apoyando (CONFIRM o REDUCE en la direccion de research, FLIP en la
    contraria) se mantiene; lo demas va a cero. Antes, sin cartera todos los
    objetivos eran cero y el plan mandaba vender hasta los CONFIRM.
    """
    supported = {"CONFIRM": research_sign, "REDUCE": research_sign, "FLIP": -research_sign}.get(verdict_)
    if supported is not None and current * supported > 0:
        return current
    return 0


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


def _final(current: int, target: int, action: str) -> int:
    return current if action == "HOLD" else target


WRDS_COMPONENTS = ("value", "quality", "analyst", "short_interest")


def missing_wrds_weight(weights: dict[str, float]) -> float:
    """Fraccion del Quant Score que depende de WRDS (0 si todo es de precio)."""
    total = sum(float(v) for v in weights.values() if float(v) > 0)
    if total <= 0:
        return 0.0
    return sum(float(weights.get(k, 0.0)) for k in WRDS_COMPONENTS) / total


def build_plan(result, p: PlanParams) -> dict[str, object]:
    """Plan completo a partir del resultado del analisis."""
    s = result.settings
    research = {**{t: 1 for t in result.longs}, **{t: -1 for t in result.shorts}}
    aligned = result.agreement["aligned_score"]
    signs, verdicts = final_signs(research, aligned, p)

    traded = signs[signs != 0]
    method = result.primary["method"]
    weights = pd.Series(0.0, index=signs.index)
    note = ""
    cov = result.primary["cov"]
    has_book = bool((traded > 0).any() and (traded < 0).any())
    if has_book:
        vol = pd.Series(np.sqrt(np.diag(cov.loc[traded.index, traded.index]) * 252), index=traded.index)
        params = ConstructionParams(
            gross=float(s.get("portfolio.gross_exposure", 2.0)), net=float(s.get("portfolio.net_exposure", 0.0)),
            max_position=max(float(s.get("portfolio.max_position", 0.35)),
                             float(s.get("portfolio.gross_exposure", 2.0)) / 2
                             / max(1, min((traded > 0).sum(), (traded < 0).sum()))))
        built = construction.build(method, traded, cov=cov, vol=vol, params=params,
                                   mu=pd.Series(0.0, index=traded.index), betas=result.primary.get("betas"))
        if built.status == "ok":
            weights.loc[built.weights.index] = built.weights
        else:
            note = f"{method} fallo con las direcciones corregidas ({built.note}); se usa equal_weight"
            weights.loc[traded.index] = construction.equal_weight(traded, params)
    else:
        note = ("tras la correccion no quedan largos Y cortos: no hay cartera long/short. "
                "Los niveles se dan por nombre, sin tamano de cartera.")

    capital = s.initial_capital
    rows = []
    for t in signs.index:
        close = result.clean.close[t].dropna()
        high = result.clean.high[t].reindex(close.index).fillna(close)
        low = result.clean.low[t].reindex(close.index).fillna(close)
        a = atr(high, low, close)
        side = int(signs[t]) if signs[t] != 0 else research[t]
        lv = levels(float(close.iloc[-1]), float(a.iloc[-1]), side, p)

        size = size_position(float(weights.get(t, 0.0)), verdicts[t], lv["entry"], lv["stop_pct"], capital, p)

        rows.append({
            "ticker": t,
            "research": "LONG" if research[t] > 0 else "SHORT",
            "verdict": verdicts[t],
            "action": ("BUY" if signs[t] > 0 else "SELL SHORT") if signs[t] != 0 else "NO TRADE",
            "aligned_score": float(aligned.get(t, np.nan)),
            **lv,
            "atr": float(a.iloc[-1]),
            "weight": math.copysign(size["notional"] / capital, side) if signs[t] != 0 else 0.0,
            "notional": size["notional"],
            "shares": size["shares"],
            "risk_capital_pct": size["shares"] * lv["risk_per_share"] / capital,
            "size_capped_by_risk": size["size_capped_by_risk"],
            "p_tp1_before_stop": hit_rate(close, high, low, a, side, p, p.tp_r[0]),
            "p_tp2_before_stop": hit_rate(close, high, low, a, side, p, p.tp_r[1]),
        })
    plan = pd.DataFrame(rows).set_index("ticker")
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
        if has_book:
            planned = -int(row["shares"]) if row["action"] == "SELL SHORT" else int(row["shares"])
        else:
            planned = planned_without_book(row["verdict"], research[t], current)
        target = target_shares(row["verdict"], research[t], planned, current)
        action, qty = order_for(current, target, row["entry"], capital, p)
        final[t] = _final(current, target, action)
        # Niveles del lado de la posicion que QUEDA: una posicion mantenida contra
        # research es larga aunque research dijera SHORT. Sin posicion, sin niveles.
        side = int(np.sign(final[t]))
        lv = levels(row["entry"], row["atr"], side, p) if side else {}
        orders.append({"ticker": t, "signal": action, "qty": qty, "current": current, "target": final[t],
                       "price": row["entry"], "stop": lv.get("stop"), "tp1": lv.get("tp1"),
                       "tp2": lv.get("tp2"), "verdict": row["verdict"],
                       "risk_capital_pct": abs(final[t]) * abs(row["entry"] - lv["stop"]) / capital if side else 0.0,
                       "hold_reason": ("no long/short book: kept as is" if not has_book and action == "HOLD"
                                       else "kept against research" if target != planned
                                       else "within hold band" if action == "HOLD" else "")})

    betas = result.primary.get("betas", pd.Series(1.0, index=plan.index))
    betas = betas.reindex(plan.index).fillna(1.0)
    final_w = pd.Series(final, dtype=float) * plan["entry"] / capital
    book_beta = float((final_w * betas).sum())

    bench = result.clean.benchmark
    hedge = None
    if p.hedge_beta:
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
                           "verdict": "HEDGE"})

    handled = set(plan.index) | ({bench} if p.hedge_beta else set())
    for t, current in positions.items():
        if t not in handled:
            orders.append({"ticker": t, "signal": "NOT ANALYZED", "qty": 0, "current": int(current),
                           "target": int(current), "verdict": "NOT ANALYZED"})

    return {"plan": plan, "corrections": corrections, "note": note, "params": p, "hedge": hedge,
            "orders": pd.DataFrame(orders).set_index("ticker"), "has_positions": bool(positions),
            "beta": book_beta, "gross": float(final_w.abs().sum()), "net": float(final_w.sum()),
            "risk_at_stops": float(pd.DataFrame(orders)["risk_capital_pct"].fillna(0.0).sum()),
            "levels_date": result.clean.close.index[-1],
            "wrds_missing": (None if (result.wrds is not None and result.wrds.available)
                             else {"weight": missing_wrds_weight(s.get("signal_weights", {}) or {}),
                                   "reason": result.wrds.reason if result.wrds is not None else "not requested"})}


def params_from_settings(settings) -> PlanParams:
    tp = settings.get("trade_plan", {}) or {}
    ag = settings.get("agreement", {}) or {}
    return PlanParams(
        stop_atr=float(tp.get("stop_atr", 2.5)),
        tp_r=tuple(float(x) for x in tp.get("take_profit_r", [1.5, 3.0]))[:2],
        max_risk_per_trade=float(tp.get("max_risk_per_trade", 0.01)),
        reduce_size=float(tp.get("reduce_size", 0.5)),
        weak=float(ag.get("weak", 0.15)),
        conflict=float(ag.get("conflict", -0.15)),
        flip=float(tp.get("flip_threshold", -0.5)),
        horizon_d=int(tp.get("horizon_d", 63)),
        hedge_beta=bool(tp.get("hedge_beta", True)),
        hedge_threshold=float(tp.get("hedge_threshold", 0.05)),
        hold_band=float(tp.get("hold_band", 0.005)),
    )
