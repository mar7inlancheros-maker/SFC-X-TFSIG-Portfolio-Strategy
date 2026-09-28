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
    if (traded > 0).any() and (traded < 0).any():
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

        w = float(weights.get(t, 0.0))
        if verdicts[t] == "REDUCE":
            w *= p.reduce_size
        notional = abs(w) * capital
        risk_cap_notional = p.max_risk_per_trade * capital / lv["stop_pct"] if lv["stop_pct"] > 0 else notional
        final_notional = min(notional, risk_cap_notional) if signs[t] != 0 else 0.0
        shares = math.floor(final_notional / lv["entry"]) if lv["entry"] > 0 else 0

        rows.append({
            "ticker": t,
            "research": "LONG" if research[t] > 0 else "SHORT",
            "verdict": verdicts[t],
            "action": ("BUY" if signs[t] > 0 else "SELL SHORT") if signs[t] != 0 else "NO TRADE",
            "aligned_score": float(aligned.get(t, np.nan)),
            **lv,
            "weight": math.copysign(final_notional / capital, side) if signs[t] != 0 else 0.0,
            "notional": final_notional,
            "shares": shares,
            "risk_capital_pct": shares * lv["risk_per_share"] / capital,
            "size_capped_by_risk": bool(signs[t] != 0 and risk_cap_notional < notional),
            "p_tp1_before_stop": hit_rate(close, high, low, a, side, p, p.tp_r[0]),
            "p_tp2_before_stop": hit_rate(close, high, low, a, side, p, p.tp_r[1]),
        })
    plan = pd.DataFrame(rows).set_index("ticker")
    corrections = plan[plan["verdict"] != "CONFIRM"]
    return {"plan": plan, "corrections": corrections, "note": note, "params": p,
            "gross": float(plan["weight"].abs().sum()), "net": float(plan["weight"].sum())}


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
    )
