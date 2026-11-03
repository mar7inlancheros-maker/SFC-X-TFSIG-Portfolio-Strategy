"""Catalogo de mutantes del plan long/short: bugs plausibles que un test DEBE cazar.

Mismo formato y mismas reglas que `risk_mutants.py`; lo corre
`python tools/risk_mutation.py --catalogo qe`. Aqui un bug no da una cifra de
riesgo rara: manda una orden del lado equivocado, con un stop mal puesto o con
una fecha que no es. Cada mutante es un error que alguien podria cometer al
tocar `trade_plan`, `playbook`, `sessions` o el acuerdo con el multifactor.
"""

from __future__ import annotations

from risk_mutants import Mutant

QE = "src/quant_engine/"


def _m(label: str, module: str, old: str, new: str, *tests: str) -> Mutant:
    return Mutant(label, QE + module, old, new, tuple(f"tests/test_qe_{t}.py" for t in tests))


MUTANTS: tuple[Mutant, ...] = (
    # -- niveles y regla de cada lado -----------------------------------------
    _m("for_side da la regla del corto a los largos", "trade_plan.py",
       "if side < 0 and p.short_stop_atr is not None and p.short_tp_r is not None:",
       "if side > 0 and p.short_stop_atr is not None and p.short_tp_r is not None:", "trade_plan"),
    _m("stop del lado equivocado", "trade_plan.py",
       '"stop": entry - side * r,', '"stop": entry + side * r,', "trade_plan"),
    _m("TP1 con el multiplo del TP2", "trade_plan.py",
       '"tp1": entry + side * p.tp_r[0] * r,', '"tp1": entry + side * p.tp_r[1] * r,', "trade_plan"),
    _m("FLIP no cambia el lado", "trade_plan.py",
       '{"CONFIRM": s, "REDUCE": s, "NO TRADE": 0, "FLIP": -s}[v]',
       '{"CONFIRM": s, "REDUCE": s, "NO TRADE": 0, "FLIP": s}[v]', "trade_plan"),
    # -- simulacion de la regla -------------------------------------------------
    _m("mismo dia stop y objetivo cuenta como objetivo", "trade_plan.py",
       "                if hit_stop:\n                    outcome = (side * (worst_stop - entry) / r, \"stop\", j)",
       "                if hit_stop and not hit_tp1:\n                    outcome = (side * (worst_stop - entry) / r, \"stop\", j)",
       "trade_plan"),
    _m("un gap que salta el stop se cobra al stop", "trade_plan.py",
       "worst_stop, worst_be = min(stop, hi), min(entry, hi)",
       "worst_stop, worst_be = stop, min(entry, hi)", "trade_plan"),
    _m("coste solo de ida", "trade_plan.py",
       "r_mult -= 2 * cost * entry / r", "r_mult -= cost * entry / r", "trade_plan"),
    _m("stop por tiempo un dia tarde", "trade_plan.py",
       "if time_stop is not None and j >= time_stop:", "if time_stop is not None and j > time_stop:",
       "playbook"),
    _m("stop por tiempo tambien en largos", "trade_plan.py",
       'if side < 0 and p.short_time_stop and window.get("review_day"):',
       'if p.short_time_stop and window.get("review_day"):', "playbook"),
    _m("break-even del TP1 sin el riesgo", "trade_plan.py",
       'out["breakeven_tp1"] = 1.0 / (1.0 + p.tp_r[0])', 'out["breakeven_tp1"] = 1.0 / p.tp_r[0]',
       "playbook"),
    _m("t de la regla sin Newey-West", "trade_plan.py",
       '"t_stat": hac_t_stat(r.to_numpy(), int(trades["days"].max())),',
       '"t_stat": hac_t_stat(r.to_numpy(), 0),', "trade_plan"),
    # -- tamano y ordenes ---------------------------------------------------------
    _m("tamano propio sin el tope de nocional", "trade_plan.py",
       "shares = math.floor(min(p.max_risk_per_trade * capital * mult / unit_risk,\n"
       "                            p.max_position * capital * mult / unit))",
       "shares = math.floor(p.max_risk_per_trade * capital * mult / unit_risk)", "playbook"),
    _m("tamano propio sin el tipo de cambio", "trade_plan.py",
       "unit_risk, unit = risk_per_share * fx, entry * fx", "unit_risk, unit = risk_per_share, entry",
       "playbook"),
    _m("sin libro la orden pierde el signo", "trade_plan.py",
       "        return supported * int(solo_shares)", "        return int(solo_shares)", "trade_plan"),
    _m("cobertura de beta del lado equivocado", "trade_plan.py",
       'return {"action": "SELL SHORT" if beta_book > 0 else "BUY", "shares": shares,',
       'return {"action": "BUY" if beta_book > 0 else "SELL SHORT", "shares": shares,', "trade_plan"),
    _m("rango de tiempo con stop por tiempo en largos", "trade_plan.py",
       "if side < 0 and time_stop_day is not None and pd.notna(time_stop_day) else None)",
       "if time_stop_day is not None and pd.notna(time_stop_day) else None)", "trade_plan"),
    # -- fechas -------------------------------------------------------------------
    _m("sesiones contadas desde el cierre de los niveles", "sessions.py",
       "entry = pd.Timestamp(levels_date).normalize() + off", "entry = pd.Timestamp(levels_date).normalize()",
       "sessions"),
    _m("NYSE sin Thanksgiving", "sessions.py",
       "        USLaborDay, USThanksgivingDay,\n", "        USLaborDay,\n", "sessions"),
    _m("Toronto con el calendario de NYSE", "sessions.py",
       '    return "CA" if t.endswith(CANADA) else "OTHER"', '    return "US" if t.endswith(CANADA) else "OTHER"',
       "sessions"),
    # -- fichas -------------------------------------------------------------------
    _m("stop de compra para un largo", "playbook.py",
       'stop_word = "SELL STOP" if side > 0 else "BUY STOP"', 'stop_word = "BUY STOP" if side > 0 else "SELL STOP"',
       "playbook"),
    _m("aviso de datos tambien en la cobertura", "playbook.py",
       '(orders["target"] != orders["current"]) & (orders["verdict"] != "HEDGE")',
       '(orders["target"] != orders["current"])', "playbook"),
    _m("total con su propia cobertura aunque el plan cubra", "playbook.py",
       "    if plan_hedge:\n        # La exposicion que se cubre", "    if False:\n        # La exposicion que se cubre",
       "playbook"),
    _m("DCF roto muestra el ponderado", "playbook.py",
       '    if val.get("critical"):\n        # Con un aviso critico', '    if False:\n        # Con un aviso critico',
       "playbook"),
    # -- multifactor --------------------------------------------------------------
    _m("multifactor sube tambien a los cortos", "data/multifactor.py",
       "return 0 if (direction < 0 and agree > 0) else agree", "return agree", "multifactor"),
    _m("quintil extremo al 10% en vez del 20%", "data/multifactor.py",
       "    if percentile >= 1 - EXTREME_QUINTILE:", "    if percentile >= 0.9:", "multifactor"),
)
