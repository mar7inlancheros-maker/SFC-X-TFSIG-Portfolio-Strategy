"""Motor long/short: precios verificados, numeros del plan que cuadran y fichas de ejecucion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine import playbook as pb
from quant_engine import trade_plan as tp
from quant_engine.data import cleaning, loader, validation
from quant_engine.data.loader import MarketData

# ---------------------------------------------------------------------------
#  Precios
# ---------------------------------------------------------------------------


def test_barra_con_maximo_bajo_el_cierre_se_repara_y_se_cuenta():
    idx = pd.bdate_range("2026-01-01", periods=3)
    frame = pd.DataFrame({"Open": [10.0, 10, 10], "High": [11.0, 9.5, 11], "Low": [9.0, 9, 10.5],
                          "Close": [10.5, 10.2, 10.0], "Volume": 1.0}, index=idx)
    fixed, n = cleaning.repair_ohlc(frame)
    assert n == 2
    assert fixed["High"].iloc[1] == pytest.approx(10.2) and fixed["Low"].iloc[2] == pytest.approx(10.0)
    assert (fixed["High"] >= fixed[["Open", "Close"]].max(axis=1)).all()


def test_salto_con_forma_de_split():
    s = pd.Series([100.0, 101, 50.4, 51, 51.5, 52])
    assert cleaning.split_like_jumps(s) == 1
    assert cleaning.split_like_jumps(pd.Series([100.0, 92, 95, 97])) == 0


def _market(**closes):
    cal = pd.bdate_range("2025-01-01", periods=200)
    data = {"SPY": pd.Series(100.0, index=cal)}
    data.update(closes)
    ohlcv = {t: pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6}) for t, c in data.items()}
    return cal, MarketData(ohlcv=ohlcv)


def test_salto_x20_excluye_y_ultimo_precio_raro_avisa():
    cal = pd.bdate_range("2025-01-01", periods=200)
    unit = pd.Series(100.0, index=cal)
    unit.iloc[100:] = 1.0                       # libras leidas como peniques
    last = pd.Series(100.0, index=cal)
    last.iloc[-1] = 140.0                       # el ultimo cierre es la entrada del plan
    stale = pd.Series(100.0, index=cal[:-4])    # sin precio en las 4 ultimas sesiones
    _, market = _market(UNIT=unit, LAST=last, STALE=stale)
    clean = cleaning.align(market, "SPY")
    q = validation.quality_report(clean, market, ["UNIT", "LAST", "STALE"], cal[0])
    assert q.loc["UNIT", "status"] == "FAIL" and "x20" in q.loc["UNIT", "notes"]
    assert q.loc["LAST", "status"] == "WARN" and "+40%" in q.loc["LAST", "notes"]
    assert q.loc["STALE", "status"] == "WARN" and "4 sessions old" in q.loc["STALE", "notes"]


# ---------------------------------------------------------------------------
#  Conviccion, tamano propio e integridad del plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verdict, aligned, evidence, unproven, expected", [
    ("CONFIRM", 0.7, "predictive", False, "HIGH"),
    ("CONFIRM", 0.7, "none", False, "MEDIUM"),       # sin evidencia, techo MEDIUM
    ("CONFIRM", 0.2, "predictive", False, "MEDIUM"),  # el backtest del nombre ya no resta: no predijo
    ("REDUCE", 0.05, "predictive", False, "LOW"),
    ("FLIP", -0.9, "none", True, "LOW"),
    ("FLIP", -0.9, "inverse", False, "LOW"),
    ("NO TRADE", -0.3, "none", False, ""),
])
def test_conviccion(verdict, aligned, evidence, unproven, expected):
    assert tp.conviction(verdict, aligned, evidence, unproven) == expected


def test_tamano_propio_por_riesgo_con_tope_de_nocional_y_fx():
    p = tp.PlanParams(max_risk_per_trade=0.01, max_position=0.35)
    # Stop a 5 por accion, CAD a 0,70: el stop cuesta 3,5 USD por accion -> 2.857 acciones.
    out = tp.solo_size(entry=100.0, risk_per_share=5.0, fx=0.7, capital=1_000_000, p=p)
    assert out["shares"] == 2857 and out["risk_pct"] <= 0.01
    # Stop muy cerca: el tope de nocional (35%) manda.
    tight = tp.solo_size(entry=100.0, risk_per_share=0.1, fx=1.0, capital=1_000_000, p=p)
    assert tight["shares"] == 3500 and tight["notional"] == pytest.approx(350_000)
    assert tp.solo_size(100.0, 5.0, 1.0, 1_000_000, p, mult=0.5)["shares"] == 1000


def _good_plan():
    plan = pd.DataFrame({
        "action": ["BUY", "SELL SHORT"], "entry": [100.0, 50.0], "stop": [95.0, 52.0],
        "tp1": [107.5, 47.0], "tp2": [115.0, 44.0], "fx": [1.0, 0.7],
        "shares": [1000, 2000], "notional": [100_050.0, 70_010.0],
    }, index=["A", "B.TO"])
    plan["risk_capital_pct"] = plan["shares"] * (plan["entry"] - plan["stop"]).abs() * plan["fx"] / 1_000_000
    orders = pd.DataFrame({"signal": ["BUY", "SELL SHORT"], "qty": [1000, 2000], "current": [0, 0],
                           "target": [1000, -2000], "verdict": ["CONFIRM", "CONFIRM"],
                           "risk_capital_pct": plan["risk_capital_pct"].to_numpy()}, index=plan.index)
    return plan, orders


def test_un_plan_correcto_pasa_la_verificacion():
    plan, orders = _good_plan()
    p = tp.PlanParams()
    assert tp.check_plan(plan, orders, p, 1_000_000, float(orders["risk_capital_pct"].sum()), None) == []


@pytest.mark.parametrize("breaks, needle", [
    (lambda pl, o: pl.__setitem__("stop", [105.0, 52.0]), "side of a"),           # stop de un largo por encima
    (lambda pl, o: pl.__setitem__("shares", [2000, 2000]), "do not match"),     # acciones de mas
    (lambda pl, o: pl.__setitem__("entry", [np.nan, 50.0]), "invalid"),
    (lambda pl, o: o.__setitem__("qty", [999, 2000]), "order of 999"),
])
def test_la_verificacion_caza_numeros_que_no_cuadran(breaks, needle):
    plan, orders = _good_plan()
    breaks(plan, orders)
    plan["risk_capital_pct"] = (plan["shares"] * (plan["entry"] - plan["stop"]).abs() * plan["fx"]
                                / 1_000_000).fillna(0.0)
    errors = tp.check_plan(plan, orders, tp.PlanParams(), 1_000_000,
                           float(orders["risk_capital_pct"].sum()), None)
    assert any(needle in e for e in errors), errors


def test_la_verificacion_caza_riesgo_por_encima_del_tope_y_cobertura_mal():
    plan, orders = _good_plan()
    plan.loc["A", ["shares", "notional"]] = [3000, 300_050.0]
    plan["risk_capital_pct"] = plan["shares"] * (plan["entry"] - plan["stop"]).abs() * plan["fx"] / 1_000_000
    hedge = {"shares": 100, "price": 500.0, "beta_before": 0.2, "beta_after": 0.2}
    errors = tp.check_plan(plan, orders, tp.PlanParams(), 1_000_000, 0.0, hedge)
    assert any("cap" in e for e in errors)
    assert any("sum" in e for e in errors)
    assert any("hedge" in e for e in errors)


# ---------------------------------------------------------------------------
#  Fichas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("si, dtc, expected", [
    (0.25, 2.0, "HIGH"), (0.05, 9.0, "HIGH"), (0.12, 1.0, "MEDIUM"), (0.02, 1.0, "LOW"), (None, None, "UNKNOWN"),
])
def test_riesgo_de_squeeze(si, dtc, expected):
    assert pb.squeeze_risk(si, dtc) == expected


def _row(entry=100.0, stop=105.0, tp1=92.5, tp2=85.0, ccy="USD"):
    return pd.Series({"entry": entry, "stop": stop, "tp1": tp1, "tp2": tp2, "atr": 2.0, "currency": ccy})


def test_como_hacer_el_corto_paso_a_paso():
    steps = pb.execution_steps("ARES", -1, 1000, current=0, row=_row(), fx=1.0, capital=1_000_000,
                               horizon_d=63, levels_date=pd.Timestamp("2026-10-16"), borrow_cost=0.0025,
                               short_facts={"si_pct": 0.117, "days_to_cover": 13.6, "squeeze": "HIGH"},
                               earnings=pd.Timestamp("2026-11-01"))
    text = " ".join(steps)
    assert steps[0].startswith("Locate first") and "squeeze risk HIGH" in steps[0]
    assert "SELL SHORT 1,000 ARES" in text and "BUY STOP 1,000 at 105.00" in text and "ABOVE the entry" in text
    assert "BUY TO COVER LIMIT 500 at 92.50" in text and "BUY TO COVER LIMIT 500 at 85.00" in text
    assert "Margin: about 50,000 USD" in text and "dividend" in text
    assert "Earnings on 2026-11-01" in text


def test_si_se_tiene_el_lado_contrario_primero_se_cierra():
    steps = pb.execution_steps("AMC", 1, 300, current=-500, row=_row(100, 95, 107.5, 115), fx=1.0,
                               capital=1_000_000, horizon_d=63, levels_date=pd.Timestamp("2026-10-16"),
                               borrow_cost=0.0025)
    assert steps[0] == "Close the current short first: BUY TO COVER 500 at market."
    assert any("SELL STOP 300 at 95.00" in s for s in steps)
    assert any("earnings date unknown" in s for s in steps)


def test_corto_en_cad_dice_la_moneda_y_el_margen_en_usd():
    steps = pb.execution_steps("ATZ.TO", -1, 1000, 0, _row(ccy="CAD"), fx=0.7, capital=1_000_000,
                               horizon_d=63, levels_date=pd.Timestamp("2026-10-16"), borrow_cost=0.0025)
    text = " ".join(steps)
    assert "on TSX, in CAD" in text and "about 35,000 USD" in text and "0.7000 USD per CAD" in text


def test_cobertura_de_una_posicion_suelta():
    short = pb.standalone_hedge(1.5, 1000, -1, 100.0, 1.0, "SPY", 500.0)
    assert short["action"] == "BUY" and short["shares"] == 300
    long_ = pb.standalone_hedge(1.0, 1000, 1, 100.0, 1.0, "SPY", 500.0)
    assert long_["action"] == "SELL SHORT" and long_["shares"] == 200


def test_fundamentales_avisan_de_valores_no_creibles():
    lines = pb.fundamentals_lines(pd.Series({"earnings_yield": 0.036, "fcf_yield": 1.046, "roe": 0.25,
                                             "meanrec": 2.0, "numrec": 15}))
    assert lines[0] == "earnings yield 3.6% (P/E 28)"
    assert "verify" in lines[1]
    assert "15 analysts" in lines[-1]


def test_por_que_traduce_contribuciones_a_numeros():
    row = pd.Series({"si_pct_float": 0.117, "mom_12_1": 0.45, "mom_6m": 0.2, "mom_3m": -0.05})
    out = pb.why_lines("short_interest -0.15, momentum +0.28", row)
    assert out[0] == "short interest (-0.15): 11.7% of shares sold short"
    assert out[1].startswith("momentum (+0.28): 12-1 momentum +45%")


# ---------------------------------------------------------------------------
#  Fechas de resultados
# ---------------------------------------------------------------------------


def test_fechas_de_resultados_sin_red_y_sin_inventar(monkeypatch):
    store = {}
    monkeypatch.setattr(loader.cache, "read_json", lambda name: store.get(name))
    monkeypatch.setattr(loader.cache, "write_json", lambda name, data: store.__setitem__(name, data))

    def fetch(t):
        if t == "BOOM":
            raise RuntimeError("Yahoo cayo")
        return {"OK": pd.Timestamp("2026-11-05"), "OLD": pd.Timestamp("2026-07-30")}.get(t)

    today = pd.Timestamp("2026-10-18")
    out = loader.load_earnings_dates(["OK", "OLD", "BOOM", "NONE"], today, fetch=fetch)
    assert out == {"OK": pd.Timestamp("2026-11-05")}   # pasada, caida o sin dato: sin fecha
    # Segunda llamada del mismo dia: de la cache, sin pedir nada.
    assert loader.load_earnings_dates(["OK"], today, fetch=lambda t: 1 / 0) == {"OK": pd.Timestamp("2026-11-05")}


def test_avisos_de_datos_solo_lo_que_cambia_la_ejecucion():
    flags = pb.execution_flags({"last_gap": 4, "last_jump": -0.3, "split_like": 1, "extreme_returns": 0,
                                "ohlc_repaired": 0, "filled": 33})
    assert len(flags) == 3
    assert flags[0].startswith("last price is 4 sessions old") and "-30%" in flags[1]
    assert pb.execution_flags({"filled": 33, "last_gap": 1}) == []   # festivos de otra bolsa: no cambian la orden


def test_las_posiciones_sueltas_se_ven_juntas_con_una_sola_cobertura():
    cards = {
        "A": {"ticker": "A", "signal": "SHORT", "steps": ["x"], "sized_by": "standalone", "notional": 100_000.0,
              "beta": 1.5, "risk_pct": 0.005},
        "B": {"ticker": "B", "signal": "LONG", "steps": ["x"], "sized_by": "standalone", "notional": 50_000.0,
              "beta": None, "risk_pct": 0.01},
        "C": {"ticker": "C", "signal": "NO TRADE", "steps": [], "sized_by": "standalone"},
    }
    total = pb.standalone_total(cards, "SPY", 500.0)
    assert total["names"] == ["A", "B"] and total["net"] == pytest.approx(-50_000)
    assert total["beta_usd"] == pytest.approx(-150_000 + 50_000)    # beta desconocida cuenta como 1
    assert total["hedge_action"] == "BUY" and total["hedge_shares"] == 200
    assert total["risk_pct"] == pytest.approx(0.015)
    # Si el plan ya cubre en sus ordenes, el total da ESA cobertura.
    planned = pb.standalone_total(cards, "SPY", 500.0, {"action": "BUY", "shares": 190, "price": 501.0,
                                                         "beta_before": -0.095})
    assert planned["hedge_in_plan"] and planned["hedge_shares"] == 190 and planned["bench_price"] == 501.0
    assert planned["beta_usd"] == pytest.approx(-190 * 501.0)       # la exposicion que la cobertura cubre


def test_con_cobertura_en_el_plan_la_ficha_no_pide_otra():
    row = pd.Series({"entry": 100.0, "stop": 105.0, "tp1": 92.5, "tp2": 85.0, "atr": 2.0, "currency": "USD"})
    hedge = {"beta": 1.5, "ticker": "SPY", "shares": 30, "price": 500.0, "action": "BUY",
             "in_plan": {"action": "BUY", "shares": 190, "ticker": "SPY"}}
    steps = pb.execution_steps("X", -1, 100, 0, row, 1.0, 1_000_000, 63, pd.Timestamp("2026-10-30"), 0.0025,
                               hedge=hedge)
    text = " ".join(steps)
    assert "one BUY 190 SPY (1-year betas of the book): do not add a hedge per name" in text
    assert "Optional market hedge" not in text
    note = pb.card_size_note({"shares": 100, "sized_by": "standalone", "plan_hedged": True}, 0.01)
    assert "market hedge takes out the beta" in note and "directional" not in note


# ---------------------------------------------------------------------------
#  Rango de tiempo de los cortos
# ---------------------------------------------------------------------------


def test_la_regla_anota_el_dia_del_tp1_sin_cambiar_el_resultado():
    idx = pd.bdate_range("2026-01-01", periods=7)
    c = pd.Series([100, 100, 104, 108, 108, 108, 108], index=idx, dtype=float)
    out = tp.simulate_trades(c, c + 0.2, c - 0.2, pd.Series(1.0, index=idx), 1,
                             tp.PlanParams(horizon_d=5, history_d=1, stop_atr=2.5, tp_r=(1.5, 3.0)))
    # Entrada en la segunda sesion (history_d = 1): el TP1 llega al dia siguiente.
    assert out["tp1_day"].iloc[0] == 1 and out["r_mult"].iloc[0] == pytest.approx(2.25)


def _trades(n_fast=60, n_slow=40, late_hits=10, late_day=40):
    """60 llegan al TP1 en 5-14 dias; de 40 lentos, `late_hits` llegan despues (dia `late_day`)."""
    rows = [{"r_mult": 1.0, "exit": "tp1_tp2", "days": 30, "tp1_day": 5 + i % 10} for i in range(n_fast)]
    rows += [{"r_mult": 0.5, "exit": "tp1_time", "days": 63, "tp1_day": late_day} for _ in range(late_hits)]
    rows += [{"r_mult": -0.3, "exit": "time", "days": 63, "tp1_day": np.nan} for _ in range(n_slow - late_hits)]
    return pd.DataFrame(rows)


def test_ventana_de_tenencia_y_revision_por_tiempo():
    p = tp.PlanParams(tp_r=(1.5, 3.0))                          # break-even del TP1: 1 / 2,5 = 40%
    w = tp.holding_window(_trades(), p)
    assert w["n"] == 100 and w["p25"] == 30 and w["p75"] == 63
    tp1_days = pd.Series([5 + i % 10 for i in range(60)] + [40] * 10, dtype=float)
    k = int(np.ceil(tp1_days.quantile(0.75)))
    assert w["review_day"] == k
    # Vivos en k: los 40 lentos (y los rapidos con TP1 despues de k, si los hay).
    alive_fast = sum(1 for i in range(60) if 5 + i % 10 > k)
    assert w["late_tp1_rate"] == pytest.approx((10 + alive_fast) / (40 + alive_fast))
    assert w["cut_at_review"]                                   # 10/40 < 40% de break-even
    # 20 de 40 lentos llegan tarde (dia 50): seguir esperando compensa.
    patient = tp.holding_window(_trades(late_hits=20, late_day=50), p)
    assert patient["late_tp1_rate"] == pytest.approx(0.5) and not patient["cut_at_review"]


def test_rango_de_tiempo_del_corto_con_fechas_coste_y_dividendo():
    row = pd.Series({"entry": 100.0, "hold_p25": 10.0, "hold_median": 21.0, "hold_p75": 40.0,
                     "tp1_median_days": 8.0, "review_day": 15.0, "late_tp1_rate": 0.2, "cut_at_review": True})
    lines = pb.time_window("ARES", row, -1, 1000, 1.0, pd.Timestamp("2026-10-30"), 63, 0.0252,
                           ex_dividend=pd.Timestamp("2026-11-20"))
    text = " ".join(lines)
    assert lines[0].startswith("Open on 2026-11-02")            # viernes -> lunes
    assert "10-40 sessions, from 2026-11-16 to 2026-12-30" in text   # Thanksgiving y Navidad no son sesiones
    assert "Review on 2026-11-23" in text and "consider covering" in text    # regla apagada: solo informa
    assert "Hard exit: cover whatever is left by 2027-02-03" in text
    assert "about 210 USD" in text                              # 100.000 x 2,52% x 21/252
    assert "Ex-dividend on 2026-11-20" in text


def test_fechas_ex_dividendo(monkeypatch):
    store = {}
    monkeypatch.setattr(loader.cache, "read_json", lambda name: store.get(name))
    monkeypatch.setattr(loader.cache, "write_json", lambda name, data: store.__setitem__(name, data))
    out = loader.load_ex_dividend_dates(["X"], pd.Timestamp("2026-10-30"),
                                        fetch=lambda t: pd.Timestamp("2026-11-20"))
    assert out == {"X": pd.Timestamp("2026-11-20")} and "qe_ex_dividend_dates" in store


def test_stop_por_tiempo_cierra_sin_tp1_y_no_cambia_la_regla_por_defecto():
    idx = pd.bdate_range("2026-01-01", periods=8)
    c = pd.Series([100, 100, 100.5, 101, 101.5, 102, 102, 102], index=idx, dtype=float)
    p = tp.PlanParams(horizon_d=6, history_d=1, stop_atr=2.5, tp_r=(1.5, 3.0))
    plain = tp.simulate_trades(c, c + 0.2, c - 0.2, pd.Series(1.0, index=idx), 1, p)
    timed = tp.simulate_trades(c, c + 0.2, c - 0.2, pd.Series(1.0, index=idx), 1, p, time_stop=2)
    assert plain["exit"].iloc[0] == "time"
    assert timed["exit"].iloc[0] == "time_stop" and timed["days"].iloc[0] == 2
    assert timed["r_mult"].iloc[0] == pytest.approx((101 - 100) / 2.5)


@pytest.mark.parametrize("si, rate", [(0.02, 0.004), (0.10, 0.02), (0.25, 0.08), (None, None)])
def test_tramo_de_prestamo_por_interes_en_corto(si, rate):
    tier = pb.borrow_tier(si)
    assert (tier is None) if rate is None else tier[1] == rate


def test_rango_de_tiempo_avisa_del_prestamo_caro():
    row = pd.Series({"entry": 100.0, "hold_p25": 10.0, "hold_median": 21.0, "hold_p75": 40.0})
    lines = pb.time_window("X", row, -1, 1000, 1.0, pd.Timestamp("2026-10-30"), 63, 0.0025, si_pct=0.25)
    assert any("likely hard to borrow" in x and "about 667 USD" in x for x in lines)   # 100.000 x 8% x 21/252


def test_stop_por_tiempo_fuera_de_muestra_elige_el_dia_en_la_primera_mitad():
    idx = pd.bdate_range("2023-01-02", periods=400)
    rng = np.random.default_rng(9)
    tp1 = np.where(rng.random(400) < 0.6, rng.integers(3, 15, 400), np.nan)
    trades = pd.DataFrame({"r_mult": rng.normal(0, 1, 400), "exit": np.where(np.isnan(tp1), "time", "tp1_tp2"),
                           "days": 63, "tp1_day": tp1}, index=idx)
    timed = trades.assign(r_mult=trades["r_mult"] + 0.1)        # el stop por tiempo suma 0,1R
    out = tp.time_stop_out_of_sample(trades, timed, tp.PlanParams())
    first_half = trades.iloc[:200]
    assert out["review_day"] == tp.holding_window(first_half, tp.PlanParams())["review_day"]
    assert out["split"] == idx[200] and out["n"] == 200
    assert out["improvement"] == pytest.approx(0.1)
    assert tp.time_stop_out_of_sample(trades.iloc[:150], timed.iloc[:150], tp.PlanParams()) is None


def test_salida_a_bolsa_reciente_tiene_ficha_limitada_sin_tamano():
    from types import SimpleNamespace

    from quant_engine.settings import build_settings

    cal = pd.bdate_range("2025-01-01", periods=300)
    rng = np.random.default_rng(1)
    ipo = pd.Series(50 * np.exp(np.cumsum(rng.normal(0, 0.02, 70))), index=cal[-70:])
    market = MarketData(ohlcv={
        "SPY": pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1e6}, index=cal),
        "NEW": pd.DataFrame({"Open": ipo, "High": ipo * 1.02, "Low": ipo * 0.98, "Close": ipo, "Volume": 1e6}),
    })
    clean = cleaning.align(market, "SPY")
    quality = validation.quality_report(clean, market, ["NEW"], cal[0])
    result = SimpleNamespace(settings=build_settings(("AAPL",), ("NEW",)), quality=quality, clean=clean, valuation={})
    card = pb.limited_card(result, "NEW", tp.PlanParams())
    assert card["signal"] == "NOT ANALYZED" and card["headline"].startswith("LIMITED VIEW NEW: only 70 sessions")
    assert "levels for research's SHORT" in card["fundamentals"][0] and card["shares"] == 0 and not card["steps"]
    assert card["technical"] and card["technical"][0].startswith("Trend")
    short = SimpleNamespace(**{**result.__dict__, "quality": quality.assign(sessions=20)})
    assert pb.limited_card(short, "NEW", tp.PlanParams())["headline"].startswith("NOT ANALYZED")


def test_stop_por_tiempo_es_regla_solo_en_cortos():
    w = {"review_day": 20.0}
    assert tp.rule_time_stop(-1, w, tp.PlanParams()) == 20
    assert tp.rule_time_stop(1, w, tp.PlanParams()) is None
    assert tp.rule_time_stop(-1, w, tp.PlanParams(short_time_stop=False)) is None
    assert tp.rule_time_stop(-1, {}, tp.PlanParams()) is None


def test_el_corto_lleva_el_paso_del_stop_por_tiempo():
    row = pd.Series({"entry": 100.0, "stop": 105.0, "tp1": 92.5, "tp2": 85.0, "atr": 2.0, "currency": "USD",
                     "time_stop_day": 20.0})
    steps = pb.execution_steps("X", -1, 100, 0, row, 1.0, 1_000_000, 63, pd.Timestamp("2026-10-30"), 0.0025)
    assert any(s.startswith("Time stop: if TP1 has not been reached by 2026-12-01") for s in steps)
    long_row = row.copy()
    long_row[["stop", "tp1", "tp2"]] = [95.0, 107.5, 115.0]
    longs = pb.execution_steps("X", 1, 100, 0, long_row, 1.0, 1_000_000, 63,
                               pd.Timestamp("2026-10-30"), 0.0025)
    assert not any(s.startswith("Time stop") for s in longs)


def test_la_ficha_del_corto_da_el_stop_por_tiempo_y_la_del_largo_no_corta():
    base = {"entry": 100.0, "hold_p25": 10.0, "hold_median": 21.0, "hold_p75": 40.0, "tp1_median_days": 8.0,
            "review_day": 15.0, "late_tp1_rate": 0.2, "cut_at_review": True}
    short = pd.Series({**base, "time_stop_day": 15.0, "bt_exp_r": 0.12, "bt_exp_r_no_time_stop": 0.05})
    text = " ".join(pb.time_window("ARES", short, -1, 1000, 1.0, pd.Timestamp("2026-10-30"), 63, 0.0252))
    assert "Time stop (rule for shorts): if TP1 has not been reached by 2026-11-23 (15 sessions)" in text
    assert "+0.12R with it and +0.05R without it" in text
    long_ = " ".join(pb.time_window("ARES", pd.Series(base), 1, 1000, 1.0, pd.Timestamp("2026-10-30"), 63, 0.0))
    assert "keep the stop and hold" in long_ and "Time stop" not in long_


def test_pasos_y_ventana_dan_la_misma_fecha_para_la_misma_regla():
    row = pd.Series({"entry": 100.0, "stop": 105.0, "tp1": 92.5, "tp2": 85.0, "atr": 2.0, "currency": "USD",
                     "time_stop_day": 20.0, "hold_p25": 10.0, "hold_median": 18.0, "hold_p75": 20.0})
    d = pd.Timestamp("2026-10-30")
    steps = " ".join(pb.execution_steps("ARES", -1, 100, 0, row, 1.0, 1_000_000, 63, d, 0.0025))
    window = " ".join(pb.time_window("ARES", row, -1, 100, 1.0, d, 63, 0.0025))
    assert "reached by 2026-12-01 (20 sessions)" in steps and "reached by 2026-12-01 (20 sessions)" in window
    assert "hit by 2027-02-03 (63 sessions)" in steps and "left by 2027-02-03 (63 sessions)" in window


def test_el_html_no_pega_las_lineas_de_la_ficha():
    from quant_engine.reporting.plan_html import _sentences

    assert _sentences(["verify the value per share", "DCF check: financial company (x)", "Monte Carlo: 82%."]) == \
        "verify the value per share. DCF check: financial company (x). Monte Carlo: 82%."


def _val(critical):
    return {"available": True, "source": "SEC XBRL", "fiscal_end": "2026-06-30", "trading_currency": "USD",
            "price": 38.03, "value_cases": {"bear": 0.0, "base": 0.0, "bull": 1481.13}, "value_expected": 370.28,
            "upside_base": -1.0, "stance": "NOT DECISION-GRADE" if critical else "UNDERVALUED", "wacc": 0.093,
            "terminal_growth": 0.025, "tv_share": 0.5, "implied_cagr": 0.105, "base_cagr": 0.207, "years": 10,
            "mc_prob_above_price": 0.47, "flags": ["thin operating margin (0.7%)"],
            "critical": ["equity value is zero or negative"] if critical else []}


def test_un_dcf_sin_grado_de_decision_no_da_escenarios_ni_ponderado():
    lines, verdict = pb.valuation_lines("SHORT", _val(True))
    text = " ".join(lines)
    assert "NOT DECISION-GRADE" in text and "for reference only" in text and "DCF check: thin operating" in text
    assert "1,481" not in text and "370.28" not in text and "Monte Carlo:" not in text and "implies" not in text
    assert "not decision-grade" in verdict
    ok = " ".join(pb.valuation_lines("LONG", _val(False))[0])
    assert "probability-weighted 370.28" in ok and "Monte Carlo: value above the price in 47%" in ok


def test_el_resumen_del_stop_por_tiempo_cuenta_solo_los_cortos():
    from types import SimpleNamespace

    from rich.console import Console

    from quant_engine.reporting import terminal

    plan = pd.DataFrame({"signal": ["SHORT", "SHORT", "LONG"], "oos_ts_improvement": [0.1, -0.02, 0.3]},
                        index=["A", "B", "C"])
    con = Console(record=True, width=200)
    terminal.time_stop_evidence(con, SimpleNamespace(trade_plan={"plan": plan}))
    text = con.export_text()
    assert "improved E(R) in 1 of 2 shorts here (mean +0.04R)" in text


def test_los_avisos_de_datos_van_junto_a_la_orden():
    from types import SimpleNamespace

    orders = pd.DataFrame({"target": [-717, 0, 162], "current": [0, 0, 0], "verdict": ["REDUCE", "NO TRADE", "HEDGE"]},
                          index=["INTC", "AMC", "SPY"])
    stats = {"INTC": {"last_jump": -0.29}, "AMC": {"last_jump": -0.4}, "SPY": {"last_jump": -0.3}}
    r = SimpleNamespace(trade_plan={"orders": orders}, clean=SimpleNamespace(stats=stats))
    out = pb.order_flags(r)
    assert list(out) == ["INTC"] and "-29%" in out["INTC"][0]     # sin orden o cobertura: no se avisa aqui
