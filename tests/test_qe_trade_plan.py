import numpy as np
import pandas as pd
import pytest

from quant_engine import trade_plan as tp

P = tp.PlanParams()


@pytest.mark.parametrize("aligned,expected", [
    (0.8, "CONFIRM"), (0.15, "CONFIRM"), (0.0, "REDUCE"), (-0.15, "NO TRADE"),
    (-0.3, "NO TRADE"), (-0.5, "FLIP"), (-1.2, "FLIP"), (np.nan, "REDUCE"),
])
def test_verdict_thresholds(aligned, expected):
    assert tp.verdict(aligned, P) == expected


def test_final_signs_flip_and_no_trade():
    research = {"A": 1, "B": -1, "C": 1, "D": -1}
    aligned = pd.Series({"A": 0.5, "B": -0.8, "C": -0.3, "D": 0.0})
    signs, verdicts = tp.final_signs(research, aligned, P)
    assert signs.to_dict() == {"A": 1, "B": 1, "C": 0, "D": -1}
    assert verdicts.to_dict() == {"A": "CONFIRM", "B": "FLIP", "C": "NO TRADE", "D": "REDUCE"}


def test_levels_long_and_short_are_mirrored():
    long = tp.levels(100.0, 2.0, 1, P)
    short = tp.levels(100.0, 2.0, -1, P)
    assert long["stop"] == pytest.approx(95.0)
    assert long["tp1"] == pytest.approx(107.5)
    assert long["tp2"] == pytest.approx(115.0)
    assert short["stop"] == pytest.approx(105.0)
    assert short["tp1"] == pytest.approx(92.5)
    assert short["tp2"] == pytest.approx(85.0)
    assert long["stop_pct"] == pytest.approx(0.05)


def test_atr_constant_range():
    close = pd.Series(100.0, index=range(40))
    a = tp.atr(close + 1, close - 1, close)
    assert a.iloc[-1] == pytest.approx(2.0)
    assert a.iloc[:13].isna().all()


def _path(drift):
    close = pd.Series(100.0 + drift * np.arange(400))
    return close, close + 0.5, close - 0.5


def test_hit_rate_uptrend_long_wins_short_loses():
    close, high, low = _path(0.5)
    a = tp.atr(high, low, close)
    p = tp.PlanParams(horizon_d=40, history_d=200)
    assert tp.hit_rate(close, high, low, a, 1, p, 1.5) == pytest.approx(1.0)
    assert tp.hit_rate(close, high, low, a, -1, p, 1.5) == pytest.approx(0.0)


def test_hit_rate_same_day_both_counts_as_stop():
    # Rango diario enorme: cada dia toca stop y TP a la vez.
    close = pd.Series(100.0, index=range(200))
    high, low = close + 50, close - 50
    a = pd.Series(1.0, index=close.index)
    p = tp.PlanParams(horizon_d=10, history_d=100)
    assert tp.hit_rate(close, high, low, a, 1, p, 1.5) == 0.0


def test_params_from_settings_reads_yaml_sections():
    class S:
        def get(self, key, default=None):
            return {"trade_plan": {"stop_atr": 3, "take_profit_r": [2, 4], "max_risk_per_trade": 0.02,
                                   "flip_threshold": -0.6, "horizon_d": 21},
                    "agreement": {"weak": 0.2, "conflict": -0.2}}.get(key, default)

    p = tp.params_from_settings(S())
    assert (p.stop_atr, p.tp_r, p.max_risk_per_trade, p.flip, p.horizon_d) == (3.0, (2.0, 4.0), 0.02, -0.6, 21)
    assert (p.weak, p.conflict) == (0.2, -0.2)


def test_size_position_risk_cap_binds_on_wide_stop():
    # 20% del capital con stop al 10%: perderia 2% > 1%. Se recorta a 10% del capital.
    out = tp.size_position(0.20, "CONFIRM", 50.0, 0.10, 1_000_000, P)
    assert out["notional"] == pytest.approx(100_000)
    assert out["shares"] == 2_000
    assert out["size_capped_by_risk"]


def test_size_position_reduce_halves_and_no_trade_is_zero():
    out = tp.size_position(0.10, "REDUCE", 100.0, 0.05, 1_000_000, P)
    assert out["notional"] == pytest.approx(50_000)
    assert not out["size_capped_by_risk"]
    assert tp.size_position(0.10, "NO TRADE", 100.0, 0.05, 1_000_000, P)["shares"] == 0


def test_beta_hedge_sells_benchmark_when_book_is_long_beta():
    w = pd.Series({"A": 0.5, "B": -0.5})
    betas = pd.Series({"A": 1.4, "B": 0.8})
    h = tp.beta_hedge(w, betas, 500.0, 1_000_000, 0.05)
    assert h["action"] == "SELL SHORT"
    assert h["beta_before"] == pytest.approx(0.3)
    assert h["shares"] == 600
    assert abs(h["beta_after"]) < 1e-9


def test_beta_hedge_skips_small_beta():
    w = pd.Series({"A": 0.5, "B": -0.5})
    assert tp.beta_hedge(w, pd.Series({"A": 1.02, "B": 1.0}), 500.0, 1_000_000, 0.05) is None


CAP = 1_000_000


@pytest.mark.parametrize("current,target,expected", [
    (0, 0, ("NO TRADE", 0)),
    (0, 500, ("BUY", 500)),
    (0, -300, ("SELL SHORT", 300)),
    (500, 1_000, ("BUY", 500)),
    (1_000, 500, ("SELL", 500)),
    (500, 0, ("SELL", 500)),
    (-300, 0, ("BUY TO COVER", 300)),
    (-300, -600, ("SELL SHORT", 300)),
    (-600, -300, ("BUY TO COVER", 300)),
    (200, -300, ("FLIP TO SHORT", 500)),
    (-200, 300, ("FLIP TO LONG", 500)),
    (500, 510, ("HOLD", 0)),       # 10 x 100 = 1.000 < 0,5% de 1M
    (-500, -520, ("HOLD", 0)),
    (500, 500, ("HOLD", 0)),
])
def test_order_for(current, target, expected):
    assert tp.order_for(current, target, 100.0, CAP, P) == expected


def test_hold_band_does_not_apply_across_sides():
    # Pasar de +5 a -5 acciones es poco dinero, pero cambia de lado: no es HOLD.
    assert tp.order_for(5, -5, 100.0, CAP, P) == ("FLIP TO SHORT", 10)


def test_no_trade_keeps_position_against_research():
    # Research dice SHORT, el dato lo contradice, y ya se esta largo: no se cierra.
    assert tp.target_shares("NO TRADE", -1, 0, 400) == 400
    # Misma situacion pero la posicion va con research: se cierra.
    assert tp.target_shares("NO TRADE", -1, 0, -400) == 0
    assert tp.target_shares("CONFIRM", 1, 250, 100) == 250


def test_drivers_signed_by_research():
    comps = pd.Series({"momentum": 1.0, "value": -2.0, "quality": 0.5})
    w = {"momentum": 0.5, "value": 0.25, "quality": 0.25}
    pro, con = tp.drivers(comps, w, 1)
    assert pro[0].startswith("momentum") and con == ["value -0.50"]
    # Para un SHORT los signos se invierten: lo que resta a un largo suma a un corto.
    pro_s, con_s = tp.drivers(comps, w, -1)
    assert pro_s == ["value +0.50"] and con_s[0].startswith("momentum")


def test_drivers_renormalize_over_available():
    comps = pd.Series({"momentum": 1.0, "value": np.nan})
    pro, con = tp.drivers(comps, {"momentum": 0.5, "value": 0.5}, 1)
    assert pro == ["momentum +1.00"] and con == []


def test_missing_wrds_weight():
    w = {"momentum": 0.2, "value": 0.15, "quality": 0.15, "analyst": 0.15, "short_interest": 0.05,
         "volatility": 0.3}
    assert tp.missing_wrds_weight(w) == pytest.approx(0.5)
    assert tp.missing_wrds_weight({"momentum": 1.0}) == 0.0


def test_sin_cartera_no_se_liquida_lo_que_el_modelo_apoya():
    # Solo queda una pata: CONFIRM y REDUCE se mantienen del lado de research,
    # FLIP del lado contrario; lo demas va a cero.
    assert tp.planned_without_book("CONFIRM", 1, 500) == 500
    assert tp.planned_without_book("REDUCE", -1, -200) == -200
    assert tp.planned_without_book("CONFIRM", 1, -100) == 0      # del lado equivocado
    assert tp.planned_without_book("FLIP", 1, -300) == -300      # ya esta del lado del modelo
    assert tp.planned_without_book("FLIP", 1, 300) == 0
    assert tp.planned_without_book("NO TRADE", -1, -400) == 0
    assert tp.planned_without_book("CONFIRM", 1, 0) == 0         # sin tamano no se abre
