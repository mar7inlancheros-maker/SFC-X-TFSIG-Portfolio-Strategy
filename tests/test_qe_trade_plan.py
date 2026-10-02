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


def test_aviso_si_el_plan_no_queda_neutral():
    # Auditoria #16. Tras REDUCE y el tope de riesgo por operacion, el plan
    # quedaba en bruta 1,17 y neta +0,21 sin ningun aviso.
    w = pd.Series({"A": 0.18, "B": 0.09, "C": -0.06})
    notes = tp.book_warnings(w, gross=2.0, net=0.0)
    assert any("net +0.21" in n for n in notes)
    assert any("gross 0.33" in n for n in notes)


def test_sin_aviso_si_el_plan_cumple():
    w = pd.Series({"A": 0.5, "B": 0.5, "C": -0.5, "D": -0.5})
    assert tp.book_warnings(w, gross=2.0, net=0.0) == []
