"""Motor long/short: lectura tecnica, maximo de 52 semanas y momentum residual."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine.factors import momentum, technical
from quant_engine.signals import composite


def _series(values, start="2025-01-01"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.bdate_range(start, periods=len(values)))


def test_maximo_de_52_semanas():
    s = _series(np.r_[np.linspace(100, 120, 200), np.linspace(120, 100, 100)])
    assert momentum.high_52w_ratio(s) == pytest.approx(100 / 120)
    assert np.isnan(momentum.high_52w_ratio(s.iloc[:100]))


def test_momentum_residual_ve_el_alfa_y_no_la_beta():
    rng = np.random.default_rng(4)
    m = _series(rng.normal(0.0005, 0.01, 1100))
    noise = _series(rng.normal(0.0, 0.004, 1100))
    beta_only = 1.5 * m + noise
    alpha = beta_only.copy()
    alpha.iloc[-273:-21] += 0.003            # alfa solo en los meses 12 a 2
    assert abs(momentum.residual_momentum(beta_only, m)) < 2.5
    assert momentum.residual_momentum(alpha, m) > 5
    # El mes mas reciente se salta: un salto ahi no cuenta.
    late = beta_only.copy()
    late.iloc[-10:] += 0.05
    assert momentum.residual_momentum(late, m) == pytest.approx(momentum.residual_momentum(beta_only, m))


def test_tendencia_alcista_da_mas_di_positivo_y_adx_alto():
    up = _series(np.linspace(50, 150, 300))
    d = technical.directional(up * 1.01, up * 0.99, up).iloc[-1]
    assert d["plus_di"] > d["minus_di"] and d["adx"] > 25


def test_pivotes_y_niveles_mas_cercanos():
    wave = 100 + 10 * np.sin(np.linspace(0, 6 * np.pi, 126))
    s = _series(wave)
    res, sup = technical.pivots(s, s)
    assert max(res) == pytest.approx(wave.max(), abs=0.5) and min(sup) == pytest.approx(wave.min(), abs=0.5)
    below, above = technical.nearest_levels(100.0, res, sup)
    assert below < 100 < above


def _view(values):
    s = _series(values)
    return technical.technical_view(s, s * 1.005, s * 0.995, _series(np.full(len(values), 1e6)))


def test_lectura_de_una_subida_ordenada():
    t = _view(100 * np.exp(np.linspace(0, 0.6, 300)))
    assert t["trend"] == "UP" and t["stance"] == "BULLISH" and t["rsi"] > 70
    assert t["from_high"] == pytest.approx(0.0, abs=1e-9)
    assert technical.describe(t)[0].startswith("Trend UP (strong")


def test_consejos_de_entrada_segun_el_lado():
    hot = _view(100 * np.exp(np.linspace(0, 0.6, 300)))
    long_tips = technical.timing_advice(hot, 1, hot["price"], hot["price"] * 0.95, hot["price"] * 1.08)
    assert any(t.startswith("Extended") for t in long_tips)
    assert any("trend supports the long" in t for t in long_tips)
    short_tips = technical.timing_advice(hot, -1, hot["price"], hot["price"] * 1.05, hot["price"] * 0.92)
    assert any(t.startswith("Shorting into a strong uptrend") for t in short_tips)
    cold = _view(100 * np.exp(np.linspace(0, -0.6, 300)))
    assert any(t.startswith("Oversold") for t in technical.timing_advice(cold, -1, cold["price"],
                                                                           cold["price"] * 1.05, cold["price"] * 0.9))


def test_stop_y_soporte():
    t = {"available": True, "rsi": 50, "pct_b": 0.5, "trend": "SIDEWAYS", "adx": 15, "plus_di": 20,
         "minus_di": 20, "atr": 2.0, "sma20": 100, "sma50": 100, "support": 95.0, "resistance": 104.0}
    good = technical.timing_advice(t, 1, entry=100, stop=94.5, tp1=103)
    assert any("good placement" in x for x in good)
    tight = technical.timing_advice(t, 1, entry=100, stop=95.2, tp1=110)
    assert any("a test of support would stop you out" in x for x in tight)
    assert any("Resistance 104.00 before TP1" in x for x in tight)


def test_factores_con_peso_cero_no_mueven_el_score():
    rng = np.random.default_rng(0)
    f = pd.DataFrame(rng.normal(size=(30, 4)), columns=["mom_12_1", "vol_annual", "pct_52w_high", "resid_mom"],
                     index=[f"T{i}" for i in range(30)])
    base = composite.quant_score(f, {"momentum": 0.5, "volatility": 0.5})
    with_zero = composite.quant_score(f, {"momentum": 0.5, "volatility": 0.5, "high_52w": 0.0,
                                          "residual_momentum": 0.0})
    pd.testing.assert_series_equal(base["quant_score"], with_zero["quant_score"])
    assert with_zero["high_52w"].notna().all() and with_zero["residual_momentum"].notna().all()


def test_movimiento_de_resultados_con_fechas_reales_y_con_aproximacion():
    idx = pd.bdate_range("2023-01-02", periods=780)
    r = pd.Series(0.001, index=idx)
    reports = [idx[60], idx[120], idx[185], idx[250], idx[310], idx[375]]
    for i, d in enumerate(reports):
        r.loc[d + pd.offsets.BDay(1)] = 0.08 if i % 2 else -0.06   # publican tras el cierre
    real = technical.earnings_moves(r, reports)
    assert real["source"].startswith("actual") and real["median"] == pytest.approx(0.07)
    proxy = technical.earnings_moves(r)
    assert proxy["source"].startswith("proxy") and proxy["n"] >= 8
    assert technical.earnings_moves(r.iloc[:100]) is None
