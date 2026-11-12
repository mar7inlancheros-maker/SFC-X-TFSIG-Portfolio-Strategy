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
    assert technical.describe(t)[0].startswith("Long-term trend UP (50/200-day averages); 14-day direction up, strong")


def test_consejos_de_entrada_segun_el_lado():
    hot = _view(100 * np.exp(np.linspace(0, 0.6, 300)))
    long_tips = technical.timing_advice(hot, 1, hot["price"], hot["price"] * 0.95, hot["price"] * 1.08)
    assert any(t.startswith("Extended") for t in long_tips)
    assert any(t.startswith("Long-term trend UP, on the side of the long") for t in long_tips)
    # Cada condicion medida cita lo que se midio y no aconseja esperar.
    flagged = [t for t in long_tips if t.startswith(("Extended", "Long-term trend"))]
    assert len(flagged) == 2 and all("Measured (" in t and "no reason to change the order" in t for t in flagged)
    assert all("smaller R" not in t and "more likely to be hit" not in t for t in long_tips)
    assert all(t.isascii() for t in long_tips)
    short_tips = technical.timing_advice(hot, -1, hot["price"], hot["price"] * 1.05, hot["price"] * 0.92)
    assert any(t.startswith("Shorting into a strong uptrend") for t in short_tips)
    cold = _view(100 * np.exp(np.linspace(0, -0.6, 300)))
    assert any(t.startswith("Oversold") for t in technical.timing_advice(cold, -1, cold["price"],
                                                                           cold["price"] * 1.05, cold["price"] * 0.9))


def test_stop_y_soporte():
    t = {"available": True, "rsi": 50, "pct_b": 0.5, "trend": "SIDEWAYS", "adx": 15, "plus_di": 20,
         "minus_di": 20, "atr": 2.0, "sma20": 100, "sma50": 100, "support": 95.0, "resistance": 104.0}
    good = technical.timing_advice(t, 1, entry=100, stop=94.5, tp1=103)
    assert any(x.startswith("Stop 94.50 sits just under support 95.00.") for x in good)
    tight = technical.timing_advice(t, 1, entry=100, stop=95.2, tp1=110)
    assert any("a test of support reaches the stop" in x for x in tight)
    before = [x for x in tight if x.startswith("Resistance 104.00 before TP1")]
    # Medido: casi todas las entradas largas tienen una resistencia antes del TP1.
    assert before and "It holds on 87% of entries, so it says little." in before[0]
    # Ninguna afirmacion sin medir: ni "buena colocacion" ni "se frenara ahi".
    assert all("good placement" not in x and "stall" not in x for x in good + tight)
    assert all("Measured (" in x for x in good + tight)


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


def _ohlc(n=420, seed=11):
    rng = np.random.default_rng(seed)
    c = _series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n))))
    return c, c * (1 + rng.uniform(0, 0.01, n)), c * (1 - rng.uniform(0, 0.01, n))


def test_los_indicadores_de_una_sesion_no_ven_el_futuro():
    c, h, lo = _ohlc()
    base = technical.indicator_series(c, h, lo)
    later = c.copy()
    later.iloc[300:] *= 3                     # precios posteriores a la sesion 299, triplicados
    moved = technical.indicator_series(later, h.where(h.index < c.index[300], h * 3),
                                       lo.where(lo.index < c.index[300], lo * 3))
    pd.testing.assert_frame_equal(base.iloc[:300], moved.iloc[:300])


def test_la_ficha_y_el_estudio_leen_lo_mismo():
    c, h, lo = _ohlc()
    view = technical.technical_view(c, h, lo)
    last = technical.indicator_series(c, h, lo).iloc[-1]
    for key in ("trend", "rsi", "pct_b", "adx", "plus_di", "minus_di", "atr", "sma200_slope"):
        assert view[key] == last[key], key
    flags = technical.timing_flags(view)
    for side, prefix in ((1, "long"), (-1, "short")):
        for name, value in flags[side].items():
            assert bool(value) == bool(last[f"{prefix}_{name}"]), (side, name)


def test_condiciones_de_entrada_por_lado():
    hot = {"rsi": 75, "pct_b": 0.9, "trend": "UP", "adx": 30, "plus_di": 30, "minus_di": 10}
    f = technical.timing_flags(hot)
    assert f[1]["extended"] and f[1]["with_trend"] and not f[1]["against_trend"]
    assert f[-1]["against_trend"] and not f[-1]["extended"] and not f[-1]["with_trend"]
    weak = dict(hot, adx=20)                  # sin fuerza no hay "tendencia fuerte en contra"
    assert not technical.timing_flags(weak)[-1]["against_trend"]
    cold = {"rsi": 50, "pct_b": -0.1, "trend": "DOWN", "adx": 30, "plus_di": 10, "minus_di": 30}
    g = technical.timing_flags(cold)
    assert g[-1]["extended"] and g[-1]["with_trend"] and g[1]["against_trend"]


def test_niveles_frente_al_stop_y_al_tp1_por_lado():
    t = {"atr": 2.0, "support": 95.0, "resistance": 104.0}
    # Largo: stop 94,5 justo bajo el soporte 95 (0,5 < 0,5 x 2 = 1).
    assert technical.level_flags(t, 1, 100, 94.5, 103) == {
        "stop_behind_level": True, "level_inside_stop": False, "level_before_tp1": False}
    # Largo: stop 95,2 por encima del soporte: un test del soporte lo toca; resistencia antes del TP1.
    assert technical.level_flags(t, 1, 100, 95.2, 110) == {
        "stop_behind_level": False, "level_inside_stop": True, "level_before_tp1": True}
    # Largo: soporte 3 ATR por encima del stop: ni detras ni dentro.
    assert not any(technical.level_flags(t, 1, 100, 89.0, 103).values())
    # Corto: espejo con la resistencia detras del stop y el soporte antes del TP1.
    assert technical.level_flags(t, -1, 100, 104.6, 97) == {
        "stop_behind_level": True, "level_inside_stop": False, "level_before_tp1": False}
    assert technical.level_flags(t, -1, 100, 103.9, 90) == {
        "stop_behind_level": False, "level_inside_stop": True, "level_before_tp1": True}
    nan = {"atr": 2.0, "support": float("nan"), "resistance": float("nan")}
    assert not any(technical.level_flags(nan, 1, 100, 94, 106).values())
    assert not any(technical.level_flags(nan, -1, 100, 106, 94).values())
