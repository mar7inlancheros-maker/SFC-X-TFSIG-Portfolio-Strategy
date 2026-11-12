"""Inferencia por nombre y para la canasta (quant_engine.inference.statistics)."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from quant_engine.inference import statistics as st


def _ar1(phi: float, n: int = 3000, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.01, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return pd.Series(x)


def test_ratio_de_varianzas_distingue_tendencia_reversion_y_paseo_aleatorio():
    iid = st.variance_ratio_test(_ar1(0.0), 5)
    assert abs(iid["vr"] - 1) < 0.1 and abs(iid["z"]) < 3
    trend = st.variance_ratio_test(_ar1(0.3), 5)
    assert trend["vr"] > 1.3 and trend["z"] > 4 and trend["p"] < 0.001
    revert = st.variance_ratio_test(_ar1(-0.3), 5)
    assert revert["vr"] < 0.8 and revert["z"] < -4
    assert st.variance_ratio_test(_ar1(0.0, n=50), 5) == {}          # muy poca historia


def test_el_z_robusto_coincide_con_el_homocedastico_si_la_varianza_es_constante():
    # Con varianza constante, delta(j) ~ 1 y theta ~ 2(2q - 1)(q - 1)/(3q).
    r = _ar1(0.0, n=20000, seed=4)
    q = 5
    out = st.variance_ratio_test(r, q)
    theta_h = 2 * (2 * q - 1) * (q - 1) / (3 * q)
    z_h = math.sqrt(len(r)) * (out["vr"] - 1) / math.sqrt(theta_h)
    assert out["z"] == pytest.approx(z_h, rel=0.05)


def test_jarque_bera_ve_las_colas_gordas():
    rng = np.random.default_rng(2)
    normal = st.jarque_bera(pd.Series(rng.normal(0, 0.01, 2000)))
    fat = st.jarque_bera(pd.Series(rng.standard_t(3, 2000) * 0.01))
    assert normal["p"] > 0.01 and abs(normal["excess_kurtosis"]) < 0.5
    assert fat["p"] < 1e-6 and fat["excess_kurtosis"] > 2


def test_bootstrap_por_bloques_y_su_anchura_con_solape():
    rng = np.random.default_rng(3)
    iid = st.block_bootstrap_mean(rng.normal(0.1, 1.0, 2000), block=1)
    assert iid["lo"] < 0.1 < iid["hi"] and iid["p_positive"] > 0.99
    # Observaciones solapadas: el bloque ensancha el intervalo, como debe.
    overlap = _ar1(0.9, n=2000, seed=5)
    narrow = st.block_bootstrap_mean(overlap, block=1)
    wide = st.block_bootstrap_mean(overlap, block=40)
    assert (wide["hi"] - wide["lo"]) > 2 * (narrow["hi"] - narrow["lo"])
    assert st.block_bootstrap_mean([0.1] * 10, block=5) == {}


def test_sharpe_probabilistico_con_la_formula():
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    sr = r.mean() / r.std(ddof=1)
    d = r - r.mean()
    g3 = (d ** 3).mean() / (d ** 2).mean() ** 1.5
    g4 = (d ** 4).mean() / (d ** 2).mean() ** 2
    expected = NormalDist().cdf(sr * math.sqrt(len(r) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    assert st.probabilistic_sharpe(r) == pytest.approx(expected)
    better = st.probabilistic_sharpe(r + 0.0005)
    assert better > st.probabilistic_sharpe(r)


def test_el_sharpe_deflactado_descuenta_las_variantes_probadas():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0006, 0.01, 1000))
    one = st.deflated_sharpe(r, [r.mean() / r.std()])
    many = st.deflated_sharpe(r, rng.normal(0.02, 0.03, 50))
    assert one["sr_star"] == 0.0 and one["dsr"] == pytest.approx(one["psr"])
    assert many["sr_star"] > 0 and many["dsr"] < many["psr"]
    # E[max] de N normales estandar ~ 2,25 con N = 50 (Bailey y Lopez de Prado).
    assert st.expected_max_sharpe(np.r_[np.full(25, -1.0), np.full(25, 1.0)]) == pytest.approx(
        1.0102 * ((1 - st.EULER_GAMMA) * NormalDist().inv_cdf(1 - 1 / 50)
                  + st.EULER_GAMMA * NormalDist().inv_cdf(1 - 1 / (50 * math.e))), rel=1e-3)


def test_regimen_de_volatilidad_ve_un_pico_reciente():
    rng = np.random.default_rng(8)
    r = pd.Series(np.r_[rng.normal(0, 0.01, 900), rng.normal(0, 0.035, 25)])
    out = st.volatility_regime(r)
    assert out["label"] == "elevated" and out["current_vol"] > out["long_run_vol"]
    assert st.volatility_regime(r.iloc[:50]) == {}                   # GARCH necesita un ano


def test_la_ficha_lo_dice_en_frases():
    rng = np.random.default_rng(9)
    lines = st.describe(st.name_statistics(pd.Series(rng.standard_t(4, 1200) * 0.012)))
    text = " ".join(lines)
    assert "Random-walk test" in text and "Jarque-Bera" in text and "GARCH(1,1)" in text
    assert text.isascii()


def test_holm_y_benjamini_hochberg_a_mano():
    p = [0.01, 0.04, 0.03, float("nan"), 0.5]
    # Holm: 4 x 0,01 = 0,04; 3 x 0,03 = 0,09; 2 x 0,04 = 0,08 -> 0,09 (monotono); 1 x 0,5.
    np.testing.assert_allclose(st.holm(p), [0.04, 0.09, 0.09, np.nan, 0.5])
    # BH: 0,01 x 4/1 = 0,04; 0,03 x 4/2 = 0,06; 0,04 x 4/3 = 0,0533; 0,5 -> minimo desde arriba.
    np.testing.assert_allclose(st.benjamini_hochberg(p), [0.04, 0.0533333, 0.0533333, np.nan, 0.5], rtol=1e-5)
    assert np.isnan(st.holm([np.nan])).all() and len(st.benjamini_hochberg([])) == 0
    rng = np.random.default_rng(10)
    u = rng.uniform(size=40)
    assert (st.benjamini_hochberg(u) <= st.holm(u) + 1e-12).all()       # BH nunca es mas estricto


def test_el_veredicto_de_la_canasta_usa_el_q_valor():
    # 20 nombres de paseo aleatorio y uno con tendencia: con 40 pruebas, el p
    # suelto marcaria alguno por suerte; el q-valor deja solo al de verdad.
    stats = {f"N{i}": st.name_statistics(_ar1(0.0, n=1500, seed=100 + i)) for i in range(20)}
    stats["TREND"] = st.name_statistics(_ar1(0.3, n=1500, seed=99))
    st.adjust_random_walk(stats)
    q = {t: min(v["q_value"] for v in s["random_walk"].values()) for t, s in stats.items()}
    assert q["TREND"] < 0.001
    assert all(v["family"] == 42 for s in stats.values() for v in s["random_walk"].values())
    for t, s in stats.items():
        for v in s["random_walk"].values():
            assert v["q_value"] >= v["p"] - 1e-12
    text = " ".join(st.describe(stats["TREND"]))
    assert "trending" in text and "Benjamini-Hochberg" in text and "42 tests" in text
    nulls = [v for t, s in stats.items() if t != "TREND" for v in s["random_walk"].values()]
    assert sum(v["p"] < 0.05 for v in nulls) >= 1          # el p suelto marca alguno por suerte...
    assert sum(v["q_value"] < 0.05 for v in nulls) == 0    # ...el q-valor, ninguno
    fluke = next(t for t, s in stats.items() if t != "TREND" and min(v["p"] for v in s["random_walk"].values()) < 0.05)
    assert "trending" not in " ".join(st.describe(stats[fluke]))
    assert "mean-reverting" not in " ".join(st.describe(stats[fluke]))
