"""VaR, ES y su backtest, con series cuya respuesta se conoce de antemano."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig.risk import var


def _serie(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


def test_el_var_historico_es_el_cuantil_con_signo_de_perdida():
    # 1..100 como perdidas del -1% al -100%: el 5% peor empieza en -95%.
    r = _serie(-np.arange(1, 101) / 100.0)
    assert var.historical_var(r, 0.95) == pytest.approx(0.9505, abs=1e-3)
    # El ES es la media de la cola, siempre al menos tan grande como el VaR.
    assert var.historical_es(r, 0.95) >= var.historical_var(r, 0.95)


def test_var_y_es_normales_coinciden_con_la_formula_cerrada():
    rng = np.random.default_rng(1)
    r = _serie(rng.normal(0.0005, 0.01, 5000))
    mu, sd = r.mean(), r.std(ddof=1)
    z = NormalDist().inv_cdf(0.01)
    assert var.normal_var(r, 0.99) == pytest.approx(-(mu + z * sd))
    assert var.normal_es(r, 0.99) == pytest.approx(-(mu - sd * NormalDist().pdf(z) / 0.01))


def test_cornish_fisher_sin_asimetria_ni_curtosis_es_la_normal():
    # Una muestra simetrica y mesocurtica construida a mano: cuantiles normales.
    grid = (np.arange(20000) + 0.5) / 20000
    r = _serie([NormalDist(0.0, 0.01).inv_cdf(u) for u in grid])
    assert var.cornish_fisher_var(r, 0.99) == pytest.approx(var.normal_var(r, 0.99), rel=0.02)


def test_con_colas_gordas_cornish_fisher_supera_a_la_normal_al_99():
    rng = np.random.default_rng(2)
    r = _serie(rng.standard_t(3, 20000) * 0.01)
    assert var.cornish_fisher_var(r, 0.99) > var.normal_var(r, 0.99)


def test_la_confianza_se_escribe_como_fraccion():
    with pytest.raises(ValueError, match="0.99, no 99"):
        var.historical_var(_serie(np.zeros(100)), 99)


def test_la_sigma_ewma_del_dia_t_no_ve_el_retorno_de_t():
    r = _serie([0.01] * 100 + [0.20])
    sigma, sigma_next = var.ewma_sigma(r, 0.94)
    # El choque del ultimo dia no puede estar en su propia prevision...
    assert sigma.iloc[-1] == pytest.approx(0.01, rel=0.05)
    # ...pero si en la de manana.
    assert sigma_next > 0.04


def test_fhs_sube_cuando_la_volatilidad_de_hoy_es_mayor_que_la_media():
    rng = np.random.default_rng(3)
    calm = rng.normal(0, 0.005, 1000)
    storm = rng.normal(0, 0.03, 40)
    r = _serie(np.concatenate([calm, storm]))
    fhs_v, _ = var.fhs_var_es(r, 0.99, 0.94)
    assert fhs_v > var.historical_var(r, 0.99)


def test_la_tabla_trae_todos_los_metodos_confianzas_y_horizontes():
    rng = np.random.default_rng(4)
    r = _serie(rng.normal(0, 0.01, 800))
    table = var.var_table(r, [0.95, 0.99], [1, 21], 0.94)
    assert len(table) == 4 * 2 * 2
    assert set(table["method"]) == set(var.METHODS)
    # A 21 dias se pierde mas que a uno, con cualquier metodo.
    one = table[table["horizon_d"] == 1].set_index(["method", "confidence"])["var"]
    month = table[table["horizon_d"] == 21].set_index(["method", "confidence"])["var"]
    assert (month > one).all()


def test_kupiec_acepta_la_frecuencia_prometida_y_rechaza_el_exceso():
    _, p_ok = var.kupiec_pof(1000, 10, 0.01)
    _, p_bad = var.kupiec_pof(1000, 30, 0.01)
    assert p_ok > 0.9
    assert p_bad < 0.001


def test_kupiec_tambien_rechaza_un_var_que_nunca_se_excede():
    _, p = var.kupiec_pof(2000, 0, 0.01)
    assert p < 0.001


def test_christoffersen_detecta_excepciones_en_racimo():
    spread = np.zeros(1000, dtype=bool)
    spread[::100] = True
    clustered = np.zeros(1000, dtype=bool)
    clustered[500:510] = True
    assert var.christoffersen_independence(spread)[1] > 0.5
    assert var.christoffersen_independence(clustered)[1] < 0.001


@pytest.mark.parametrize("exceptions, zone", [(0, "verde"), (4, "verde"), (5, "amarillo"),
                                              (9, "amarillo"), (10, "rojo")])
def test_semaforo_de_basilea_con_250_dias_al_99(exceptions, zone):
    assert var.basel_zone(exceptions, 250, 0.01) == zone


def test_las_previsiones_del_backtest_usan_solo_el_pasado():
    """Un salto el ultimo dia no puede cambiar el VaR previsto para ese dia."""
    rng = np.random.default_rng(5)
    base = rng.normal(0, 0.01, 400)
    shocked = base.copy()
    shocked[-1] = -0.30
    a = var.rolling_var_forecasts(_serie(base), 0.99, 250, 0.94)
    b = var.rolling_var_forecasts(_serie(shocked), 0.99, 250, 0.94)
    cols = ["historico", "ewma_normal", "fhs"]
    pd.testing.assert_series_equal(a[cols].iloc[-1], b[cols].iloc[-1])


def test_un_var_correcto_sale_verde_y_uno_normal_con_colas_gordas_falla():
    rng = np.random.default_rng(6)
    r = _serie(rng.standard_t(3, 4000) * 0.01)
    summary, forecasts = var.var_backtest(r, 0.99, 250, 0.94)
    by_model = summary.set_index("model")
    assert len(forecasts) > 3000
    # La normal EWMA subestima la cola de una t(3): mas excepciones de las debidas.
    assert by_model.loc["ewma_normal", "rate"] > by_model.loc["fhs", "rate"]
    assert by_model.loc["ewma_normal", "kupiec_p"] < 0.05


def test_estadisticas_de_cola_marcan_la_curtosis():
    rng = np.random.default_rng(7)
    stats = var.tail_statistics(_serie(rng.standard_t(3, 5000) * 0.01))
    assert stats["excess_kurtosis"] > 3
    assert stats["freq_below_3sigma"] > stats["normal_freq_below_3sigma"]
    assert math.isfinite(stats["tail_ratio"])
