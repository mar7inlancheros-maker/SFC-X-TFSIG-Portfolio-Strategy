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
    with pytest.raises(ValueError, match="0.99, not 99"):
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


# ---------------------------------------------------------------------------
#  Propiedades: lo que cualquier medida de riesgo seria tiene que cumplir
# ---------------------------------------------------------------------------

_PUNTUALES = {
    "historico": (var.historical_var, var.historical_es),
    "normal": (var.normal_var, var.normal_es),
    "cornish_fisher": (var.cornish_fisher_var, var.cornish_fisher_es),
}


def _colas_gordas(seed: int, n: int = 3000) -> pd.Series:
    rng = np.random.default_rng(seed)
    return _serie(0.0003 + rng.standard_t(4, n) * 0.01)


@pytest.mark.parametrize("method", sorted(_PUNTUALES))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_duplicar_la_posicion_duplica_el_var_y_el_es(method, seed):
    """Homogeneidad positiva: el riesgo de 2x la cartera es 2x el riesgo."""
    var_fn, es_fn = _PUNTUALES[method]
    r = _colas_gordas(seed)
    assert var_fn(r * 2, 0.99) == pytest.approx(2 * var_fn(r, 0.99), rel=1e-9)
    assert es_fn(r * 2, 0.99) == pytest.approx(2 * es_fn(r, 0.99), rel=1e-9)


@pytest.mark.parametrize("method", sorted(_PUNTUALES))
def test_un_retorno_fijo_extra_reduce_la_perdida_en_la_misma_cuantia(method):
    """Invarianza a traslaciones: sumar caja segura resta exactamente su retorno."""
    var_fn, es_fn = _PUNTUALES[method]
    r = _colas_gordas(4)
    assert var_fn(r + 0.001, 0.99) == pytest.approx(var_fn(r, 0.99) - 0.001, abs=1e-9)
    assert es_fn(r + 0.001, 0.99) == pytest.approx(es_fn(r, 0.99) - 0.001, abs=1e-9)


@pytest.mark.parametrize("method", sorted(_PUNTUALES))
@pytest.mark.parametrize("seed", [5, 6])
def test_es_cubre_al_var_y_ambos_crecen_con_la_confianza(method, seed):
    var_fn, es_fn = _PUNTUALES[method]
    r = _colas_gordas(seed)
    levels = [0.90, 0.95, 0.975, 0.99]
    vars_ = [var_fn(r, c) for c in levels]
    ess = [es_fn(r, c) for c in levels]
    assert vars_ == sorted(vars_)
    assert ess == sorted(ess)
    assert all(e >= v - 1e-12 for e, v in zip(ess, vars_, strict=True))


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_el_es_historico_es_subaditivo(seed):
    """Diversificar nunca aumenta el ES (el VaR si puede; el ES no)."""
    rng = np.random.default_rng(seed)
    a = _serie(rng.standard_t(3, 4000) * 0.01)
    b = _serie(rng.standard_t(3, 4000) * 0.01)
    assert var.historical_es(a + b, 0.975) <= var.historical_es(a, 0.975) + var.historical_es(b, 0.975)


def test_la_tabla_cumple_es_mayor_o_igual_que_var_en_todas_las_filas():
    table = var.var_table(_colas_gordas(10), [0.95, 0.99], [1, 5, 21], 0.94)
    assert (table["es"] >= table["var"] - 1e-12).all()
    assert table[["var", "es"]].notna().all().all()


# ---------------------------------------------------------------------------
#  Contraste de la estadistica hecha a mano contra scipy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stat", [0.0, 0.5, 3.84, 6.63, 25.0])
@pytest.mark.parametrize("df", [1, 2])
def test_la_cola_chi_cuadrado_coincide_con_scipy(stat, df):
    stats = pytest.importorskip("scipy.stats")
    assert var._chi2_sf(stat, df) == pytest.approx(stats.chi2.sf(stat, df), abs=1e-12)


def test_kupiec_coincide_con_la_razon_de_verosimilitud_de_scipy():
    stats = pytest.importorskip("scipy.stats")
    n, x, p = 1000, 18, 0.01
    lr, p_value = var.kupiec_pof(n, x, p)
    expected = -2 * (stats.binom.logpmf(x, n, p) - stats.binom.logpmf(x, n, x / n))
    assert lr == pytest.approx(expected, rel=1e-9)
    assert p_value == pytest.approx(stats.chi2.sf(expected, 1), rel=1e-9)


@pytest.mark.parametrize("exceptions", [3, 4, 5, 9, 10, 11])
def test_el_semaforo_usa_la_binomial_exacta(exceptions):
    stats = pytest.importorskip("scipy.stats")
    cumulative = stats.binom.cdf(exceptions, 250, 0.01)
    expected = "verde" if cumulative < 0.95 else "amarillo" if cumulative < 0.9999 - 1e-6 else "rojo"
    assert var.basel_zone(exceptions, 250, 0.01) == expected


# ---------------------------------------------------------------------------
#  Bordes: series cortas, sin excepciones, grados de libertad no soportados
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [var.historical_var, var.historical_es, var.normal_var,
                                var.normal_es, var.cornish_fisher_var, var.cornish_fisher_es])
def test_con_muy_pocos_datos_no_se_inventa_un_var(fn):
    assert np.isnan(fn(_serie(np.full(10, -0.01)), 0.99))


def test_fhs_y_ewma_con_series_vacias_o_cortas():
    assert np.isnan(var.fhs_var_es(_serie(np.zeros(5)), 0.99, 0.94)[0])
    sigma, sigma_next = var.ewma_sigma(pd.Series(dtype=float), 0.94)
    assert sigma.empty and np.isnan(sigma_next)
    assert var.tail_statistics(_serie(np.zeros(5))) == {}


def test_sin_ninguna_excepcion_christoffersen_no_rechaza():
    assert var.christoffersen_independence(np.zeros(500, dtype=bool)) == (0.0, 1.0)
    assert np.isnan(var.christoffersen_independence(np.array([True]))[1])


def test_la_cola_chi_cuadrado_solo_admite_uno_o_dos_grados():
    with pytest.raises(ValueError, match="1 or 2"):
        var._chi2_sf(1.0, 3)
    assert np.isnan(var._chi2_sf(-1.0, 1))


def test_cornish_fisher_aplica_la_formula_de_libro():
    """Cuantil de Cornish-Fisher escrito aparte: z + (z2-1)S/6 + (z3-3z)K/24 - (2z3-5z)S2/36."""
    rng = np.random.default_rng(12)
    r = _serie(-np.abs(rng.standard_t(5, 5000)) * 0.01 + rng.normal(0, 0.005, 5000))
    x = r.to_numpy()
    mu, sd = x.mean(), x.std(ddof=1)
    c = (x - mu) / sd
    s, k = np.mean(c ** 3), np.mean(c ** 4) - 3
    z = NormalDist().inv_cdf(0.01)
    z_cf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    assert var.cornish_fisher_var(r, 0.99) == pytest.approx(-(mu + z_cf * sd), rel=1e-9)
    # Asimetria negativa: la cola izquierda pesa mas que en la normal.
    assert s < 0
    assert var.cornish_fisher_var(r, 0.99) > var.normal_var(r, 0.99)


def test_dias_bajo_menos_tres_sigma_se_cuentan_a_tres_sigma():
    r = _colas_gordas(13)
    stats = var.tail_statistics(r)
    x = r.to_numpy()
    assert stats["freq_below_3sigma"] == pytest.approx(np.mean(x < x.mean() - 3 * x.std(ddof=1)))
