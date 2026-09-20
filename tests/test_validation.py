"""Contrastes estadisticos: que detecten senal cuando la hay y silencio cuando no."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig import validation as val


def _panel_con_senal(n_dates=60, n_names=60, strength=0.5, seed=1):
    """Panel sintetico donde el score SI predice el retorno, con ruido."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n_dates, freq="ME")
    rows = []
    for date in dates:
        scores = rng.normal(size=n_names)
        noise = rng.normal(scale=0.08, size=n_names)
        forward = strength * 0.01 * scores + noise
        for i in range(n_names):
            rows.append({"date": date, "ticker": f"T{i:03d}",
                         "score_composite": scores[i], "forward_return": forward[i]})
    return pd.DataFrame(rows)


def _panel_sin_senal(n_dates=60, n_names=60, seed=2):
    return _panel_con_senal(n_dates=n_dates, n_names=n_names, strength=0.0, seed=seed)


def test_newey_west_con_cero_rezagos_es_el_error_estandar_clasico():
    serie = pd.Series(np.random.default_rng(0).normal(size=200))
    clasico = serie.std(ddof=0) / np.sqrt(len(serie))
    assert val.newey_west_se(serie, 0) == pytest.approx(clasico, rel=1e-9)


def test_newey_west_amplia_el_error_con_autocorrelacion_positiva():
    """Serie persistente: ignorar la autocorrelacion infla el t artificialmente."""
    rng = np.random.default_rng(3)
    x = [0.0]
    for _ in range(500):
        x.append(0.8 * x[-1] + rng.normal())
    serie = pd.Series(x)
    assert val.newey_west_se(serie, 6) > val.newey_west_se(serie, 0)


def test_ic_detecta_senal_cuando_existe():
    ic = val.information_coefficient(_panel_con_senal())
    resumen = val.ic_summary(ic)
    assert resumen["mean"] > 0.02
    assert resumen["t_stat"] > 2.0
    assert resumen["p_value"] < 0.05


def test_ic_no_inventa_senal_donde_no_la_hay():
    ic = val.information_coefficient(_panel_sin_senal())
    resumen = val.ic_summary(ic)
    assert abs(resumen["mean"]) < 0.05
    assert resumen["p_value"] > 0.05


def test_quintiles_ordenan_de_mejor_a_peor_con_senal():
    quantiles = val.quantile_returns(_panel_con_senal(strength=1.5), n_quantiles=5)
    medias = quantiles[[f"Q{i}" for i in range(1, 6)]].mean()
    assert medias["Q1"] > medias["Q5"]
    # Monotonicidad cercana a -1: el retorno cae segun empeora el quintil.
    assert val.monotonicity(quantiles, 5) < -0.8


def test_spread_de_quintiles_no_es_significativo_sin_senal():
    quantiles = val.quantile_returns(_panel_sin_senal(), n_quantiles=5)
    prueba = val.test_mean(quantiles["spread"])
    assert prueba.p_value > 0.05


def test_fama_macbeth_recupera_el_signo_del_factor():
    panel = _panel_con_senal(strength=2.0)
    panel["score_value"] = panel["score_composite"]
    resumen = val.fama_macbeth(panel, ["score_value"])
    assert resumen.loc["score_value", "mean"] > 0
    assert resumen.loc["score_value", "t_stat"] > 2.0


def test_ventanas_walk_forward_no_solapan_entrenamiento_y_prueba():
    dates = pd.date_range("2012-01-31", periods=140, freq="ME")
    ventanas = val.walk_forward_windows(dates, train_years=4, test_years=1)
    assert len(ventanas) >= 5
    for ventana in ventanas:
        assert ventana.test_start > ventana.train_end
        assert ventana.train_start < ventana.train_end


def test_walk_forward_avanza_en_el_tiempo():
    dates = pd.date_range("2012-01-31", periods=140, freq="ME")
    ventanas = val.walk_forward_windows(dates, train_years=4, test_years=1)
    inicios = [v.train_start for v in ventanas]
    assert inicios == sorted(inicios)
    assert inicios[1] > inicios[0]


def test_ic_fuera_de_muestra_se_mide_solo_en_el_tramo_de_prueba():
    panel = _panel_con_senal(n_dates=100)
    ventanas = val.walk_forward_windows(
        pd.DatetimeIndex(sorted(panel["date"].unique())), 4, 1
    )
    tabla = val.out_of_sample_ic(panel, ventanas)
    assert len(tabla) == len(ventanas)
    assert (tabla["test_months"] > 0).all()
    # Con senal estacionaria, dentro y fuera de muestra deben parecerse.
    assert tabla["test_ic"].mean() > 0.0
