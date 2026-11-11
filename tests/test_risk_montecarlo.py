"""Monte Carlo: bootstrap de la historia y simulacion parametrica de la cartera."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig.risk import montecarlo


def _rets(columns: dict[str, np.ndarray]) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=pd.bdate_range("2015-01-01", periods=n))


def test_con_retorno_constante_el_bootstrap_da_el_resultado_exacto():
    r = _rets({"strategy": np.full(500, 0.001)})
    res = montecarlo.bootstrap_paths(r, [252], n_paths=200, mean_block=10, seed=1)
    assert np.allclose(res.terminal[252][:, 0], 1.001 ** 252 - 1.0)
    assert np.allclose(res.max_drawdown[252], 0.0)


def test_la_semilla_hace_reproducible_el_resultado():
    rng = np.random.default_rng(0)
    r = _rets({"strategy": rng.normal(0.0005, 0.01, 1000)})
    a = montecarlo.bootstrap_paths(r, [252], n_paths=500, mean_block=21, seed=42)
    b = montecarlo.bootstrap_paths(r, [252], n_paths=500, mean_block=21, seed=42)
    np.testing.assert_array_equal(a.terminal[252], b.terminal[252])


def test_el_descuento_por_supervivencia_solo_toca_a_la_estrategia():
    """Estrategia identica al benchmark: con descuento, queda siempre detras."""
    rng = np.random.default_rng(1)
    series = rng.normal(0.0005, 0.01, 1500)
    r = _rets({"strategy": series, "benchmark": series})
    plain = montecarlo.bootstrap_summary(
        montecarlo.bootstrap_paths(r, [252], n_paths=1000, mean_block=21, seed=3), [0.1]
    )
    cut = montecarlo.bootstrap_summary(
        montecarlo.bootstrap_paths(r, [252], n_paths=1000, mean_block=21, seed=3,
                                   haircut_annual=0.02), [0.1]
    )
    assert plain.loc[252, "prob_underperform"] == 0.0
    assert cut.loc[252, "prob_underperform"] == 1.0
    assert cut.loc[252, "p50"] < plain.loc[252, "p50"]


def test_el_resumen_trae_probabilidades_coherentes():
    rng = np.random.default_rng(2)
    r = _rets({"strategy": rng.normal(0.0004, 0.012, 2000),
               "benchmark": rng.normal(0.0003, 0.010, 2000)})
    res = montecarlo.bootstrap_paths(r, [252, 756], n_paths=2000, mean_block=21, seed=4)
    summary = montecarlo.bootstrap_summary(res, [0.10, 0.30])
    for h in (252, 756):
        row = summary.loc[h]
        assert row["p5"] <= row["p25"] <= row["p50"] <= row["p75"] <= row["p95"]
        assert 0.0 <= row["prob_loss"] <= 1.0
        # Un drawdown del 30% es mas raro que uno del 10%.
        assert row["prob_dd_30"] <= row["prob_dd_10"]
    # A mas horizonte, mas drawdown acumulado posible.
    assert summary.loc[756, "median_max_drawdown"] <= summary.loc[252, "median_max_drawdown"]


def test_el_bootstrap_exige_historia_suficiente_para_el_bloque():
    r = _rets({"strategy": np.zeros(30)})
    with pytest.raises(ValueError, match="twice the block length"):
        montecarlo.bootstrap_paths(r, [252], n_paths=100, mean_block=21, seed=1)


def _cov(n: int = 5, vol: float = 0.015, corr: float = 0.4) -> pd.DataFrame:
    names = [f"N{i}" for i in range(n)]
    matrix = np.full((n, n), corr * vol ** 2)
    np.fill_diagonal(matrix, vol ** 2)
    return pd.DataFrame(matrix, index=names, columns=names)


def test_la_t_de_student_tiene_mas_cola_que_la_normal_con_la_misma_covarianza():
    cov = _cov()
    w = pd.Series(0.2, index=cov.index)
    table, _ = montecarlo.parametric_portfolio_mc(
        w, cov, horizons_d=[1], confidences=[0.99], n_sims=100_000, t_dof=4, seed=1
    )
    es = table.set_index("distribution")["es"]
    assert es["t_student"] > es["normal"]


def test_el_var_normal_simulado_coincide_con_el_analitico():
    cov = _cov()
    w = pd.Series(0.2, index=cov.index)
    table, _ = montecarlo.parametric_portfolio_mc(
        w, cov, horizons_d=[1, 21], confidences=[0.99], n_sims=200_000, t_dof=5, seed=2
    )
    sigma = float(np.sqrt(w @ cov @ w))
    normal = table[table["distribution"] == "normal"].set_index("horizon_d")["var"]
    assert normal[1] == pytest.approx(2.3263 * sigma, rel=0.03)
    assert normal[21] == pytest.approx(2.3263 * sigma * np.sqrt(21), rel=0.03)


def test_el_es_por_componentes_suma_el_es_total():
    cov = _cov()
    w = pd.Series([0.4, 0.3, 0.1, 0.1, 0.1], index=cov.index)
    table, component = montecarlo.parametric_portfolio_mc(
        w, cov, horizons_d=[1], confidences=[0.95, 0.99], n_sims=50_000, t_dof=5, seed=3
    )
    total = table[(table["distribution"] == "t_student") & (table["confidence"] == 0.99)]["es"].iloc[0]
    assert component["component_es"].sum() == pytest.approx(total, rel=1e-9)
    # A igual riesgo por nombre, el de mas peso explica mas cola.
    assert component.iloc[0]["ticker"] == "N0"


def test_una_covarianza_no_definida_positiva_no_rompe_la_simulacion():
    cov = _cov(3)
    cov.iloc[2, :] = 0.0
    cov.iloc[:, 2] = 0.0  # un nombre sin variacion: matriz singular
    w = pd.Series(1 / 3, index=cov.index)
    table, _ = montecarlo.parametric_portfolio_mc(
        w, cov, horizons_d=[1], confidences=[0.99], n_sims=5000, t_dof=5, seed=4
    )
    assert np.isfinite(table["var"]).all()


def test_la_racha_bajo_el_agua_se_mide_en_sesiones():
    # Siempre cae: toda la trayectoria esta bajo el maximo inicial.
    down = _rets({"strategy": np.full(300, -0.001)})
    res = montecarlo.bootstrap_paths(down, [21, 252], n_paths=100, mean_block=10, seed=1)
    assert (res.max_underwater[252] == 252).all()
    assert (res.max_underwater[21] == 21).all()
    # Siempre sube: nunca esta bajo el agua.
    up = _rets({"strategy": np.full(300, 0.001)})
    summary = montecarlo.bootstrap_summary(
        montecarlo.bootstrap_paths(up, [252], n_paths=100, mean_block=10, seed=1), [0.05]
    )
    assert summary.loc[252, "p95_max_underwater_d"] == 0


def test_los_bloques_conservan_la_autocorrelacion():
    """Con retornos autocorrelacionados, bloques largos dan mas dispersion a 21
    dias que remuestrear dia a dia: el bootstrap por bloques conserva la
    dependencia que un bootstrap simple destruiria."""
    rng = np.random.default_rng(3)
    eps = rng.normal(0, 0.01, 3000)
    ar = np.empty_like(eps)
    ar[0] = eps[0]
    for t in range(1, len(eps)):
        ar[t] = 0.5 * ar[t - 1] + eps[t]
    r = _rets({"strategy": ar})
    blocks = montecarlo.bootstrap_paths(r, [21], n_paths=4000, mean_block=50, seed=1)
    iid = montecarlo.bootstrap_paths(r, [21], n_paths=4000, mean_block=1, seed=1)
    assert blocks.terminal[21][:, 0].std() > 1.3 * iid.terminal[21][:, 0].std()


def test_el_abanico_es_ordenado_y_empieza_en_uno():
    rng = np.random.default_rng(4)
    r = _rets({"strategy": rng.normal(0.0004, 0.01, 1000)})
    fan = montecarlo.bootstrap_paths(r, [252], n_paths=1000, mean_block=21, seed=2).fan
    assert (fan.loc[0] == 1.0).all()
    assert (fan["p5"] <= fan["p25"]).all() and (fan["p25"] <= fan["p50"]).all()
    assert (fan["p50"] <= fan["p75"]).all() and (fan["p75"] <= fan["p95"]).all()


def test_el_var_t_coincide_con_la_t_escalada_de_scipy():
    """La cartera simulada es una t con la desviacion de la covarianza estimada:
    su escala es sigma x raiz((dof - 2) / dof). Sin ese factor, el VaR t sale
    un 29% inflado con 5 grados de libertad."""
    stats = pytest.importorskip("scipy.stats")
    cov = _cov()
    w = pd.Series(0.2, index=cov.index)
    table, _ = montecarlo.parametric_portfolio_mc(
        w, cov, horizons_d=[1], confidences=[0.99], n_sims=300_000, t_dof=5, seed=5
    )
    sigma = float(np.sqrt(w @ cov @ w))
    expected = -stats.t.ppf(0.01, 5) * sigma * np.sqrt(3 / 5)
    t_var = table[table["distribution"] == "t_student"]["var"].iloc[0]
    assert t_var == pytest.approx(expected, rel=0.03)


def test_el_drawdown_se_mide_desde_el_ultimo_maximo():
    """Dias de +5% y -4.5% al azar: en un ano siempre hay un -4.5% despues de un
    maximo. Medido desde el capital inicial, media trayectoria saldria sin caida."""
    rng = np.random.default_rng(6)
    r = _rets({"strategy": rng.choice([0.05, -0.045], size=2000)})
    res = montecarlo.bootstrap_paths(r, [252], n_paths=2000, mean_block=1, seed=7)
    assert (res.max_drawdown[252] <= -0.045 + 1e-12).all()


def test_el_retorno_anual_mediano_se_anualiza_bien():
    r = _rets({"strategy": np.full(800, 0.0004)})
    res = montecarlo.bootstrap_paths(r, [252, 756], n_paths=200, mean_block=10, seed=1)
    summary = montecarlo.bootstrap_summary(res, [0.1])
    annual = 1.0004 ** 252 - 1
    assert summary.loc[252, "median_annualized"] == pytest.approx(annual)
    assert summary.loc[756, "median_annualized"] == pytest.approx(annual)
