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
    with pytest.raises(ValueError, match="el doble de historia"):
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
