"""GARCH(1,1) y Monte Carlo filtrado sobre series simuladas con parametros conocidos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig.risk import garch


def _garch_series(alpha: float, beta: float, long_run: float, n: int = 4000, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    variance, out = long_run, np.empty(n)
    for t in range(n):
        out[t] = np.sqrt(variance) * rng.standard_normal()
        variance = (1 - alpha - beta) * long_run + alpha * out[t] ** 2 + beta * variance
    return pd.Series(out, index=pd.bdate_range("2010-01-01", periods=n))


def test_recupera_los_parametros_de_un_garch_conocido():
    fit = garch.fit_garch(_garch_series(0.08, 0.90, 1e-4))
    assert fit.alpha == pytest.approx(0.08, abs=0.03)
    assert fit.beta == pytest.approx(0.90, abs=0.04)
    assert fit.persistence < 1.0
    assert fit.long_run_vol == pytest.approx(np.sqrt(1e-4 * 252), rel=0.15)


def test_la_prevision_de_cada_dia_no_ve_el_retorno_de_ese_dia():
    shocked = _garch_series(0.08, 0.90, 1e-4, n=600)
    shocked.iloc[-1] = -0.25
    fit = garch.fit_garch(shocked)
    # El choque del ultimo dia no puede estar en su propia prevision, solo en
    # la del siguiente.
    assert fit.sigma_next > 2 * fit.sigma.iloc[-1]


def test_la_volatilidad_revierte_a_su_media_con_el_horizonte():
    r = _garch_series(0.08, 0.90, 1e-4)
    calm = pd.Series(np.full(40, 1e-4), index=pd.bdate_range(r.index[-1], periods=41)[1:])
    fit = garch.fit_garch(pd.concat([r, calm]))
    terms = garch.vol_term_structure(fit, [1, 21, 252, 2520])["vol_annual"]
    # Tras un periodo de calma, la volatilidad prevista sube hacia la de largo plazo.
    assert terms[1] < terms[21] < terms[252] < fit.long_run_vol
    assert terms[2520] == pytest.approx(fit.long_run_vol, rel=0.05)


def test_la_simulacion_filtrada_es_reproducible_y_ordenada():
    fit = garch.fit_garch(_garch_series(0.08, 0.90, 1e-4))
    a = garch.simulate_filtered(fit, [1, 21], n_paths=5000, seed=3)
    b = garch.simulate_filtered(fit, [1, 21], n_paths=5000, seed=3)
    np.testing.assert_array_equal(a.terminal[21], b.terminal[21])
    table = garch.simulation_var_table(a, [0.95, 0.99]).set_index(["horizon_d", "confidence"])
    assert table.loc[(1, 0.99), "var"] > table.loc[(1, 0.95), "var"]
    assert table.loc[(21, 0.99), "var"] > table.loc[(1, 0.99), "var"]
    assert (table["es"] >= table["var"]).all()


def test_con_volatilidad_alta_hoy_la_prevision_sube():
    base = _garch_series(0.08, 0.90, 1e-4, n=2000, seed=5)
    storm = pd.Series(np.random.default_rng(6).normal(0, 0.04, 15),
                      index=pd.bdate_range(base.index[-1], periods=16)[1:])
    calm_fit = garch.fit_garch(base)
    storm_fit = garch.fit_garch(pd.concat([base, storm]))
    assert storm_fit.current_vol > 1.5 * calm_fit.current_vol


def test_historia_corta_se_rechaza_con_mensaje():
    with pytest.raises(ValueError, match="al menos"):
        garch.fit_garch(pd.Series(np.zeros(100)))


def _loglik(r: pd.Series, alpha: float, beta: float) -> float:
    """Verosimilitud gaussiana con objetivo de varianza, escrita aparte del modulo."""
    eps = r.to_numpy() - r.mean()
    long_run = np.var(eps, ddof=1)
    variance, total = long_run, 0.0
    for e in eps:
        total += -0.5 * (np.log(2 * np.pi * variance) + e ** 2 / variance)
        variance = (1 - alpha - beta) * long_run + alpha * e ** 2 + beta * variance
    return total


def test_el_ajuste_es_un_maximo_local_de_la_verosimilitud():
    r = _garch_series(0.08, 0.90, 1e-4, n=1500, seed=11)
    fit = garch.fit_garch(r)
    best = _loglik(r, fit.alpha, fit.beta)
    assert best == pytest.approx(fit.loglik, rel=1e-9)
    for da, db in ((0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.03)):
        a, b = fit.alpha + da, fit.beta + db
        if a > 0 and a + b < 0.999:
            assert _loglik(r, a, b) < best


def test_la_vida_media_y_la_volatilidad_de_largo_plazo():
    fit = garch.fit_garch(_garch_series(0.05, 0.93, 1e-4))
    assert fit.half_life_d == pytest.approx(np.log(0.5) / np.log(fit.persistence))
    assert fit.long_run_vol == pytest.approx(np.sqrt(fit.long_run_var * 252))


def test_con_un_cambio_de_regimen_el_ajuste_sigue_siendo_estacionario():
    """Volatilidad diez veces mayor en la segunda mitad: la verosimilitud empuja
    la persistencia hacia 1. Por encima, la varianza de largo plazo no existe y
    omega saldria negativo."""
    rng = np.random.default_rng(0)
    r = pd.Series(np.concatenate([rng.normal(0, 0.003, 1500), rng.normal(0, 0.03, 1500)]))
    fit = garch.fit_garch(r)
    assert fit.persistence < 0.999
    assert fit.omega > 0


def test_estructura_temporal_exacta():
    """A un dia es la volatilidad de hoy; a dos, la media de hoy y de manana."""
    fit = garch.fit_garch(_garch_series(0.08, 0.90, 1e-4, n=2000, seed=2))
    terms = garch.vol_term_structure(fit, [1, 2])["vol_annual"]
    assert terms[1] == pytest.approx(fit.current_vol, rel=1e-12)
    first, long_run, p = fit.sigma_next ** 2, fit.long_run_var, fit.persistence
    two_day = (first + long_run + p * (first - long_run)) / 2
    assert terms[2] == pytest.approx(np.sqrt(two_day * 252), rel=1e-12)


def test_la_simulacion_tiene_racimos_de_volatilidad():
    """Con residuos normales, los racimos son lo unico que engorda la cola a 21
    dias. Sin ellos, la suma de 21 normales con varianza fija seria normal."""
    rng = np.random.default_rng(3)
    fit = garch.GarchFit(
        mu=0.0, omega=1e-4 * (1 - 0.15 - 0.80), alpha=0.15, beta=0.80, long_run_var=1e-4,
        loglik=0.0, sigma=pd.Series([0.01]), sigma_next=0.01,
        residuals=rng.standard_normal(20_000),
    )
    sim = garch.simulate_filtered(fit, [21], n_paths=20_000, seed=4)
    x = sim.terminal[21]
    excess_kurtosis = np.mean((x - x.mean()) ** 4) / x.var() ** 2 - 3
    assert excess_kurtosis > 0.5
