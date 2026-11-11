"""Riesgo ex-ante: Ledoit-Wolf, descomposicion de Euler, concentracion y liquidez."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig.risk import exposure


def _retornos(n_days: int = 500, n_names: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, n_days)
    data = {f"T{i}": (0.8 + 0.1 * i) * market + rng.normal(0, 0.01, n_days) for i in range(n_names)}
    data["SPY"] = market
    return pd.DataFrame(data, index=pd.bdate_range("2022-01-03", periods=n_days))


def test_ledoit_wolf_es_semidefinida_y_contrae_entre_0_y_1():
    cov, delta = exposure.ledoit_wolf(_retornos(n_days=80, n_names=30))
    assert 0.0 <= delta <= 1.0
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > -1e-12
    # Con pocas observaciones por nombre, algo contrae.
    assert delta > 0.0


def test_ledoit_wolf_conserva_las_varianzas():
    rets = _retornos()
    cov, _ = exposure.ledoit_wolf(rets)
    sample = rets.cov(ddof=0)
    np.testing.assert_allclose(np.diag(cov), np.diag(sample), rtol=1e-10)


def test_la_contribucion_de_euler_suma_la_volatilidad():
    rets = _retornos()
    w = pd.Series(0.98 / 8, index=[f"T{i}" for i in range(8)])
    sectors = pd.Series(["Technology"] * 4 + ["Energy"] * 4, index=w.index)
    risk = exposure.ex_ante_risk(w, sectors, rets, "SPY")
    assert risk.contributions["contribution"].sum() == pytest.approx(risk.volatility)
    assert risk.contributions["risk_share"].sum() == pytest.approx(1.0)
    assert risk.sectors["risk_share"].sum() == pytest.approx(1.0)


def test_una_replica_del_benchmark_tiene_beta_uno_y_tracking_error_cero():
    rets = _retornos()
    rets["COPIA"] = rets["SPY"]
    risk = exposure.ex_ante_risk(pd.Series({"COPIA": 1.0}), pd.Series({"COPIA": "Technology"}),
                                 rets[["COPIA", "SPY"]], "SPY")
    assert risk.beta == pytest.approx(1.0, abs=1e-9)
    assert risk.tracking_error == pytest.approx(0.0, abs=1e-6)


def test_la_caja_reduce_el_riesgo_en_proporcion():
    rets = _retornos()
    names = [f"T{i}" for i in range(8)]
    sectors = pd.Series("Technology", index=names)
    full = exposure.ex_ante_risk(pd.Series(1 / 8, index=names), sectors, rets, "SPY")
    half = exposure.ex_ante_risk(pd.Series(0.5 / 8, index=names), sectors, rets, "SPY")
    assert half.volatility == pytest.approx(full.volatility / 2)


def test_sin_benchmark_el_error_dice_que_hacer():
    rets = _retornos().drop(columns="SPY")
    with pytest.raises(ValueError, match="Add it to the"):
        exposure.ex_ante_risk(pd.Series({"T0": 1.0}), pd.Series({"T0": "Energy"}), rets, "SPY")


def test_historia_corta_se_avisa():
    rets = _retornos()
    rets.loc[rets.index[:400], "T0"] = np.nan
    returns, coverage = exposure.trailing_returns((1 + rets.fillna(0)).cumprod().where(rets.notna()),
                                                  list(rets.columns), rets.index[-1], 252)
    risk = exposure.ex_ante_risk(pd.Series({"T0": 0.5, "T1": 0.5}),
                                 pd.Series({"T0": "Energy", "T1": "Energy"}),
                                 returns, "SPY", coverage=coverage, min_coverage=0.8)
    assert any("T0" in w for w in risk.warnings)


def test_numero_efectivo_de_nombres():
    w = pd.Series(0.05, index=[f"T{i}" for i in range(20)])
    conc = exposure.concentration(w, pd.Series("Energy", index=w.index))
    assert conc["effective_names"] == pytest.approx(20.0)
    assert conc["invested"] == pytest.approx(1.0)
    skewed = pd.Series([0.5] + [0.5 / 19] * 19, index=w.index)
    assert exposure.concentration(skewed, pd.Series("Energy", index=w.index))["effective_names"] < 5


def test_dias_de_liquidacion_y_capacidad():
    w = pd.Series({"A": 0.5, "B": 0.5})
    adv = pd.Series({"A": 1_000_000.0, "B": 10_000_000.0})
    table, summary = exposure.liquidity(w, 1_000_000.0, adv, participation=0.2, max_days=3)
    # A: 500k / (20% de 1M) = 2.5 sesiones.
    assert table.loc["A", "days_to_liquidate"] == pytest.approx(2.5)
    assert summary["max_days_name"] == "A"
    # Capacidad: 3 dias x 200k diarios / 50% del NAV = 1.2M.
    assert summary["capacity_nav"] == pytest.approx(1_200_000.0)


def test_el_var_por_componentes_suma_el_var_normal():
    rets = _retornos()
    w = pd.Series([0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.08], index=[f"T{i}" for i in range(8)])
    risk = exposure.ex_ante_risk(w, pd.Series("Energy", index=w.index), rets, "SPY", confidence=0.99)
    assert risk.contributions["component_var_1d"].sum() == pytest.approx(risk.var_normal_1d)


def test_la_beta_de_la_cartera_es_la_media_ponderada_de_las_betas():
    rets = _retornos()
    w = pd.Series([0.5, 0.3, 0.2], index=["T0", "T3", "T7"])
    risk = exposure.ex_ante_risk(w, pd.Series("Energy", index=w.index), rets, "SPY")
    assert risk.beta == pytest.approx(float((risk.betas * w).sum()))
    # T7 lleva 1.5 x mercado en los datos: su beta es la mayor.
    assert risk.betas.idxmax() == "T7"


def test_ledoit_wolf_bordes():
    cov, delta = exposure.ledoit_wolf(_retornos()[["T0"]])
    assert delta == 0.0 and cov.shape == (1, 1)
    with pytest.raises(ValueError, match="two observations"):
        exposure.ledoit_wolf(_retornos().iloc[:1])


def test_un_nombre_sin_variacion_no_rompe_la_contraccion():
    rets = _retornos()
    rets["PLANO"] = 0.0
    cov, delta = exposure.ledoit_wolf(rets)
    assert np.isfinite(cov.to_numpy()).all()
    assert cov.loc["PLANO"].abs().max() == pytest.approx(0.0)
    assert 0.0 <= delta <= 1.0


def test_una_cartera_toda_en_caja():
    conc = exposure.concentration(pd.Series({"A": 0.0}), pd.Series({"A": "Energy"}))
    assert conc["cash"] == 1.0 and conc["n_names"] == 0


def test_exposicion_a_factores_frente_al_universo():
    w = pd.Series({"A": 0.5, "B": 0.5})
    holdings = pd.DataFrame({"score_momentum": [2.0, 1.0], "score_value": [np.nan, -1.0]},
                            index=["A", "B"])
    universe = pd.DataFrame({"score_momentum": [0.0, 0.5, -0.5], "score_value": [0.0, 1.0, -1.0]})
    fx = exposure.factor_exposure(w, holdings, universe)
    assert fx.loc["momentum", "portfolio"] == pytest.approx(1.5)
    assert fx.loc["momentum", "active"] == pytest.approx(1.5)
    # Sin dato de valor para A: se pondera con lo que hay y se dice cuanto peso cubre.
    assert fx.loc["value", "portfolio"] == pytest.approx(-1.0)
    assert fx.loc["value", "weight_covered"] == pytest.approx(0.5)


def test_liquidez_sin_volumen_no_inventa_dias():
    w = pd.Series({"A": 1.0})
    _, summary = exposure.liquidity(w, 1e6, pd.Series({"A": np.nan}), participation=0.2, max_days=3)
    assert np.isnan(summary["max_days"])
    assert summary["weight_without_adv"] == pytest.approx(1.0)


def test_las_betas_recuperan_la_sensibilidad_construida():
    """T0 se mueve 0.8 veces el mercado y T7 1.5 veces: las betas tienen que
    salir cerca, no solo ser coherentes entre si."""
    rets = _retornos(n_days=2000)
    names = [f"T{i}" for i in range(8)]
    risk = exposure.ex_ante_risk(pd.Series(1 / 8, index=names), pd.Series("Energy", index=names),
                                 rets, "SPY")
    assert risk.betas["T0"] == pytest.approx(0.8, abs=0.1)
    assert risk.betas["T7"] == pytest.approx(1.5, abs=0.1)
    solo = exposure.ex_ante_risk(pd.Series({"T7": 1.0}), pd.Series({"T7": "Energy"}),
                                 rets[["T7", "SPY"]], "SPY")
    assert solo.beta == pytest.approx(1.5, abs=0.1)


def test_la_contraccion_se_queda_en_1_cuando_el_objetivo_ya_es_la_verdad():
    """Datos con correlacion constante de verdad: la intensidad optima pasa de
    1 y se corta. Sin el corte, la mezcla extrapolaria mas alla del objetivo."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(0, 0.01, (200, 1)) + rng.normal(0, 0.01, (200, 10)))
    cov, delta = exposure.ledoit_wolf(x)
    assert delta == 1.0
    sd = np.sqrt(np.diag(cov))
    corr = cov.to_numpy() / np.outer(sd, sd)
    off = corr[~np.eye(10, dtype=bool)]
    assert off.max() - off.min() < 1e-12


def test_el_numero_efectivo_no_cuenta_la_caja():
    """Diez nombres iguales con la mitad en caja siguen siendo diez apuestas."""
    w = pd.Series(0.05, index=[f"T{i}" for i in range(10)])
    conc = exposure.concentration(w, pd.Series("Energy", index=w.index))
    assert conc["effective_names"] == pytest.approx(10.0)
    assert conc["cash"] == pytest.approx(0.5)


def test_ratio_de_diversificacion_de_nombres_independientes():
    """N nombres independientes con la misma volatilidad: DR = raiz(N)."""
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.01, (5000, 9)), columns=[f"T{i}" for i in range(9)])
    rets["SPY"] = rng.normal(0, 0.01, 5000)
    w = pd.Series(1 / 9, index=[f"T{i}" for i in range(9)])
    risk = exposure.ex_ante_risk(w, pd.Series("Energy", index=w.index), rets, "SPY")
    assert risk.diversification_ratio == pytest.approx(3.0, rel=0.05)
    solo = exposure.ex_ante_risk(pd.Series({"T0": 1.0}), pd.Series({"T0": "Energy"}),
                                 rets[["T0", "SPY"]], "SPY")
    assert solo.diversification_ratio == pytest.approx(1.0)


def test_la_ventana_de_covarianza_tiene_exactamente_window_retornos():
    rets = _retornos()
    close = (1 + rets).cumprod() * 100
    returns, coverage = exposure.trailing_returns(close, ["T0", "SPY"], close.index[-1], 252)
    assert len(returns) == 252
    assert returns.index[-1] == close.index[-1]
    assert (coverage == 1.0).all()


def test_el_volumen_diario_es_la_mediana_y_no_se_deja_llevar_por_un_pico():
    dates = pd.bdate_range("2024-01-01", periods=63)
    close = pd.DataFrame({"A": 10.0}, index=dates)
    volume = pd.DataFrame({"A": 100_000.0}, index=dates)
    volume.iloc[-1] = 100_000_000.0   # inclusion en un indice: un dia raro
    adv = exposure.average_dollar_volume(close, volume, dates[-1], 63)
    assert adv["A"] == pytest.approx(1_000_000.0)
