"""Motor long/short: matematica por activo y estadistica, con resultados conocidos."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant_engine.data.cleaning import annualize_return, daily_rf
from quant_engine.factors import beta as beta_mod
from quant_engine.factors import mean_reversion, momentum, performance, volatility
from quant_engine.inference import correlation, covariance, tests


def _days(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


# ---------------------------------------------------------------------------
#  Anualizacion y rendimiento
# ---------------------------------------------------------------------------


def test_tasa_diaria_es_por_composicion_no_division():
    d = daily_rf(0.04)
    assert (1 + d) ** 252 == pytest.approx(1.04)
    assert d != pytest.approx(0.04 / 252, rel=1e-6)


def test_anualizar_duplicar_en_cinco_anos():
    assert annualize_return(1.0, 5) == pytest.approx(2 ** 0.2 - 1)


def test_sharpe_con_retorno_constante_igual_a_rf_no_es_positivo():
    r = pd.Series(daily_rf(0.04), index=_days(300)) + np.random.default_rng(0).normal(0, 1e-9, 300)
    out = performance.risk_adjusted(r, 0.04)
    assert abs(out["sharpe"]) < 3  # ruido numerico, no un Sharpe real


def test_volatilidad_a_la_baja_promedia_sobre_todas_las_sesiones():
    """Semidesviacion respecto al objetivo: divide entre N, no entre los negativos."""
    r = pd.Series([0.01, -0.01, 0.01, -0.01] * 50, index=_days(200))
    out = performance.risk_metrics(r, 0.0)
    esperado = math.sqrt((0.01 ** 2) / 2) * math.sqrt(252)
    assert out["downside_vol"] == pytest.approx(esperado, rel=1e-6)


def test_drawdown_maximo_y_recuperacion_de_una_serie_conocida():
    r = pd.Series([0.10, -0.50, 0.0, 1.20, 0.0], index=_days(5))
    out = performance.drawdown_stats(r)
    assert out["max_dd"] == pytest.approx(-0.50)
    # 1.1 -> 0.55 -> 0.55 -> 1.21 (recupera en la sesion 3 desde el minimo)
    assert out["recovery_d"] == 2


def test_drawdown_sin_recuperar_da_nan():
    r = pd.Series([0.10, -0.30, 0.05], index=_days(3))
    assert np.isnan(performance.drawdown_stats(r)["recovery_d"])


def test_retornos_de_ventana():
    close = pd.Series(np.linspace(100, 200, 300), index=_days(300))
    tr = performance.trailing_returns(close)
    assert tr["1D"] == pytest.approx(close.iloc[-1] / close.iloc[-2] - 1)
    assert tr["1Y"] == pytest.approx(close.iloc[-1] / close.iloc[-253] - 1)


# ---------------------------------------------------------------------------
#  Beta y alfa
# ---------------------------------------------------------------------------


def test_beta_recupera_la_pendiente_verdadera():
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(0.0004, 0.01, 1000), index=_days(1000))
    y = 1.5 * m + pd.Series(rng.normal(0, 0.005, 1000), index=m.index)
    out = beta_mod.regression(y, m, 0.0)
    assert out["beta"] == pytest.approx(1.5, abs=0.05)
    assert abs(out["alpha_t"]) < 3


def test_alfa_no_absorbe_la_tasa_libre_de_riesgo():
    """Sin restar rf, un activo de beta 0,5 mostraria (1 - 0,5) x rf de alfa falso."""
    rng = np.random.default_rng(2)
    rf_d = daily_rf(0.05)
    m_ex = pd.Series(rng.normal(0.0003, 0.01, 2000), index=_days(2000))
    m = m_ex + rf_d
    # Ruido pequeno: el error estandar del alfa anual queda en ~0,3%.
    y = rf_d + 0.5 * m_ex + pd.Series(rng.normal(0, 0.0005, 2000), index=m.index)
    out = beta_mod.regression(y, m, 0.05)
    # Sin restar rf saldria cerca de (1 - 0,5) x 5% = 2,5%.
    assert abs(out["alpha_annual"]) < 0.01


def test_beta_movil_de_un_activo_identico_al_mercado_es_uno():
    m = pd.Series(np.random.default_rng(3).normal(0, 0.01, 400), index=_days(400))
    assert beta_mod.rolling_beta(m, m, 60).dropna().iloc[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
#  Momentum, reversion, volatilidad
# ---------------------------------------------------------------------------


def test_rsi_de_una_serie_que_solo_sube_es_cien():
    """Antes salia NaN: perdida media cero anulaba la division."""
    close = pd.Series(np.arange(1, 60, dtype=float), index=_days(59))
    assert momentum.rsi(close).iloc[-1] == pytest.approx(100.0)


def test_rsi_de_una_serie_plana_es_cincuenta():
    close = pd.Series(100.0, index=_days(59))
    assert momentum.rsi(close).iloc[-1] == pytest.approx(50.0)


def test_momentum_12_1_salta_el_ultimo_mes():
    s = pd.Series(100.0, index=_days(300))
    s.iloc[-21:] = 200.0  # salto solo en el ultimo mes
    f = momentum.momentum_features(s)
    assert f["mom_12_1"] == pytest.approx(0.0)
    assert f["mom_1m"] > 0.9


def test_calidad_de_tendencia_lineal_es_uno():
    close = pd.Series(np.exp(np.linspace(0, 1, 300)), index=_days(300))
    assert momentum.trend_quality(close) == pytest.approx(1.0, abs=1e-9)


def test_ratio_de_varianzas_de_un_paseo_aleatorio_ronda_uno():
    r = pd.Series(np.random.default_rng(4).normal(0, 0.01, 5000), index=_days(5000))
    assert mean_reversion.variance_ratio(r, 5, window=5000) == pytest.approx(1.0, abs=0.1)


def test_clasificacion_de_regimen_sigue_las_reglas_declaradas():
    assert mean_reversion.classify_regime(
        {"zscore_20d": 2.0, "variance_ratio_5": 0.8, "vol_adj_momentum": 0.1, "above_sma200": 1.0}, 0.5, 1.5
    ) == "MEAN REVERSION"
    # Estirado pero persistente: no es reversion.
    assert mean_reversion.classify_regime(
        {"zscore_20d": 2.0, "variance_ratio_5": 1.2, "vol_adj_momentum": 1.0, "above_sma200": 1.0}, 0.5, 1.5
    ) == "MOMENTUM UP"
    # Momentum contra el lado de la SMA200: neutral.
    assert mean_reversion.classify_regime(
        {"zscore_20d": 0.0, "variance_ratio_5": 1.0, "vol_adj_momentum": 1.0, "above_sma200": -1.0}, 0.5, 1.5
    ) == "NEUTRAL"


def test_regimen_de_volatilidad_por_percentil_propio():
    rng = np.random.default_rng(5)
    calm = rng.normal(0, 0.005, 600)
    wild = rng.normal(0, 0.05, 30)
    r = pd.Series(np.concatenate([calm, wild]), index=_days(630))
    close = 100 * (1 + r).cumprod()
    out = volatility.volatility_features(r, close, None, None, lam=0.94,
                                         thresholds={"low": 0.2, "high": 0.8, "extreme": 0.95})
    assert out["vol_regime"] == "EXTREME"


# ---------------------------------------------------------------------------
#  Covarianza y correlacion
# ---------------------------------------------------------------------------


def _returns(n=500, k=4, seed=6):
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, (n, 1))
    return pd.DataFrame(common + rng.normal(0, 0.01, (n, k)), index=_days(n), columns=list("ABCD")[:k])


def test_ewma_con_lambda_cercano_a_uno_se_acerca_a_la_muestral():
    r = _returns()
    e = covariance.ewma(r, 0.9999)
    s = covariance.sample(r)
    assert np.allclose(e.to_numpy(), s.to_numpy(), rtol=0.05)


def test_estimadores_son_semidefinidos_positivos():
    for name, v in covariance.estimate_all(_returns(), 0.94).items():
        assert v["psd"], name


def test_numero_de_condicion_detecta_activos_casi_identicos():
    r = _returns()
    r["D"] = r["A"] + np.random.default_rng(7).normal(0, 1e-7, len(r))
    assert covariance.condition_number(covariance.sample(r)) > covariance.ILL_CONDITIONED
    _, name = covariance.for_optimization(r)
    assert name == "ledoit_wolf_cc"


def test_clusters_agrupan_lo_muy_correlacionado():
    corr = pd.DataFrame([[1, .9, .1], [.9, 1, .1], [.1, .1, 1]], index=list("ABC"), columns=list("ABC"))
    groups = [sorted(g) for g in correlation.clusters(corr, 0.6).values()]
    assert ["A", "B"] in groups and ["C"] in groups


# ---------------------------------------------------------------------------
#  Contrastes
# ---------------------------------------------------------------------------


def test_permutacion_exacta_con_separacion_perfecta():
    """5 contra 5 totalmente separados: solo 2 de 252 reparticiones igualan el extremo."""
    p = tests.permutation_p(np.array([10, 11, 12, 13, 14.]), np.array([1, 2, 3, 4, 5.]))
    assert p == pytest.approx(2 / 252)


def test_bootstrap_contiene_la_diferencia_verdadera():
    rng = np.random.default_rng(8)
    a, b = rng.normal(1.0, 0.5, 200), rng.normal(0.0, 0.5, 200)
    lo, hi = tests.bootstrap_ci(a, b, seed=1)
    assert lo < 1.0 < hi


def test_spread_con_senal_es_significativo():
    idx = _days(1000)
    rng = np.random.default_rng(9)
    long_r = pd.Series(rng.normal(0.001, 0.01, 1000), index=idx)
    short_r = pd.Series(rng.normal(-0.001, 0.01, 1000), index=idx)
    out = tests.spread_test(long_r, short_r)
    assert out["p_value"] < 0.01
