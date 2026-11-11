"""Riesgo movil y calibracion del modelo de riesgo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig.risk import history


def _mercado(n_days: int = 900, n_names: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    market = rng.normal(0.0003, 0.01, n_days)
    rets = pd.DataFrame(market[:, None] + rng.normal(0, 0.01, (n_days, n_names)),
                        index=dates, columns=[f"T{i}" for i in range(n_names)])
    close = (1 + rets).cumprod() * 100.0
    return dates, rets, close, market


def test_riesgo_movil_de_una_copia_del_benchmark():
    dates, _, _, market = _mercado()
    bench = pd.Series((1 + market).cumprod() * 100, index=dates)
    rolling = history.rolling_risk(bench, bench, 63).dropna()
    assert rolling["beta"].iloc[-1] == pytest.approx(1.0)
    assert rolling["tracking_error"].iloc[-1] == pytest.approx(0.0, abs=1e-12)


def test_el_resumen_situa_el_valor_de_hoy():
    summary = history.rolling_summary(pd.DataFrame({"vol": np.linspace(0.1, 0.3, 100)}))
    assert summary.loc["vol", "current"] == pytest.approx(0.3)
    assert summary.loc["vol", "percentile"] == pytest.approx(1.0)


def _backtest_equiponderado(rets: pd.DataFrame, scale: float = 1.0):
    """Rebalanceo mensual 1/N; `scale` amplifica el NAV para romper la calibracion."""
    month_starts = rets.index[rets.index.to_series().dt.month.diff() != 0][12:]
    names = list(rets.columns)
    holdings = pd.concat([
        pd.DataFrame({"date": d, "ticker": names, "weight": 1.0 / len(names)})
        for d in month_starts
    ])
    nav = (1 + rets.mean(axis=1) * scale).cumprod() * 1e5
    return holdings, nav


def test_un_modelo_bien_especificado_sale_calibrado():
    _, rets, close, _ = _mercado(n_days=1500)
    holdings, nav = _backtest_equiponderado(rets)
    table, summary = history.risk_model_calibration(holdings, close, nav)
    assert summary["n_periods"] > 40
    assert summary["ci_low"] - 0.15 < summary["bias_statistic"] < summary["ci_high"] + 0.15
    assert (table["predicted_period_vol"] > 0).all()


def test_si_la_cartera_real_oscila_el_doble_el_modelo_subestima():
    _, rets, close, _ = _mercado(n_days=1500)
    holdings, nav = _backtest_equiponderado(rets, scale=2.0)
    _, summary = history.risk_model_calibration(holdings, close, nav)
    assert summary["bias_statistic"] > summary["ci_high"]
    assert summary["verdict"] == "understates risk"


def test_la_prevision_no_usa_precios_posteriores_al_rebalanceo():
    _, rets, close, _ = _mercado(n_days=800)
    holdings, nav = _backtest_equiponderado(rets)
    table_a, _ = history.risk_model_calibration(holdings, close, nav)
    altered = close.copy()
    altered.iloc[-30:] = altered.iloc[-30:].mul(np.linspace(1, 3, 30), axis=0)
    table_b, _ = history.risk_model_calibration(holdings, altered, nav)
    # Solo pueden cambiar las previsiones cuya ventana toca el tramo alterado.
    pd.testing.assert_series_equal(table_a["predicted_period_vol"].iloc[:-2],
                                   table_b["predicted_period_vol"].iloc[:-2])


def test_calibracion_sin_posiciones_devuelve_vacio():
    _, _, close, _ = _mercado()
    table, summary = history.risk_model_calibration(pd.DataFrame(), close, pd.Series(dtype=float))
    assert table.empty and summary == {}


def test_rebalanceos_con_pocos_nombres_modelables_se_saltan():
    _, rets, close, _ = _mercado(n_days=600)
    holdings, nav = _backtest_equiponderado(rets)
    # Diez nombres en cartera y un minimo de once: ninguna prevision es fiable.
    table, summary = history.risk_model_calibration(holdings, close, nav, min_names=11)
    assert table.empty and summary == {}
    # Seis nombres sin historia hasta el final: solo cuatro son modelables, y
    # la tabla dice que fraccion del peso se pudo modelar.
    young = close.copy()
    young.iloc[:-30, :6] = np.nan
    table, _ = history.risk_model_calibration(holdings, young, nav, min_names=3)
    assert not table.empty
    assert table["weight_modelled"].min() == pytest.approx(0.4)


def test_riesgo_movil_sin_benchmark_solo_trae_volatilidad():
    dates, _, _, market = _mercado()
    nav = pd.Series((1 + market).cumprod(), index=dates)
    assert list(history.rolling_risk(nav, None, 63).columns) == ["vol"]


def test_el_sesgo_es_la_desviacion_tipica_de_z():
    _, rets, close, _ = _mercado(n_days=1500)
    holdings, nav = _backtest_equiponderado(rets)
    table, summary = history.risk_model_calibration(holdings, close, nav)
    assert summary["bias_statistic"] == pytest.approx(table["z"].std(ddof=1), rel=1e-12)
    assert summary["ci_high"] - 1.0 == pytest.approx(1.96 * np.sqrt(1 / (2 * len(table))))


def test_beta_movil_de_una_cartera_apalancada():
    """Doble del benchmark dia a dia: beta 2, no 0.5."""
    dates, _, _, market = _mercado()
    bench = pd.Series((1 + market).cumprod() * 100, index=dates)
    double = pd.Series((1 + 2 * market).cumprod() * 100, index=dates)
    rolling = history.rolling_risk(double, bench, 63).dropna()
    assert rolling["beta"].iloc[-1] == pytest.approx(2.0)
