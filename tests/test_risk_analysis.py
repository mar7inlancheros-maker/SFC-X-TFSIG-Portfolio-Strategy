"""Analisis de riesgo completo y reporte, sobre un mercado sintetico."""

from __future__ import annotations

import copy
import tomllib

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig import risk_report
from sfc_tfsig.config import ConfigError, config_from_dict, risk_config_from_dict
from sfc_tfsig.paths import RISK_CONFIG
from sfc_tfsig.risk import analysis

from conftest import BASE_CONFIG

N = 12


@pytest.fixture
def risk_cfg():
    with RISK_CONFIG.open("rb") as fh:
        data = tomllib.load(fh)
    # Menos trayectorias: el test mide la tuberia, no la precision.
    data["montecarlo"]["bootstrap"]["n_paths"] = 500
    data["montecarlo"]["parametric"]["n_sims"] = 5000
    return risk_config_from_dict(data)


@pytest.fixture
def mercado():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2016-01-01", periods=1400)
    market = rng.standard_t(4, len(dates)) * 0.007
    rets = pd.DataFrame(
        {f"T{i}": 0.0004 + (0.7 + 0.05 * i) * market + rng.normal(0, 0.012, len(dates))
         for i in range(N)},
        index=dates,
    )
    rets["SPY"] = market + 0.0003
    close = (1 + rets).cumprod() * 50.0
    volume = pd.DataFrame(2e6, index=dates, columns=close.columns)
    tickers = [f"T{i}" for i in range(N)]
    weights = pd.Series(0.98 / N, index=tickers)
    sectors = pd.Series(["Technology"] * 4 + ["Financials"] * 4 + ["Energy"] * 4, index=tickers)
    nav = (1 + rets[tickers].mean(axis=1)).cumprod() * 100_000.0
    bench = close["SPY"] / close["SPY"].iloc[0] * 100_000.0
    scores = pd.DataFrame({"score_momentum": rng.normal(1, 0.5, N),
                           "score_quality": rng.normal(0.3, 0.5, N)}, index=tickers)
    return {"close": close, "volume": volume, "weights": weights, "sectors": sectors,
            "nav": nav, "bench": bench, "scores": scores}


def _completo(m, risk_cfg):
    return analysis.analyze(
        m["weights"], m["sectors"], float(m["nav"].iloc[-1]), m["close"], "SPY", risk_cfg,
        strategy_nav=m["nav"], benchmark_nav=m["bench"], as_of=m["close"].index[-1],
        volume_wide=m["volume"], factor_scores=m["scores"],
    )


def test_el_analisis_completo_rellena_todas_las_piezas(mercado, risk_cfg):
    a = _completo(mercado, risk_cfg)
    assert a.status in ("OK", "ALERTA", "EXCEDIDO")
    assert a.ex_ante.volatility > 0
    assert set(a.strategy_var["method"]) == {"historico", "normal", "cornish_fisher", "fhs"}
    assert len(a.var_backtest) == 3
    assert list(a.bootstrap_summary.index) == [252, 756, 1260]
    assert len(a.hypothetical) == len(risk_cfg.get("stress.hypothetical"))
    assert len(a.historical) == len(risk_cfg.get("stress.historical"))
    # Todos los limites del TOML aparecen, y el drawdown ya tiene dato.
    assert set(a.limits["limit"]) == set(risk_cfg.section("limits")) - {"warning_fraction"}
    assert a.limits.set_index("limit").loc["max_current_drawdown", "status"] != "SIN DATO"


def test_la_contribucion_al_riesgo_suma_la_volatilidad(mercado, risk_cfg):
    a = _completo(mercado, risk_cfg)
    assert a.ex_ante.contributions["risk_share"].sum() == pytest.approx(1.0)
    assert a.ex_ante.contributions["contribution"].sum() == pytest.approx(a.ex_ante.volatility)


def test_el_var_del_limite_es_la_t_al_99_a_un_dia(mercado, risk_cfg):
    a = _completo(mercado, risk_cfg)
    row = a.book_var[(a.book_var["method"] == "mc_t_student")
                     & (a.book_var["confidence"] == 0.99) & (a.book_var["horizon_d"] == 1)]
    assert a.measures["var_99_1d"] == pytest.approx(float(row["var"].iloc[0]))


def test_la_comprobacion_previa_no_necesita_historia(mercado, risk_cfg):
    m = mercado
    a = analysis.analyze_book(m["weights"], m["sectors"], 100_000.0, m["close"], "SPY",
                              risk_cfg, as_of=m["close"].index[-1], volume_wide=m["volume"])
    assert a.var_backtest is None and a.bootstrap is None
    assert a.limits.set_index("limit").loc["max_current_drawdown", "status"] == "SIN DATO"
    text = risk_report.render_pre_trade(a)
    assert "Riesgo de la cartera objetivo" in text
    text.encode("ascii")


def test_el_reporte_es_ascii_y_trae_las_secciones(mercado, risk_cfg):
    a = _completo(mercado, risk_cfg)
    cfg = config_from_dict(BASE_CONFIG)
    text = risk_report.build_risk_report(a, cfg, risk_cfg)
    # La consola de Windows redirigida usa cp1252: nada fuera de ASCII.
    text.encode("ascii")
    for section in ("Semaforo de limites", "riesgo ex-ante", "VaR y Expected Shortfall",
                    "Backtest del VaR", "Monte Carlo", "Pruebas de estres",
                    "Liquidez y capacidad", "Limitaciones"):
        assert section in text
    assert risk_cfg.fingerprint in text and cfg.fingerprint in text


def test_un_sector_mal_escrito_se_rechaza(risk_cfg):
    data = copy.deepcopy(dict(risk_cfg.data))
    data["stress"]["hypothetical"][0]["sectors"] = {"Financial": -0.2}
    with pytest.raises(ConfigError, match="Financial"):
        analysis.validate_policy(risk_config_from_dict(data))


def test_la_cartera_del_backtest_usa_pesos_derivados():
    dates = pd.bdate_range("2024-01-01", periods=10)
    close = pd.DataFrame({"A": 10.0, "B": 10.0}, index=dates)
    close.loc[dates[-1], "A"] = 20.0  # A se duplica tras el rebalanceo
    holdings = pd.DataFrame({"date": [dates[0]] * 2, "ticker": ["A", "B"], "shares": [50.0, 50.0],
                             "price": [10.0, 10.0], "weight": [0.5, 0.5],
                             "sector": ["Technology", "Energy"]})
    nav = pd.Series(1000.0, index=dates)
    nav.iloc[-1] = 1500.0
    weights, sectors, nav_value, as_of = analysis.book_from_backtest(holdings, close, nav)
    assert weights["A"] == pytest.approx(1000.0 / 1500.0)
    assert weights["B"] == pytest.approx(500.0 / 1500.0)
    assert as_of == dates[-1] and nav_value == 1500.0
    assert sectors["A"] == "Technology"


def test_la_politica_de_riesgo_por_defecto_es_valida(risk_cfg):
    analysis.validate_policy(risk_cfg)
