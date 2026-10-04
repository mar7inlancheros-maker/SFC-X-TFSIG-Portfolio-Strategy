"""Analisis de riesgo completo y reporte, sobre un mercado sintetico."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from conftest import BASE_CONFIG
from risk_helpers import light_risk_config, synthetic_book

from sfc_tfsig import risk_report
from sfc_tfsig.config import ConfigError, config_from_dict, risk_config_from_dict
from sfc_tfsig.risk import analysis

N = 12


@pytest.fixture(scope="module")
def risk_cfg():
    return light_risk_config()


@pytest.fixture(scope="module")
def mercado():
    return synthetic_book(n_names=N)


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


def _panel(m):
    rng = np.random.default_rng(9)
    dates = pd.date_range(m["close"].index[0], m["close"].index[-1], freq="ME")
    tickers = list(m["close"].columns.drop("SPY"))
    extra = [f"U{i}" for i in range(60)]
    sectors = dict(m["sectors"])
    sectors.update({t: ["Technology", "Financials", "Energy"][i % 3] for i, t in enumerate(extra)})
    return pd.concat([
        pd.DataFrame({"date": d, "ticker": tickers + extra,
                      "sector": [sectors[t] for t in tickers + extra],
                      "score_momentum": rng.normal(0, 1, len(tickers) + len(extra)),
                      "market_cap": np.exp(rng.normal(22, 1, len(tickers) + len(extra)))})
        for d in dates
    ])


def _con_universo(m):
    """Mercado con 60 nombres mas, para que la regresion transversal tenga seccion."""
    rng = np.random.default_rng(11)
    base = m["close"]["SPY"].pct_change().fillna(0).to_numpy()
    extra = pd.DataFrame({f"U{i}": (1 + base + rng.normal(0, 0.01, len(base))).cumprod() * 20
                          for i in range(60)}, index=m["close"].index)
    return pd.concat([m["close"], extra], axis=1)


def test_el_analisis_ampliado_rellena_garch_factores_estres_y_calibracion(mercado, risk_cfg):
    m = mercado
    close = _con_universo(m)
    holdings = pd.concat([
        pd.DataFrame({"date": d, "ticker": list(m["weights"].index), "weight": m["weights"].to_numpy()})
        for d in m["close"].index[300::21]
    ])
    a = analysis.analyze(
        m["weights"], m["sectors"], float(m["nav"].iloc[-1]), close, "SPY", risk_cfg,
        strategy_nav=m["nav"], benchmark_nav=m["bench"], as_of=m["close"].index[-1],
        volume_wide=m["volume"], factor_scores=m["scores"], panel=_panel(m),
        holdings=holdings, cost_bps=20.0, turnover=0.2, rebalances_per_year=12,
    )
    assert set(a.garch_fits) == {"cartera", "estrategia"}
    assert set(a.conditional_var["series"]) == {"cartera", "estrategia"}
    assert a.factor_risk is not None and a.factor_risk.total_vol > 0
    assert a.stressed and a.stressed["stressed_var"] >= a.stressed["current_var"]
    assert len(a.covariance_stress) == 1 + len(risk_cfg.get("stress.covariance"))
    assert a.calibration_summary["n_periods"] > 10
    assert a.capacity is not None and np.isfinite(a.capacity_nav)
    assert a.liquidation.loc["crisis", "cost"] > a.liquidation.loc["normal", "cost"]
    checks = a.limits.set_index("limit")
    assert checks.loc["max_stressed_var_99_1d", "status"] != "SIN DATO"
    assert checks.loc["max_liquidation_cost", "status"] != "SIN DATO"

    text = risk_report.build_risk_report(a, config_from_dict(BASE_CONFIG), risk_cfg)
    text.encode("ascii")
    for section in ("Modelo factorial", "Volatilidad condicional", "VaR estresado",
                    "Choques de volatilidad", "Calibracion", "Capacidad"):
        assert section in text


# ---------------------------------------------------------------------------
#  Ramas: errores con mensaje, datos que faltan y avisos
# ---------------------------------------------------------------------------


def _libro(m, risk_cfg, close=None, **kwargs):
    return analysis.analyze_book(
        m["weights"], m["sectors"], 100_000.0, m["close"] if close is None else close, "SPY",
        risk_cfg, as_of=m["close"].index[-1], **kwargs,
    )


def test_la_cartera_objetivo_se_convierte_en_pesos_y_sectores():
    target = pd.DataFrame({"ticker": ["A", "B"], "weight": [0.6, 0.38],
                           "sector": ["Energy", None], "score_composite": [1.0, 0.5]})
    weights, sectors = analysis.book_from_target(target)
    assert weights.to_dict() == {"A": 0.6, "B": 0.38}
    assert sectors.to_dict() == {"A": "Energy", "B": "Unknown"}
    no_sector = analysis.book_from_target(target.drop(columns="sector"))[1]
    assert set(no_sector) == {"Unknown"}


def test_sin_posiciones_el_error_lo_dice():
    with pytest.raises(ValueError, match="no dejo posiciones"):
        analysis.book_from_backtest(pd.DataFrame(), pd.DataFrame(), pd.Series([1.0]))


def test_sin_precios_del_benchmark_el_error_dice_que_hacer(mercado, risk_cfg):
    close = mercado["close"].copy()
    close["SPY"] = np.nan
    with pytest.raises(ValueError, match="anadelo"):
        _libro(mercado, risk_cfg, close=close)


def test_sin_volumen_ni_coste_se_avisa_y_la_liquidez_queda_sin_dato(mercado, risk_cfg):
    a = _libro(mercado, risk_cfg)
    assert any("sin volumen" in w for w in a.warnings)
    assert any("sin coste fijo" in w for w in a.warnings)
    assert a.liquidation is None
    assert a.limits.set_index("limit").loc["max_liquidation_days", "status"] == "SIN DATO"


def test_un_peso_cero_no_entra_en_el_analisis(mercado, risk_cfg):
    weights = mercado["weights"].copy()
    weights.iloc[0] = 0.0
    a = analysis.analyze_book(weights, mercado["sectors"], 100_000.0, mercado["close"], "SPY",
                              risk_cfg, as_of=mercado["close"].index[-1])
    assert weights.index[0] not in a.weights.index
    assert weights.index[0] not in a.ex_ante.contributions.index


def test_con_historia_corta_garch_y_var_estresado_avisan_sin_romper(mercado, risk_cfg):
    short = mercado["close"].tail(200)
    a = analysis.analyze_book(mercado["weights"], mercado["sectors"], 100_000.0, short, "SPY",
                              risk_cfg, as_of=short.index[-1])
    assert any("GARCH" in w for w in a.warnings)
    assert any("VaR estresado" in w for w in a.warnings)
    assert a.stressed is None and "cartera" not in a.garch_fits
    assert a.limits.set_index("limit").loc["max_stressed_var_99_1d", "status"] == "SIN DATO"


def test_la_historia_larga_del_estres_sustituye_a_la_del_panel(mercado, risk_cfg):
    """`stress_close` trae anos anteriores: el VaR estresado y el GARCH de la
    cartera tienen que usarlos, no la historia corta del panel."""
    m = mercado
    rng = np.random.default_rng(21)
    early = pd.bdate_range(end=m["close"].index[0] - pd.Timedelta(days=1), periods=500)
    # Antes del panel hubo una crisis de mercado: un choque comun del 4% diario.
    # Comun, no independiente: el ruido propio de doce nombres se diversifica.
    common = rng.normal(0, 0.04, (len(early), 1))
    early_rets = pd.DataFrame(common + rng.normal(0, 0.005, (len(early), len(m["close"].columns))),
                              index=early, columns=m["close"].columns)
    early_close = (1 + early_rets).cumprod() * 10.0
    scale = m["close"].iloc[0] / early_close.iloc[-1]
    long_close = pd.concat([early_close * scale, m["close"].iloc[1:]])

    base = _completo(m, risk_cfg)
    with_long = analysis.analyze(
        m["weights"], m["sectors"], float(m["nav"].iloc[-1]), m["close"], "SPY", risk_cfg,
        strategy_nav=m["nav"], benchmark_nav=m["bench"], as_of=m["close"].index[-1],
        volume_wide=m["volume"], factor_scores=m["scores"], stress_close=long_close,
    )
    assert with_long.stressed["history_start"] < base.stressed["history_start"]
    assert with_long.stressed["window_start"] < m["close"].index[0]
    assert with_long.stressed["stressed_var"] > 2 * base.stressed["stressed_var"]
    assert (with_long.conditional_var["series"] == "cartera").sum() == (
        base.conditional_var["series"] == "cartera").sum()


def test_si_el_modelo_factorial_ve_mas_riesgo_se_avisa(mercado, risk_cfg, monkeypatch):
    from sfc_tfsig.risk import factor_model

    def stub(weights, *args, **kwargs):
        return factor_model.FactorRisk(
            total_vol=1.0, factor_vol=1.0, specific_vol=0.0,
            decomposition=pd.DataFrame({"variance_share": {"momentum": 1.0, "especifico": 0.0},
                                        "vol_contribution": {"momentum": 1.0, "especifico": 0.0}}),
            factor_contributions=pd.DataFrame(), exposures=pd.Series(dtype=float),
            factor_vols=pd.Series(dtype=float), mean_r2=0.2, n_days=252,
            factor_cov=pd.DataFrame(), specific_var=pd.Series(dtype=float),
        )

    monkeypatch.setattr(analysis.factor_model, "factor_risk", stub)
    a = _libro(mercado, risk_cfg, panel=pd.DataFrame({"date": [mercado["close"].index[0]]}))
    assert any("mas volatilidad que Ledoit-Wolf" in w for w in a.warnings)


def test_si_el_modelo_factorial_ve_mucho_menos_riesgo_tambien_se_avisa(mercado, risk_cfg, monkeypatch):
    """La regla es una diferencia de mas del 25% en cualquier sentido."""
    from sfc_tfsig.risk import factor_model

    def stub(weights, *args, **kwargs):
        return factor_model.FactorRisk(
            total_vol=0.001, factor_vol=0.001, specific_vol=0.0,
            decomposition=pd.DataFrame({"variance_share": {"momentum": 1.0, "especifico": 0.0},
                                        "vol_contribution": {"momentum": 0.001, "especifico": 0.0}}),
            factor_contributions=pd.DataFrame(), exposures=pd.Series(dtype=float),
            factor_vols=pd.Series(dtype=float), mean_r2=0.2, n_days=252,
            factor_cov=pd.DataFrame(), specific_var=pd.Series(dtype=float),
        )

    monkeypatch.setattr(analysis.factor_model, "factor_risk", stub)
    a = _libro(mercado, risk_cfg, panel=pd.DataFrame({"date": [mercado["close"].index[0]]}))
    assert any("menos volatilidad que Ledoit-Wolf" in w for w in a.warnings)


def test_historia_larga_sin_benchmark_cae_a_la_del_panel_sin_romper(mercado, risk_cfg):
    """Antes `book_backcast` lanzaba y `riesgo` moria sin reporte."""
    m = mercado
    sin_spy = m["close"].drop(columns="SPY")
    a = analysis.analyze(
        m["weights"], m["sectors"], float(m["nav"].iloc[-1]), m["close"], "SPY", risk_cfg,
        strategy_nav=m["nav"], benchmark_nav=m["bench"], as_of=m["close"].index[-1],
        volume_wide=m["volume"], factor_scores=m["scores"], stress_close=sin_spy,
    )
    assert any(w.startswith("historia de la cartera:") for w in a.warnings)
    assert a.stressed is not None                      # con la historia del panel
    assert (a.conditional_var["series"] == "cartera").any()


def test_si_el_modelo_factorial_falla_se_avisa_y_sigue(mercado, risk_cfg):
    future = pd.DataFrame({"date": [mercado["close"].index[-1] + pd.Timedelta(days=40)],
                           "ticker": ["T0"], "sector": ["Energy"]})
    a = _libro(mercado, risk_cfg, panel=future)
    assert a.factor_risk is None
    assert any(w.startswith("modelo factorial:") for w in a.warnings)
    assert a.status != "SIN DATO"


def test_el_drawdown_actual_es_una_perdida_positiva(mercado, risk_cfg):
    m = mercado
    nav = m["nav"].copy()
    nav.iloc[-30:] = nav.iloc[-30:] * 0.85
    a = analysis.analyze(
        m["weights"], m["sectors"], float(nav.iloc[-1]), m["close"], "SPY", risk_cfg,
        strategy_nav=nav, benchmark_nav=m["bench"], as_of=m["close"].index[-1],
    )
    expected = 1.0 - nav.iloc[-1] / nav.cummax().iloc[-1]
    assert a.current_drawdown == pytest.approx(expected)
    assert a.current_drawdown > 0.10
    assert a.measures["current_drawdown"] == pytest.approx(expected)


def test_la_liquidacion_de_crisis_usa_los_multiplicadores_de_la_politica(mercado, risk_cfg):
    from sfc_tfsig.risk import exposure, liquidity

    m = mercado
    a = _libro(m, risk_cfg, volume_wide=m["volume"], cost_bps=20.0)
    liq = risk_cfg.section("liquidity")
    adv = exposure.average_dollar_volume(
        m["close"].reindex(columns=a.weights.index), m["volume"], a.as_of,
        int(risk_cfg.get("exposure.adv_window_d")),
    ).reindex(a.weights.index)
    names = list(a.weights.index)
    daily_vol = pd.Series(np.sqrt(np.diag(a.ex_ante.cov_daily.loc[names, names])), index=names)
    _, expected = liquidity.liquidation_cost(
        a.weights, a.nav, adv, daily_vol, cost_bps=20.0, impact_coef=float(liq["impact_coef"]),
        adv_mult=float(liq["stress_adv_mult"]), vol_mult=float(liq["stress_vol_mult"]),
        spread_mult=float(liq["stress_cost_mult"]),
    )
    assert a.liquidation.loc["crisis", "cost"] == pytest.approx(expected)


def test_el_var_del_limite_existe_aunque_la_politica_no_pida_el_99(mercado):
    import copy as copy_mod

    data = copy_mod.deepcopy(dict(light_risk_config().data))
    data["var"]["confidence"] = [0.95]
    cfg = risk_config_from_dict(data)
    a = analysis.analyze_book(mercado["weights"], mercado["sectors"], 100_000.0, mercado["close"],
                              "SPY", cfg, as_of=mercado["close"].index[-1])
    assert np.isfinite(a.measures["var_99_1d"])
    assert 0.99 in set(a.book_var["confidence"])
