"""Motor long/short: configuracion, limpieza de datos, senales y acuerdo con research."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine.app import parse_fraction
from quant_engine.data import cleaning, validation
from quant_engine.data.loader import MarketData
from quant_engine.settings import SettingsError, build_settings, parse_tickers
from quant_engine.signals import agreement, composite


# ---------------------------------------------------------------------------
#  Configuracion
# ---------------------------------------------------------------------------


def test_tickers_se_normalizan_y_deduplican():
    assert parse_tickers(" aapl, msft ,AAPL;nvda ") == ("AAPL", "MSFT", "NVDA")


@pytest.mark.parametrize("text,value", [("0.05%", 0.0005), ("0.0005", 0.0005), ("4%", 0.04), ("0.04", 0.04)])
def test_fracciones_aceptan_porcentaje_o_decimal(text, value):
    assert parse_fraction(text) == pytest.approx(value)


def test_un_ticker_no_puede_estar_largo_y_corto():
    with pytest.raises(SettingsError, match="LONG y SHORT a la vez"):
        build_settings("AAPL,MSFT", "MSFT,TSLA")


def test_el_benchmark_no_puede_estar_en_la_cartera():
    with pytest.raises(SettingsError, match="benchmark"):
        build_settings("SPY,AAPL", "TSLA")


def test_topes_que_impiden_invertir_la_pata_se_rechazan():
    with pytest.raises(SettingsError, match="no alcanza"):
        build_settings("AAPL", "TSLA")  # 1 nombre x 35% < 100% de pata


def test_coste_en_unidades_equivocadas_se_rechaza():
    with pytest.raises(SettingsError, match="unidades"):
        build_settings("A,B,C", "D,E,F", transaction_cost=5.0)


def test_signos_vienen_del_research():
    s = build_settings("A,B,C", "D,E,F")
    assert s.signs == {"A": 1, "B": 1, "C": 1, "D": -1, "E": -1, "F": -1}


def test_fingerprint_cambia_con_un_parametro():
    a = build_settings("A,B,C", "D,E,F")
    b = a.with_changes(construction="min_variance")
    assert a.fingerprint != b.fingerprint


# ---------------------------------------------------------------------------
#  Limpieza y calidad
# ---------------------------------------------------------------------------


def _ohlcv(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6})


def _market(calendar, **series) -> MarketData:
    data = {"SPY": _ohlcv(pd.Series(100.0, index=calendar))}
    for k, v in series.items():
        data[k] = _ohlcv(v)
    return MarketData(ohlcv=data)


def test_relleno_limitado_no_inventa_huecos_largos():
    cal = pd.bdate_range("2023-01-02", periods=30)
    s = pd.Series(100.0, index=cal).drop(cal[10:15])  # 5 sesiones sin cierre
    clean = cleaning.align(_market(cal, X=s), "SPY", max_ffill=2)
    assert clean.stats["X"]["filled"] == 2
    assert clean.stats["X"]["missing"] == 3


def test_duplicados_se_eliminan_y_se_cuentan():
    cal = pd.bdate_range("2023-01-02", periods=20)
    s = pd.Series(100.0, index=cal)
    dup = pd.concat([s, s.iloc[:3]])
    clean = cleaning.align(_market(cal, X=dup), "SPY")
    assert clean.stats["X"]["duplicates"] == 3


def test_precio_no_positivo_se_anula():
    cal = pd.bdate_range("2023-01-02", periods=20)
    s = pd.Series(100.0, index=cal)
    s.iloc[5] = 0.0
    clean = cleaning.align(_market(cal, X=s), "SPY", max_ffill=0)
    assert clean.stats["X"]["non_positive"] == 1
    assert np.isnan(clean.close.loc[cal[5], "X"])


def test_sin_benchmark_no_hay_calendario():
    with pytest.raises(ValueError, match="benchmark"):
        cleaning.align(MarketData(ohlcv={}, failures={"SPY": "sin datos"}), "SPY")


def test_calidad_marca_fail_con_historia_corta_y_warn_con_huecos():
    cal = pd.bdate_range("2022-01-03", periods=400)
    corta = pd.Series(100.0, index=cal[-50:])
    con_huecos = pd.Series(100.0 + np.arange(400) * 0.1, index=cal).drop(cal[100:104:2])
    market = _market(cal, CORTA=corta, HUECOS=con_huecos)
    market.failures["MUERTA"] = "Yahoo no devolvio datos"
    clean = cleaning.align(market, "SPY", max_ffill=0)
    report = validation.quality_report(clean, market, ["CORTA", "HUECOS", "MUERTA"], cal[0])
    assert report.loc["CORTA", "status"] == "FAIL"
    assert report.loc["MUERTA", "status"] == "FAIL"
    assert report.loc["HUECOS", "status"] in ("OK", "WARN")
    assert set(validation.failed(report)) == {"CORTA", "MUERTA"}


# ---------------------------------------------------------------------------
#  Senales y acuerdo
# ---------------------------------------------------------------------------


def _features(n=10, seed=14):
    rng = np.random.default_rng(seed)
    idx = [f"T{i}" for i in range(n)]
    return pd.DataFrame({
        "mom_12_1": rng.normal(size=n), "mom_6m": rng.normal(size=n), "mom_3m": rng.normal(size=n),
        "price_to_sma_200": rng.normal(size=n), "macd_hist_pct": rng.normal(size=n),
        "sharpe_1y": rng.normal(size=n), "vol_annual": rng.uniform(0.2, 0.6, n),
        "zscore_20d": rng.normal(size=n), "log_adv": rng.normal(20, 1, n),
        "beta": rng.uniform(0.5, 2, n), "trend_quality": rng.uniform(-1, 1, n),
    }, index=idx)


WEIGHTS = {"momentum": 0.25, "risk_adjusted_return": 0.2, "volatility": 0.15, "mean_reversion": 0.1,
           "liquidity": 0.1, "beta": 0.1, "statistical": 0.1}


def test_signos_de_los_componentes():
    f = _features()
    f.loc["T0", "vol_annual"] = 0.05   # el menos volatil
    f.loc["T1", "beta"] = 0.1          # la beta mas baja
    f.loc["T2", "zscore_20d"] = -5.0   # el mas estirado a la baja
    comps = composite.component_scores(f)
    assert comps["volatility"].idxmax() == "T0"
    assert comps["beta"].idxmax() == "T1"
    assert comps["mean_reversion"].idxmax() == "T2"


def test_score_con_un_solo_componente_es_ese_componente():
    f = _features()
    only_mom = composite.quant_score(f, {"momentum": 1.0})
    assert only_mom["quant_score"].equals(only_mom["momentum"])


def test_componente_desconocido_se_rechaza():
    with pytest.raises(ValueError, match="desconocidos"):
        composite.normalized_weights({"momentum": 1.0, "suerte": 1.0})


def test_estabilidad_del_ranking_es_alta_cuando_un_activo_domina_todo():
    f = _features()
    for col, sign in [(c, s) for comp in composite.COMPONENTS.values() for c, s in comp]:
        f[col] = np.arange(10) * sign  # todos los componentes ordenan igual
    out = composite.rank_stability(f, WEIGHTS, draws=100, seed=1)
    assert out["median_spearman"] == pytest.approx(1.0)


@pytest.mark.parametrize("aligned,label", [(0.8, "STRONG AGREEMENT"), (0.3, "WEAK AGREEMENT"),
                                           (0.0, "NEUTRAL"), (-0.5, "SIGNAL CONFLICT")])
def test_umbrales_de_acuerdo(aligned, label):
    assert agreement.classify(aligned, 0.5, 0.15, -0.15) == label


def test_un_corto_con_score_alto_es_conflicto_y_no_se_sobrescribe():
    scores = pd.DataFrame({"quant_score": [1.2, -1.2], "rank": [1, 2]}, index=["LNG", "SHT"])
    t = agreement.agreement_table(scores, {"LNG": -1, "SHT": -1}, {})
    assert t.loc["LNG", "agreement"] == "SIGNAL CONFLICT"
    assert t.loc["SHT", "agreement"] == "STRONG AGREEMENT"
    # La senal de research se conserva tal cual.
    assert (t["research"] == "SHORT").all()
