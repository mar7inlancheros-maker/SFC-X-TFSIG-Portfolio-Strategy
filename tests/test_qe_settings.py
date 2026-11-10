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
    with pytest.raises(SettingsError, match="both LONG and SHORT"):
        build_settings("AAPL,MSFT", "MSFT,TSLA")


def test_el_benchmark_no_puede_estar_en_la_cartera():
    with pytest.raises(SettingsError, match="benchmark"):
        build_settings("SPY,AAPL", "TSLA")


def test_con_pocos_nombres_el_tope_sube_al_minimo_y_se_anota():
    s = build_settings("AAPL,MSFT", "TSLA,INTC")  # 2 x 35% < 100% de pata
    assert s.get("portfolio.max_position") == pytest.approx(0.5)
    assert "raised" in s.get("_notes")[0]


def test_coste_en_unidades_equivocadas_se_rechaza():
    with pytest.raises(SettingsError, match="units"):
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
    with pytest.raises(ValueError, match="unknown components"):
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


# ---------------------------------------------------------------------------
#  Entrada: tickers y cuales van LONG
# ---------------------------------------------------------------------------


def test_los_marcados_van_long_y_el_resto_short():
    from quant_engine.app import split_long_short

    longs, shorts = split_long_short(("AAPL", "MSFT", "TSLA", "INTC"), "msft, aapl")
    assert longs == ("MSFT", "AAPL")
    assert shorts == ("TSLA", "INTC")


def test_sin_marcar_la_primera_mitad_va_long():
    from quant_engine.app import split_long_short

    assert split_long_short(("A", "B", "C"), "") == (("A", "B"), ("C",))


def test_long_fuera_de_la_lista_se_rechaza():
    from quant_engine.app import split_long_short

    with pytest.raises(SettingsError, match="not in the ticker list"):
        split_long_short(("A", "B"), "Z")


def test_parse_positions():
    from quant_engine.settings import SettingsError, parse_positions

    assert parse_positions("aapl:500, TSLA:-200 ; nvda=+1_000") == (("AAPL", 500), ("TSLA", -200), ("NVDA", 1000))
    assert parse_positions("") == ()
    assert parse_positions("AAPL:0") == ()
    for bad in ("AAPL 500", "AAPL:1.5", "AAPL:10,AAPL:20"):
        with pytest.raises(SettingsError):
            parse_positions(bad)


def test_canasta_solo_long_invierte_el_100_por_ciento():
    from quant_engine.settings import build_settings

    s = build_settings("ARES,AMC", "")
    assert s.long_only
    assert s.exposure == (1.0, 1.0)
    # Dos nombres con tope de 35% no llegan al 100%: el tope sube a 50% y se anota.
    assert s.get("portfolio.max_position") == pytest.approx(0.5)
    assert any("2 name" in n for n in s.get("_notes"))


def test_long_short_conserva_la_exposicion_del_yaml():
    from quant_engine.settings import build_settings

    s = build_settings("AAPL,MSFT,NVDA", "TSLA,INTC,BA")
    assert not s.long_only
    assert s.exposure == (2.0, 0.0)


def test_sin_long_se_rechaza():
    from quant_engine.settings import SettingsError, build_settings

    with pytest.raises(SettingsError, match="at least one LONG ticker"):
        build_settings("", "TSLA")


def test_marcar_todos_long_da_canasta_solo_long():
    from quant_engine.app import split_long_short

    longs, shorts = split_long_short(("ARES", "AMC"), "ares,amc")
    assert longs == ("ARES", "AMC") and shorts == ()


def test_pares_de_referencia_solo_para_canastas_solo_long_o_pequenas():
    from quant_engine.analysis import reference_peers
    from quant_engine.settings import build_settings

    solo_long = reference_peers(build_settings("ARES,AMC", ""))
    assert len(solo_long) >= 30 and "ARES" not in solo_long and "SPY" not in solo_long
    assert reference_peers(build_settings("AAPL,MSFT,NVDA,AMZN,META", "TSLA,INTC,BA,PYPL,NKE")) == []
    assert len(reference_peers(build_settings("AAPL,MSFT", "TSLA,INTC"))) >= 30  # 4 < min_cross_section


def test_zscore_acotado_en_3_y_sin_efecto_en_canastas_de_10():
    extremo = pd.Series([0.0] * 51 + [100.0])
    assert composite.zscore(extremo).max() == pytest.approx(composite.Z_CAP)
    diez = pd.Series([0.0] * 9 + [100.0])
    assert composite.zscore(diez).max() < composite.Z_CAP     # (n-1)/sqrt(n) = 2,85


def test_pares_se_deciden_con_los_nombres_que_quedan_tras_la_calidad():
    from quant_engine.analysis import reference_peers
    from quant_engine.settings import build_settings

    s = build_settings("ARES,AMC", "SKHY,ATZ.TO,GOLD")       # 5 pedidos
    assert reference_peers(s) == []                          # con los 5, no hacen falta
    assert len(reference_peers(s, n_names=4)) >= 30          # sin SKHY quedan 4: si
