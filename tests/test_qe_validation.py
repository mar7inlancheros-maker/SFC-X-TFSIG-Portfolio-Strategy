"""Motor long/short: validacion del Quant Score, FLIP con evidencia y robustez de datos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine import trade_plan as tp
from quant_engine.analysis import features_at
from quant_engine.data import cleaning, loader
from quant_engine.data.loader import MarketData
from quant_engine.settings import build_settings
from quant_engine.signals import composite
from quant_engine.signals import score_validation as sv

# ---------------------------------------------------------------------------
#  Fechas y retornos futuros
# ---------------------------------------------------------------------------


def test_fines_de_mes_del_calendario_completo_sin_el_mes_en_curso():
    cal = pd.bdate_range("2026-01-01", "2026-04-15")
    dates = sv.month_end_dates(cal, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-15"), 12)
    assert dates == [pd.Timestamp("2026-01-30"), pd.Timestamp("2026-02-27"), pd.Timestamp("2026-03-31")]
    # Un `last` a mitad de mes no convierte su dia en cierre de mes.
    cut = sv.month_end_dates(cal, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-17"), 12)
    assert cut[-1] == pd.Timestamp("2026-02-27")
    assert sv.month_end_dates(cal, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-15"), 2)[0] == \
        pd.Timestamp("2026-02-27")


def test_retorno_futuro_desde_el_cierre_de_la_fecha():
    idx = pd.bdate_range("2026-01-01", periods=6)
    close = pd.DataFrame({"A": [100, 101, 102, 103, 104, 110.0]}, index=idx)
    fwd = sv.forward_returns(close, [idx[1], idx[4]], horizon=3)
    assert fwd.loc[idx[1], "A"] == pytest.approx(104 / 101 - 1)
    assert np.isnan(fwd.loc[idx[4], "A"])


# ---------------------------------------------------------------------------
#  IC del score
# ---------------------------------------------------------------------------


def _synthetic(signal: float, months: int = 24, names: int = 40, seed: int = 0):
    """Caracteristica = signal x retorno futuro + ruido. signal > 0: el score predice."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-31", periods=months, freq="ME")
    tickers = [f"T{i}" for i in range(names)]
    fwd = pd.DataFrame(rng.normal(0, 0.05, (months, names)), index=dates, columns=tickers)
    noise = pd.DataFrame(rng.normal(0, 0.05, (months, names)), index=dates, columns=tickers)
    feats = signal * fwd + noise

    def features(d):
        return pd.DataFrame({"sharpe_1y": feats.loc[d]})

    return features, fwd


@pytest.mark.parametrize("signal, expected", [(1.0, "predictive"), (0.0, "none"), (-1.0, "inverse")])
def test_la_validacion_distingue_score_util_ruido_y_al_reves(signal, expected):
    features, fwd = _synthetic(signal)
    out = sv.validate(features, fwd, {"risk_adjusted_return": 1.0, "analyst": 0.5})
    assert out["available"] and out["months"] == 24
    assert sv.evidence(out, 2.0) == expected
    assert "risk_adjusted_return" in out["components"].index
    assert out["not_validated"] == ["analyst"]
    if signal > 0:
        assert out["ic_mean"] > 0.3 and out["spread_mean"] > 0


def test_pocos_meses_no_validan():
    features, fwd = _synthetic(1.0, months=4)
    out = sv.validate(features, fwd, {"risk_adjusted_return": 1.0})
    assert not out["available"] and "6" in out["reason"]
    assert sv.evidence(out, 2.0) == "unknown"


def _random_market(n_names=24, n_days=420, seed=5):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2024-01-01", periods=n_days)
    ohlcv = {}
    for i, t in enumerate(["SPY"] + [f"N{i}" for i in range(n_names)]):
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015 + 0.001 * i, n_days))), index=cal)
        ohlcv[t] = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                                 "Volume": 1e6 + i})
    return MarketData(ohlcv=ohlcv)


def test_las_caracteristicas_de_una_fecha_no_ven_precios_posteriores():
    market = _random_market()
    settings = build_settings(("N0", "N1", "N2"), ("N3", "N4", "N5"), lookback="1y")
    names = [t for t in market.ohlcv if t != "SPY"]
    clean = cleaning.align(market, "SPY")
    d = clean.calendar[300]
    before = features_at(settings, clean, names, d, None, {})
    for t in names:
        frame = market.ohlcv[t]
        later = frame.index > d
        market.ohlcv[t].loc[later, ["Open", "High", "Low", "Close"]] *= 3.0
    after = features_at(settings, cleaning.align(market, "SPY"), names, d, None, {})
    pd.testing.assert_frame_equal(before, after)
    assert len(before) == len(names) and before["mom_6m"].notna().all()


def test_valor_historico_solo_con_trimestres_ya_publicados():
    from quant_engine.data import wrds_data as wd

    market = _random_market()
    settings = build_settings(("N0", "N1", "N2"), ("N3", "N4", "N5"), lookback="1y")
    names = [t for t in market.ohlcv if t != "SPY"]
    clean = cleaning.align(market, "SPY")
    d = clean.calendar[300]
    ends = pd.date_range(d - pd.DateOffset(months=21), periods=8, freq="QE")
    rows = []
    for i, t in enumerate(names):
        for e in ends:
            # El trimestre que se publica DESPUES de `d` trae un beneficio enorme.
            niq = 1e6 if e + pd.Timedelta(days=30) > d else 10.0 + i
            rows.append({"gvkey": str(i), "datadate": e, "rdq": e + pd.Timedelta(days=30),
                         "fyearq": e.year, "fqtr": (e.month - 1) // 3 + 1, "saleq": 100.0, "niq": niq,
                         "ceqq": 200.0, "cshoq": 10.0, "curcdq": "USD"})
    wrds = wd.WrdsData(available=True, ids=pd.DataFrame({"gvkey": [str(i) for i in range(len(names))]},
                                                        index=names),
                       fundq=pd.DataFrame(rows))
    f = features_at(settings, clean, names, d, wrds, {})
    assert f["earnings_yield"].notna().all()
    price = clean.close.loc[d, "N0"]
    assert f.loc["N0", "earnings_yield"] == pytest.approx(40.0 / (price * 10.0))


def test_interes_corto_historico_respeta_el_retraso_de_publicacion():
    from quant_engine.data import wrds_data as wd

    raw = pd.DataFrame({"gvkey": "1", "iid": "01", "shortint": [1e6, 2e6, 9e6],
                        "datadate": pd.to_datetime(["2026-01-15", "2026-01-31", "2026-02-13"])})
    ids = pd.DataFrame({"gvkey": ["1"], "iid": ["01"]}, index=["X"])
    d = pd.Timestamp("2026-02-20")
    # Liquidado el 13 de febrero, publicado hacia el 25: el 20 todavia no se conocia.
    assert wd.short_interest_at(raw, ids, d, wd.SHORT_INTEREST_LAG_DAYS).loc["X", "shortint"] == 2e6
    assert wd.short_interest_at(raw, ids, d).loc["X", "shortint"] == 9e6  # hoy: todo lo de WRDS esta publicado
    other_listing = pd.DataFrame({"gvkey": ["1"], "iid": ["01C"]}, index=["X.TO"])
    assert wd.short_interest_at(raw, other_listing, d).empty


def test_analistas_historicos_solo_con_resumenes_hasta_la_fecha():
    from quant_engine.data import wrds_data as wd

    recs = pd.DataFrame({"ticker": "IB", "statpers": ["2026-01-15", "2026-04-16", "2026-07-16"],
                         "meanrec": [3.0, 2.5, 1.0], "numrec": 10})
    eps = pd.DataFrame({"ticker": "IB", "fpedats": "2026-12-31",
                        "statpers": ["2026-01-15", "2026-04-16", "2026-07-16"], "meanest": [1.0, 1.1, 5.0],
                        "stdev": 0.1, "numest": 10, "numup": 1, "numdown": 0})
    ids = pd.DataFrame({"ibtic": ["IB"]}, index=["X"])
    out = wd.analyst_panel(recs, eps, ids, pd.Timestamp("2026-05-01"))
    assert out.loc["X", "meanrec"] == 2.5                     # el de julio no existia
    assert out.loc["X", "eps_revision_3m"] == pytest.approx(0.1)
    assert out.loc["X", "rec_change_3m"] == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
#  FLIP con evidencia
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("validation, min_t, allowed", [
    ({}, 2.0, True),                                         # validacion apagada: se respeta
    ({"available": False, "reason": "x"}, 2.0, False),       # sin datos: sin evidencia
    ({"available": True, "ic_t": 2.4}, 2.0, True),
    ({"available": True, "ic_t": 0.5}, 2.0, False),
    ({"available": True, "ic_t": -3.0}, 2.0, False),
    ({"available": True, "ic_t": 0.5}, 0.0, True),           # regla desactivada
])
def test_un_flip_exige_que_el_score_haya_predicho(validation, min_t, allowed):
    assert tp.flip_allowed(validation, min_t) is allowed


def test_sin_evidencia_el_flip_queda_no_trade_y_el_resto_no_cambia():
    signs = pd.Series({"A": -1.0, "B": 1.0, "C": 0.0})
    verdicts = pd.Series({"A": "FLIP", "B": "CONFIRM", "C": "NO TRADE"})
    s, v, blocked = tp.block_flips(signs, verdicts)
    assert blocked == ["A"] and s["A"] == 0.0 and v["A"] == "NO TRADE"
    assert s["B"] == 1.0 and v["B"] == "CONFIRM"


def test_params_leen_el_umbral_del_flip():
    s = build_settings(("AAPL",), ("MSFT",))
    assert tp.params_from_settings(s).flip_min_ic_t == pytest.approx(
        float((s.get("trade_plan", {}) or {}).get("flip_min_ic_t", 2.0)))


# ---------------------------------------------------------------------------
#  Winsorizacion
# ---------------------------------------------------------------------------


def test_un_outlier_no_aplasta_al_resto_de_la_seccion_cruzada():
    rng = np.random.default_rng(2)
    base = pd.Series(rng.normal(0.04, 0.02, 40), index=[f"T{i}" for i in range(40)])
    with_outlier = base.copy()
    with_outlier.iloc[0] = 1.05  # un 105% de rendimiento de caja
    z_clean = composite.zscore(base)
    z_out = composite.zscore(with_outlier)
    # Sin recorte, la desviacion se multiplicaria y el resto quedaria cerca de cero.
    others = with_outlier.index[1:]
    assert z_out[others].std() > 0.6 * z_clean[others].std()
    assert z_out.iloc[0] <= composite.Z_CAP


def test_con_pocos_nombres_no_se_recorta():
    s = pd.Series([1.0, 2.0, 3.0, 100.0])
    assert composite.winsorize(s).equals(s)


# ---------------------------------------------------------------------------
#  Yahoo: reintento y cache vieja
# ---------------------------------------------------------------------------


def _frame(start="2026-01-01", n=5):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)


def test_lo_que_falta_se_reintenta_en_lotes_pequenos():
    calls = []

    def fake(tickers, start, end):
        calls.append(list(tickers))
        # El lote grande pierde la mitad; el reintento la trae.
        return {t: _frame() for t in tickers[: len(tickers) // 2]} if len(calls) == 1 else \
            {t: _frame() for t in tickers}

    tickers = [f"T{i}" for i in range(60)]
    out = loader.download_with_retry(tickers, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"), fake)
    assert set(out) == set(tickers)
    assert [len(c) for c in calls] == [60, 25, 5]


def test_si_yahoo_falla_se_usa_la_cache_vieja_y_se_dice(monkeypatch):
    start = pd.Timestamp("2026-01-01")
    old = _frame(n=8).reset_index(names="date")
    monkeypatch.setattr(loader.cache, "read_frame",
                        lambda name, max_age_days=None: None if max_age_days is not None else old)
    monkeypatch.setattr(loader.cache, "read_json", lambda name: {"AMZN": "2026-01-01"})
    monkeypatch.setattr(loader.cache, "write_json", lambda name, data: None)
    monkeypatch.setattr(loader.cache, "write_frame", lambda name, frame: None)
    monkeypatch.setattr(loader.cache, "age_days", lambda name: None)
    monkeypatch.setattr(loader, "_download", lambda tickers, s, e: {})
    out = loader.load_ohlcv(["AMZN", "NOPE"], start, pd.Timestamp("2026-02-01"))
    assert "AMZN" in out.ohlcv and out.stale["AMZN"] == old["date"].max()
    assert "NOPE" in out.failures and "AMZN" not in out.failures


def test_ttm_cuenta_por_fecha_de_publicacion_y_no_por_cierre():
    from quant_engine.data import wrds_data as wd

    ends = pd.to_datetime(["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"])
    rows = [{"gvkey": "1", "datadate": e, "rdq": e + pd.Timedelta(days=40), "saleq": 100.0, "niq": 10.0,
             "cshoq": 10.0, "curcdq": "USD"} for e in ends]
    rows[-1]["niq"] = 1e6
    fq = pd.DataFrame(rows)
    # El 20 de julio el trimestre de junio ya cerro pero no se ha publicado (sale el 9 de agosto).
    out = wd.ttm_fundamentals(fq, pd.Timestamp("2026-07-20"))
    assert out["fiscal_period_end"] == pd.Timestamp("2026-03-31") and out["net_income"] == pytest.approx(40.0)
    # Sin rdq se supone publicado 90 dias despues del cierre, nunca antes.
    no_rdq = fq.assign(rdq=pd.NaT)
    assert wd.ttm_fundamentals(no_rdq, pd.Timestamp("2026-09-15"))["fiscal_period_end"] == pd.Timestamp("2026-03-31")
    assert wd.ttm_fundamentals(no_rdq, pd.Timestamp("2026-10-15"))["fiscal_period_end"] == pd.Timestamp("2026-06-30")



def test_la_barra_de_hoy_nunca_entra_ni_se_guarda(monkeypatch):
    # El 10-nov-2026 a las 15:34 el 63,01 de media sesion de NFLX paso por
    # "cierre" y quedo en la cache.
    today = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=today, periods=30, name="date")
    frame = pd.DataFrame({"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.0, "Volume": 1e6}, index=idx)
    saved = {}
    monkeypatch.setattr(loader.cache, "read_frame", lambda name, max_age_days=None: None)
    monkeypatch.setattr(loader.cache, "read_json", lambda name: {})
    monkeypatch.setattr(loader.cache, "write_json", lambda name, data: None)
    monkeypatch.setattr(loader.cache, "write_frame", lambda name, f: saved.__setitem__(name, f))
    monkeypatch.setattr(loader, "_download", lambda tickers, s, e: {t: frame for t in tickers})
    out = loader.load_ohlcv(["NFLX"], idx[0], today)
    assert out.ohlcv["NFLX"].index.max() < today
    assert pd.to_datetime(saved["qe_ohlcv/NFLX"]["date"]).max() < today


def test_ultima_sesion_cerrada_de_nyse():
    assert loader.last_settled_session("2026-11-10") == pd.Timestamp("2026-11-09")
    assert loader.last_settled_session("2026-11-09") == pd.Timestamp("2026-11-06")   # lunes -> viernes
    assert loader.last_settled_session("2026-11-27") == pd.Timestamp("2026-11-25")   # tras Thanksgiving


def test_la_cache_vale_solo_si_trae_la_ultima_sesion_y_se_escribio_despues():
    idx = pd.bdate_range(end="2026-11-09", periods=10)
    f = pd.DataFrame({"Close": 1.0}, index=idx)
    today = "2026-11-10"
    assert loader.cache_usable(f, pd.Timestamp("2026-11-10 08:00"), today)
    # Escrita el mismo dia que su ultima barra: esa barra pudo ser de media sesion.
    assert not loader.cache_usable(f, pd.Timestamp("2026-11-09 15:34"), today)
    # Le falta la ultima sesion cerrada: el analisis saldria con el cierre de anteayer.
    assert not loader.cache_usable(f.iloc[:-1], pd.Timestamp("2026-11-10 08:00"), today)
    assert not loader.cache_usable(f, None, today)
