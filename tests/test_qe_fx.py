"""Motor long/short: nombres que no cotizan en USD y su emparejamiento con Compustat."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine import trade_plan as tp
from quant_engine.analysis import fx_by_ticker, fx_plan
from quant_engine.data import cleaning
from quant_engine.data import wrds_data as wd
from quant_engine.data.loader import MarketData, quote_currency

# ---------------------------------------------------------------------------
#  Moneda de cotizacion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ticker, expected", [
    ("ATZ.TO", ("CAD", 1.0)), ("WEED.V", ("CAD", 1.0)), ("VOD.L", ("GBP", 0.01)),
    ("SAP.DE", ("EUR", 1.0)), ("AAPL", ("USD", 1.0)), ("BRK-B", ("USD", 1.0)),
    ("^GSPC", ("USD", 1.0)), ("CADUSD=X", ("USD", 1.0)), ("X.ZZ", (None, 1.0)),
])
def test_moneda_por_sufijo_de_bolsa(ticker, expected):
    assert quote_currency(ticker) == expected


def test_plan_de_fx_aplica_la_escala_y_deja_fuera_el_usd():
    currency, scale, unknown = fx_plan(["ATZ.TO", "VOD.L", "AAPL", "X.ZZ"])
    assert unknown == ["X.ZZ"]
    idx = pd.bdate_range("2026-01-01", periods=2)
    fx = fx_by_ticker(currency, scale, {"CAD": pd.Series(0.7, index=idx), "GBP": pd.Series(1.3, index=idx)})
    assert set(fx) == {"ATZ.TO", "VOD.L"}
    assert fx["VOD.L"].iloc[0] == pytest.approx(0.013)  # peniques: 0,01 libras por unidad de precio


# ---------------------------------------------------------------------------
#  Conversion a USD en la limpieza
# ---------------------------------------------------------------------------


def _ohlcv(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                         "Volume": 1e6})


def _clean_with_cad(rate: pd.Series):
    cal = pd.bdate_range("2026-01-01", periods=6)
    market = MarketData(ohlcv={"SPY": _ohlcv(pd.Series(100.0, index=cal)),
                               "ATZ.TO": _ohlcv(pd.Series(50.0, index=cal))})
    return cal, cleaning.align(market, "SPY", fx={"ATZ.TO": rate}, currency={"ATZ.TO": "CAD", "SPY": "USD"})


def test_retorno_en_usd_incluye_el_tipo_de_cambio_y_los_niveles_siguen_en_moneda_local():
    cal = pd.bdate_range("2026-01-01", periods=6)
    rate = pd.Series([0.70, 0.70, 0.77, 0.77, 0.77, 0.77], index=cal)
    _, clean = _clean_with_cad(rate)
    # Precio local plano, el CAD sube un 10%: el inversor en USD gana un 10%.
    assert clean.returns["ATZ.TO"].iloc[2] == pytest.approx(0.10)
    assert clean.close["ATZ.TO"].iloc[-1] == pytest.approx(50.0 * 0.77)
    close, high, _ = clean.local_prices("ATZ.TO")
    assert close.iloc[-1] == pytest.approx(50.0) and high.iloc[-1] == pytest.approx(50.5)
    assert clean.currency["ATZ.TO"] == "CAD"
    assert clean.fx_rate("SPY").eq(1.0).all() and clean.close["SPY"].iloc[-1] == 100.0


def test_festivo_de_divisas_arrastra_el_ultimo_cambio_y_nunca_usa_uno_futuro():
    cal = pd.bdate_range("2026-01-01", periods=6)
    # Sin cambio el primer dia ni el cuarto (festivo en el mercado de divisas).
    rate = pd.Series([0.70, 0.71, 0.80, 0.72], index=cal[[1, 2, 4, 5]])
    _, clean = _clean_with_cad(rate)
    assert np.isnan(clean.close["ATZ.TO"].iloc[0])
    assert clean.fx["ATZ.TO"].iloc[3] == pytest.approx(0.71)


# ---------------------------------------------------------------------------
#  Compustat e IBES
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yahoo, expected", [
    ("ATZ.TO", ("ATZ.", "CAN")), ("TECK-B.TO", ("TECK.B", "CAN")), ("RY.TO", ("RY.", "CAN")),
    ("BRK-B", ("BRK.B", "USA")), ("RY", ("RY", "USA")), ("aapl", ("AAPL", "USA")),
])
def test_ticker_de_compustat(yahoo, expected):
    assert wd.compustat_ticker(yahoo) == expected


def _securities():
    return pd.DataFrame({
        "tic": ["ATZAF", "ATZ.", "9640B", "RY", "RY.", "RY.PR.S"],
        "gvkey": ["187163", "187163", "187163", "015633", "015633", "015633"],
        "iid": ["01", "01C", "02C", "01", "01C", "75C"],
        "ibtic": ["ATZ", None, None, "RY2", "RY1", None],
        "excntry": ["USA", "CAN", "CAN", "USA", "CAN", "CAN"],
        "secstat": ["A", "A", "A", "A", "A", "A"],
        "gsector": ["25", "25", "25", "40", "40", "40"],
    })


def test_toronto_se_empareja_con_su_emision_y_hereda_el_ibtic_de_la_hermana():
    wanted = {t: wd.compustat_ticker(t) for t in ("ATZ.TO", "RY.TO", "RY", "NOPE.TO")}
    ids = wd.match_securities(_securities(), wanted)
    assert set(ids.index) == {"ATZ.TO", "RY.TO", "RY"}
    assert ids.loc["ATZ.TO", "gvkey"] == "187163" and ids.loc["ATZ.TO", "iid"] == "01C"
    assert ids.loc["ATZ.TO", "ibtic"] == "ATZ"     # del ADR de EE.UU.
    assert ids.loc["RY.TO", "ibtic"] == "RY1"      # la suya, no la del listado de EE.UU.
    assert ids.loc["RY", "iid"] == "01" and ids.loc["RY", "ibtic"] == "RY2"


def test_sin_coincidencias_no_revienta():
    ids = wd.match_securities(_securities(), {"NOPE": ("NOPE", "USA")})
    assert ids.empty and "gvkey" in ids.columns


def test_capitalizacion_en_la_moneda_en_que_reporta():
    prices = pd.Series({"ATZ.TO": 70.0, "SHOP.TO": 100.0, "AAPL": 200.0, "X": 10.0, "Y": 5.0})
    currencies = pd.Series({"ATZ.TO": "CAD", "SHOP.TO": "USD", "AAPL": "USD", "X": "BRL"})
    out = wd.price_in_currency(prices, currencies, {"CAD": 0.7})
    assert out["ATZ.TO"] == pytest.approx(100.0)
    assert out["SHOP.TO"] == 100.0 and out["AAPL"] == 200.0
    assert np.isnan(out["X"])                      # moneda sin tipo de cambio: sin ratio
    assert out["Y"] == 5.0                         # sin moneda declarada: USD


def test_ttm_lleva_la_moneda_del_ultimo_trimestre():
    ends = pd.date_range("2025-03-31", periods=5, freq="QE")
    fq = pd.DataFrame({"gvkey": "1", "datadate": ends, "rdq": ends + pd.Timedelta(days=30),
                       "fyearq": ends.year, "fqtr": (ends.month - 1) // 3 + 1, "saleq": 100.0,
                       "cshoq": 10.0, "curcdq": "CAD"})
    assert wd.ttm_fundamentals(fq, pd.Timestamp("2026-12-31"))["currency"] == "CAD"
    assert wd.ttm_fundamentals(fq.drop(columns="curcdq"), pd.Timestamp("2026-12-31"))["currency"] == "USD"


# ---------------------------------------------------------------------------
#  Tamano en USD y significancia de la regla
# ---------------------------------------------------------------------------


def test_tamano_divide_el_nocional_en_usd_por_el_precio_convertido():
    p = tp.PlanParams(max_risk_per_trade=1.0)
    out = tp.size_position(0.07, "CONFIRM", entry=100.0, stop_pct=0.05, capital=1_000_000, p=p, fx=0.7)
    assert out["notional"] == pytest.approx(70_000)
    assert out["shares"] == 1000                   # 70.000 USD / (100 CAD x 0,70)


def test_t_de_newey_west_descuenta_el_solape():
    rng = np.random.default_rng(3)
    iid = rng.normal(0.1, 1.0, 2000)
    naive = iid.mean() / (iid.std() / np.sqrt(len(iid)))
    assert tp.hac_t_stat(iid, 0) == pytest.approx(naive, rel=1e-3)
    # Media movil de 20: cada valor comparte 19 sumandos con el siguiente, como
    # dos entradas de la regla abiertas en dias seguidos.
    shocks = rng.normal(0.0, 1.0, 2020)
    overlap = np.convolve(shocks, np.ones(20) / 20, mode="valid") + 0.05
    naive = overlap.mean() / (overlap.std() / np.sqrt(len(overlap)))
    assert abs(tp.hac_t_stat(overlap, 20)) < abs(naive) / 2


def test_estadisticas_de_la_regla_traen_su_t():
    trades = pd.DataFrame({"r_mult": [2.0, -1.0, -1.0, 0.5], "exit": "x", "days": [10, 3, 5, 20]})
    assert np.isfinite(tp.trade_stats(trades)["t_stat"])
