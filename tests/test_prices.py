"""Capa de precios: transformaciones puras y memoria de fallos de descarga.

Nada aqui toca la red: `_tidy` recibe la forma exacta que devuelve yfinance
(columnas MultiIndex Price x Ticker) construida a mano.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig.data import cache, prices as prices_mod


def _yf_response(tickers, sessions, closes=None, volumes=None):
    """Replica la forma de `yf.download(...)` con varias acciones."""
    closes = closes or {t: np.linspace(100, 120, len(sessions)) for t in tickers}
    volumes = volumes or {t: np.full(len(sessions), 1_000_000.0) for t in tickers}
    columns = pd.MultiIndex.from_product([["Close", "Volume"], tickers],
                                         names=["Price", "Ticker"])
    data = {}
    for t in tickers:
        data[("Close", t)] = closes[t]
        data[("Volume", t)] = volumes[t]
    frame = pd.DataFrame(data, index=sessions).reindex(columns=columns)
    frame.index.name = "Date"
    return frame


def test_tidy_convierte_columnas_multiindex_en_marco_largo():
    sessions = pd.bdate_range("2023-01-02", periods=5)
    raw = _yf_response(["AAA", "BBB"], sessions)
    long = prices_mod._tidy(raw, ["AAA", "BBB"])

    assert set(long.columns) == {"date", "ticker", "close", "volume"}
    assert len(long) == 10
    assert set(long["ticker"]) == {"AAA", "BBB"}


def test_tidy_descarta_dias_sin_cierre():
    sessions = pd.bdate_range("2023-01-02", periods=4)
    closes = {"AAA": np.array([100.0, np.nan, 102.0, 103.0])}
    raw = _yf_response(["AAA"], sessions, closes=closes,
                       volumes={"AAA": np.full(4, 1e6)})
    long = prices_mod._tidy(raw, ["AAA"])
    assert len(long) == 3


def test_tidy_con_respuesta_vacia_devuelve_marco_con_tipos():
    vacio = prices_mod._tidy(pd.DataFrame(), ["AAA"])
    assert vacio.empty
    assert list(vacio.columns) == ["date", "ticker", "close", "volume"]


def test_retornos_mensuales_desde_cierres_diarios():
    sessions = pd.bdate_range("2023-01-02", "2023-03-31")
    close = pd.DataFrame({"AAA": np.linspace(100.0, 130.0, len(sessions))}, index=sessions)
    mensual = prices_mod.monthly_returns(close)
    assert len(mensual.dropna()) == 2
    assert (mensual.dropna() > 0).all().all()


def test_mediana_de_volumen_ignora_un_dia_extraordinario():
    """Un solo dia de volumen enorme no puede hacer pasar por liquida a una accion."""
    sessions = pd.bdate_range("2023-01-02", periods=120)
    volumen = np.full(len(sessions), 1_000.0)
    volumen[60] = 500_000_000.0
    long = pd.DataFrame({
        "date": sessions,
        "ticker": "AAA",
        "close": 10.0,
        "volume": volumen,
    })
    mediana = prices_mod.median_dollar_volume(long, window_d=63)
    assert mediana["AAA"].dropna().max() < 50_000.0


def test_retorno_futuro_mira_hacia_adelante():
    sessions = pd.bdate_range("2023-01-02", "2023-06-30")
    close = pd.DataFrame({"AAA": np.linspace(100.0, 200.0, len(sessions))}, index=sessions)
    forward = prices_mod.forward_returns(close, horizon_m=1)
    mensual = prices_mod.month_end(close)
    esperado = mensual["AAA"].iloc[1] / mensual["AAA"].iloc[0] - 1.0
    assert forward["AAA"].iloc[0] == pytest.approx(esperado)
    # El ultimo mes no tiene futuro dentro de la muestra.
    assert pd.isna(forward["AAA"].iloc[-1])


def test_los_fallos_de_descarga_se_recuerdan(tmp_path, monkeypatch):
    """Sin memoria de fallos, cada corrida vuelve a pedir los tickers muertos."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    prices_mod._record_failures(["MUERTA", "FANTASMA"])
    recientes = prices_mod._recent_failures()
    assert recientes == {"MUERTA", "FANTASMA"}


def test_un_fallo_viejo_se_vuelve_a_intentar(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    viejo = (pd.Timestamp.today().normalize() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    cache.write_json(prices_mod._FAILURES, {"ANTIGUA": viejo})
    assert "ANTIGUA" not in prices_mod._recent_failures()
