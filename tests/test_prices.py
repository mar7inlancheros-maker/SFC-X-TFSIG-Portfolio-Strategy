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


# ---------------------------------------------------------------------------
#  Descarga incremental: planificador
# ---------------------------------------------------------------------------

INICIO = pd.Timestamp("2009-07-31")
FIN = pd.Timestamp("2026-09-23")


def _registro(desde, hasta):
    return {"from": desde, "to": hasta}


def test_ticker_nuevo_pide_la_serie_completa():
    trabajos = prices_mod.plan_downloads(["NUEVO"], {}, INICIO, FIN)
    assert trabajos == [(["NUEVO"], INICIO, FIN)]


def test_ticker_que_salio_a_bolsa_despues_no_se_repide_cada_corrida():
    """El bug: primer precio en 2015 > inicio 2009 parecia 'falta historia'."""
    registro = {"IPO2015": _registro("2009-07-31", "2026-09-22")}
    assert prices_mod.plan_downloads(["IPO2015"], registro, INICIO, FIN) == []


def test_ticker_que_dejo_de_cotizar_no_se_repide_cada_corrida():
    """Se pidio hasta ayer; que su ultimo precio sea de 2018 da igual."""
    registro = {"MUERTA2018": _registro("2009-07-31", "2026-09-22")}
    assert prices_mod.plan_downloads(["MUERTA2018"], registro, INICIO, FIN) == []


def test_la_actualizacion_mensual_pide_solo_la_cola():
    registro = {"AAA": _registro("2009-07-31", "2026-08-31"),
                "BBB": _registro("2009-07-31", "2026-08-29")}
    trabajos = prices_mod.plan_downloads(["AAA", "BBB"], registro, INICIO, FIN)
    assert len(trabajos) == 1
    tickers, desde, hasta = trabajos[0]
    assert tickers == ["AAA", "BBB"]
    # Desde la ultima descarga (menos la tolerancia), no desde 2009.
    assert desde >= pd.Timestamp("2026-08-20")
    assert hasta == FIN


def test_un_ticker_muy_atrasado_no_arrastra_a_los_demas():
    """Agrupar las colas por mes evita bajar anos de historia para todos."""
    registro = {"AL_DIA": _registro("2009-07-31", "2026-08-31"),
                "ATRASADO": _registro("2009-07-31", "2020-01-31")}
    trabajos = prices_mod.plan_downloads(["AL_DIA", "ATRASADO"], registro, INICIO, FIN)
    por_ticker = {t: desde for tickers, desde, _ in trabajos for t in tickers}
    assert por_ticker["AL_DIA"] >= pd.Timestamp("2026-08-20")
    assert por_ticker["ATRASADO"] < pd.Timestamp("2020-02-01")


def test_si_nunca_se_pidio_desde_el_inicio_se_pide_completo():
    registro = {"CORTO": _registro("2020-01-02", "2026-09-22")}
    trabajos = prices_mod.plan_downloads(["CORTO"], registro, INICIO, FIN)
    assert trabajos == [(["CORTO"], INICIO, FIN)]


def test_el_registro_anota_lo_pedido_aunque_no_vengan_datos():
    registro = {}
    prices_mod._update_ledger(registro, ["X"], INICIO, FIN)
    prices_mod._update_ledger(registro, ["X"], pd.Timestamp("2026-08-01"), pd.Timestamp("2026-10-01"))
    assert registro["X"] == {"from": "2009-07-31", "to": "2026-10-01"}
