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

    assert set(long.columns) == {"date", "ticker", "close", "close_split", "close_raw", "volume"}
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
    assert list(vacio.columns) == ["date", "ticker", "close", "close_split", "close_raw", "volume"]


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


# ---------------------------------------------------------------------------
#  Precio real frente a precio ajustado (auditoria #1)
#
#  Los filtros de precio y la capitalizacion usaban el cierre ajustado por
#  splits y dividendos. NVDA cerro 2014 a 20,05 USD; ajustado por los splits de
#  2021 (4:1) y 2024 (10:1) sale a 0,48, y el modelo la excluia por precio y
#  por capitalizacion hasta 2017-2020. Eso es informacion del futuro.
# ---------------------------------------------------------------------------


def _yf_unadjusted(sessions, close, adj_close, splits, volume):
    """Forma de `yf.download(auto_adjust=False, actions=True)` para un ticker."""
    columns = pd.MultiIndex.from_product(
        [["Adj Close", "Close", "Stock Splits", "Volume"], ["AAA"]], names=["Price", "Ticker"]
    )
    frame = pd.DataFrame(
        {("Adj Close", "AAA"): adj_close, ("Close", "AAA"): close,
         ("Stock Splits", "AAA"): splits, ("Volume", "AAA"): volume},
        index=sessions,
    ).reindex(columns=columns)
    frame.index.name = "Date"
    return frame


def test_el_precio_real_deshace_los_splits_posteriores():
    sessions = pd.bdate_range("2024-06-05", periods=4)
    # Split 10:1 el tercer dia. Yahoo da `Close` ya dividido entre 10 antes del
    # split; el precio que de verdad cotizaba era 10 veces mayor.
    raw = _yf_unadjusted(
        sessions,
        close=[100.0, 101.0, 102.0, 103.0],
        adj_close=[99.0, 100.0, 101.5, 103.0],
        splits=[0.0, 0.0, 10.0, 0.0],
        volume=[5e6, 5e6, 5e6, 5e6],
    )
    long = prices_mod._tidy(raw, ["AAA"]).set_index("date")

    assert long["close"].tolist() == pytest.approx([99.0, 100.0, 101.5, 103.0])
    assert long["close_split"].tolist() == pytest.approx([100.0, 101.0, 102.0, 103.0])
    assert long["close_raw"].tolist() == pytest.approx([1000.0, 1010.0, 102.0, 103.0])


def test_el_precio_para_filtros_es_el_real_no_el_ajustado():
    sessions = pd.bdate_range("2024-06-05", periods=3)
    long = pd.DataFrame({
        "date": list(sessions), "ticker": "AAA",
        "close": [0.48, 0.49, 0.50], "close_split": [0.50, 0.51, 0.52],
        "close_raw": [20.0, 20.4, 20.8], "volume": [1e8, 1e8, 1e8],
    })
    wide = prices_mod.level_close(long)
    assert wide["AAA"].tolist() == pytest.approx([20.0, 20.4, 20.8])


def test_el_volumen_en_dolares_usa_el_precio_sin_ajuste_por_dividendos():
    sessions = pd.bdate_range("2024-01-02", periods=63)
    long = pd.DataFrame({
        "date": list(sessions), "ticker": "AAA",
        "close": np.full(63, 40.0), "close_split": np.full(63, 50.0),
        "close_raw": np.full(63, 500.0), "volume": np.full(63, 1e6),
    })
    mediana = prices_mod.median_dollar_volume(long)
    # Precio y volumen de Yahoo estan los dos ajustados por splits: su producto
    # es el volumen en dolares real. El ajuste por dividendos lo infravaloraba.
    assert mediana["AAA"].iloc[-1] == pytest.approx(50.0 * 1e6)
