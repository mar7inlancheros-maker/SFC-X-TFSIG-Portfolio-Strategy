"""Carga de precios del motor long/short: sesion en curso y validez de la cache.

Sin red: `_download` y la cache se sustituyen por versiones en memoria.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quant_engine.data import loader


def _bars(dates: list[str], close: float = 100.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
                         "Volume": 1e6}, index=idx)


class _MemoryCache:
    def __init__(self):
        self.frames, self.jsons = {}, {}

    def read_frame(self, name, max_age_days=None):
        return self.frames.get(name)

    def write_frame(self, name, df):
        self.frames[name] = df.copy()

    def read_json(self, name, max_age_days=None):
        return self.jsons.get(name)

    def write_json(self, name, payload):
        self.jsons[name] = payload


@pytest.mark.parametrize("now, expected", [
    ("2026-10-02 11:50", "2026-10-01"),   # viernes, mercado abierto: la ultima completa es el jueves
    ("2026-10-02 16:05", "2026-10-01"),   # recien cerrado: la barra aun no es definitiva
    ("2026-10-02 17:30", "2026-10-02"),   # viernes, ya cerrado
    ("2026-10-03 10:00", "2026-10-02"),   # sabado
])
def test_ultima_sesion_completa(now, expected):
    # Auditoria #7. `as_of` era hoy aunque el mercado siguiera abierto.
    ts = pd.Timestamp(now, tz="America/New_York")
    assert loader.last_complete_session(ts) == pd.Timestamp(expected)


def test_la_barra_de_la_sesion_en_curso_no_entra(monkeypatch):
    # Yahoo devuelve una barra intradia para hoy si se le pide hasta hoy + 1.
    # Con `end` = ultima sesion completa, esa barra no puede entrar aunque
    # Yahoo la mande.
    mem = _MemoryCache()
    monkeypatch.setattr(loader, "cache", mem)
    monkeypatch.setattr(loader, "_download",
                        lambda tickers, start, end: {t: _bars(["2026-09-30", "2026-10-01", "2026-10-02"])
                                                     for t in tickers})
    out = loader.load_ohlcv(["AAA"], pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-01"),
                            now=pd.Timestamp("2026-10-02 11:50", tz="America/New_York"))
    assert out.ohlcv["AAA"].index.max() == pd.Timestamp("2026-10-01")


def test_cache_bajada_antes_del_cierre_se_vuelve_a_pedir(monkeypatch):
    # Una cache escrita el 1 de octubre a las 15:00 (Nueva York) lleva la barra
    # intradia del dia 1. Al dia siguiente `end` es el 1 de octubre: la cache
    # tiene esa fecha pero con un precio que no es el de cierre.
    mem = _MemoryCache()
    mem.frames[loader._cache_name("AAA")] = _bars(["2026-09-30", "2026-10-01"], close=90.0).reset_index()
    mem.jsons[loader._REQUESTED] = {"AAA": "2026-09-01"}
    mem.jsons[loader._FETCHED] = {"AAA": "2026-10-01T15:00:00-04:00"}
    calls = []

    def fake_download(tickers, start, end):
        calls.append(list(tickers))
        return {t: _bars(["2026-09-30", "2026-10-01"], close=100.0) for t in tickers}

    monkeypatch.setattr(loader, "cache", mem)
    monkeypatch.setattr(loader, "_download", fake_download)
    out = loader.load_ohlcv(["AAA"], pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-01"),
                            now=pd.Timestamp("2026-10-02 11:50", tz="America/New_York"))
    assert calls == [["AAA"]]
    assert out.ohlcv["AAA"]["Close"].iloc[-1] == 100.0


def test_cache_bajada_tras_el_cierre_se_reutiliza(monkeypatch):
    mem = _MemoryCache()
    mem.frames[loader._cache_name("AAA")] = _bars(["2026-09-30", "2026-10-01"]).reset_index()
    mem.jsons[loader._REQUESTED] = {"AAA": "2026-09-01"}
    mem.jsons[loader._FETCHED] = {"AAA": "2026-10-01T23:28:00-04:00"}
    monkeypatch.setattr(loader, "cache", mem)
    monkeypatch.setattr(loader, "_download", lambda *a: pytest.fail("no deberia descargar"))
    out = loader.load_ohlcv(["AAA"], pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-01"),
                            now=pd.Timestamp("2026-10-02 11:50", tz="America/New_York"))
    assert out.ohlcv["AAA"].index.max() == pd.Timestamp("2026-10-01")
