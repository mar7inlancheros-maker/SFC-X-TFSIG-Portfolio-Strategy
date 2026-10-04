import os
import time

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig.data import cache


@pytest.fixture
def tmp_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    return tmp_path


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", None],
        "filed": pd.to_datetime(["2019-01-05", "2019-12-31", None]),
        "value": [1.5, np.nan, 3.0],
        "shares": [10, 20, 30],
    })


def test_ida_y_vuelta_conserva_tipos_y_fechas(tmp_cache):
    df = _frame()
    cache.write_frame("sub/dir/x", df)
    back = cache.read_frame("sub/dir/x")
    pd.testing.assert_frame_equal(back, df)
    # La razon de no usar CSV: una fecha leida como texto ordena mal.
    assert back["filed"].dtype.kind == "M"
    assert (tmp_cache / "sub" / "dir" / f"x{cache.SUFFIX}").exists()


def test_falta_y_corrupta_son_ausencia(tmp_cache):
    assert cache.read_frame("nada") is None
    (tmp_cache / f"roto{cache.SUFFIX}").write_bytes(b"esto no es un pickle")
    assert cache.read_frame("roto") is None


def test_caducidad(tmp_cache):
    path = cache.write_frame("viejo", _frame())
    hace_10_dias = time.time() - 10 * 86400
    os.utime(path, (hace_10_dias, hace_10_dias))
    assert cache.read_frame("viejo", max_age_days=5) is None
    assert cache.read_frame("viejo", max_age_days=30) is not None
    assert cache.read_frame("viejo") is not None


def test_migra_parquet_viejo_y_conserva_su_fecha(tmp_cache, monkeypatch):
    legacy = tmp_cache / "precios.parquet"
    legacy.write_bytes(b"parquet de mentira")
    hace_10_dias = time.time() - 10 * 86400
    os.utime(legacy, (hace_10_dias, hace_10_dias))
    monkeypatch.setattr(cache, "_parquet_readable", lambda: True)
    monkeypatch.setattr(pd, "read_parquet", lambda path: _frame())

    # Con caducidad de 5 dias, el dato migrado sigue siendo de hace 10: no vale.
    assert cache.read_frame("precios", max_age_days=5) is None
    assert (tmp_cache / f"precios{cache.SUFFIX}").exists()
    assert cache.age_days("precios") == pytest.approx(10, abs=0.01)
    pd.testing.assert_frame_equal(cache.read_frame("precios"), _frame())


def test_sin_motor_de_parquet_la_cache_vieja_cuenta_como_ausente(tmp_cache, monkeypatch):
    (tmp_cache / "precios.parquet").write_bytes(b"x")
    monkeypatch.setattr(cache, "_parquet_readable", lambda: False)
    assert cache.read_frame("precios") is None
    assert not (tmp_cache / f"precios{cache.SUFFIX}").exists()
