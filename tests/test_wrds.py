"""Conector de WRDS, probado sin red: la conexion es un doble de prueba.

Cada test de la segunda mitad fija un defecto concreto de la primera version:
conexiones que no se cerraban, cache que devolvia datos de otra consulta, `.env`
leido desde el directorio equivocado y sin tolerar el BOM de PowerShell, y
procesos sin terminal que se quedaban esperando una contrasena.
"""

import pandas as pd
import pytest

from sfc_tfsig.data import cache, wrds as wrds_mod


class FakeConnection:
    def __init__(self, username="demo_user", fail=False):
        self.username = username
        self.calls = []
        self.closed = False
        self.fail = fail

    def raw_sql(self, query):
        self.calls.append(query)
        if self.fail:
            raise RuntimeError("fallo de consulta")
        return pd.DataFrame({"date": ["2024-01-02"], "permno": [10001]})

    def list_libraries(self):
        return ["ff", "crsp", "comp"]

    def close(self):
        self.closed = True


@pytest.fixture
def pgpass(tmp_path, monkeypatch):
    """Contrasena guardada: la conexion puede abrirse sin terminal."""
    path = tmp_path / "pgpass.conf"
    path.write_text("wrds-pgdata.wharton.upenn.edu:9737:wrds:demo_user:x\n")
    monkeypatch.setattr(wrds_mod, "_pgpass_path", lambda: path)
    return path


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


def _patch_wrds(monkeypatch, conn):
    captured = {}

    def fake_connection(**kwargs):
        captured.update(kwargs)
        return conn

    monkeypatch.setattr(wrds_mod, "_import_wrds",
                        lambda: type("W", (), {"Connection": staticmethod(fake_connection)})())
    return captured


# ---------------------------------------------------------------------------
#  Comportamiento basico
# ---------------------------------------------------------------------------


def test_get_connection_usa_wrds_username_del_entorno(monkeypatch, pgpass):
    monkeypatch.setenv("WRDS_USERNAME", "demo_user")
    captured = _patch_wrds(monkeypatch, FakeConnection())
    conn = wrds_mod.get_connection()
    assert conn.username == "demo_user"
    assert captured["wrds_username"] == "demo_user"


def test_fetch_table_usa_la_cache_si_existe(monkeypatch, isolated_cache):
    sql = "SELECT * FROM crsp.msf LIMIT 10"
    expected = pd.DataFrame({"permno": [10001]})
    cache.write_frame(wrds_mod._cache_key("wrds_test", sql), expected)
    monkeypatch.setattr(wrds_mod, "get_connection", lambda: pytest.fail("no debia conectar"))

    result = wrds_mod.fetch_table("crsp.msf", cache_name="wrds_test", limit=10)
    pd.testing.assert_frame_equal(result, expected)


def test_fetch_table_guarda_en_cache_si_falta(monkeypatch, isolated_cache):
    conn = FakeConnection()
    monkeypatch.setattr(wrds_mod, "get_connection", lambda: conn)
    result = wrds_mod.fetch_table("crsp.msf", cache_name="wrds_test", limit=10)
    key = wrds_mod._cache_key("wrds_test", "SELECT * FROM crsp.msf LIMIT 10")
    pd.testing.assert_frame_equal(cache.read_frame(key), result)


# ---------------------------------------------------------------------------
#  Defecto 1: conexiones que no se cerraban
# ---------------------------------------------------------------------------


def test_la_conexion_se_cierra_tras_la_consulta(monkeypatch):
    conn = FakeConnection()
    monkeypatch.setattr(wrds_mod, "get_connection", lambda: conn)
    wrds_mod.query("SELECT 1")
    assert conn.closed


def test_la_conexion_se_cierra_aunque_la_consulta_falle(monkeypatch):
    conn = FakeConnection(fail=True)
    monkeypatch.setattr(wrds_mod, "get_connection", lambda: conn)
    with pytest.raises(RuntimeError, match="fallo de consulta"):
        wrds_mod.query("SELECT 1")
    assert conn.closed


def test_una_conexion_pasada_se_reutiliza_y_no_se_cierra(monkeypatch):
    shared = FakeConnection()
    monkeypatch.setattr(wrds_mod, "get_connection", lambda: pytest.fail("no debia abrir otra"))
    wrds_mod.query("SELECT 1", conn=shared)
    wrds_mod.query("SELECT 2", conn=shared)
    assert shared.calls == ["SELECT 1", "SELECT 2"]
    assert not shared.closed


# ---------------------------------------------------------------------------
#  Defecto 2: la cache ignoraba la consulta
# ---------------------------------------------------------------------------


def test_otra_consulta_con_el_mismo_nombre_no_devuelve_datos_viejos(monkeypatch, isolated_cache):
    conn = FakeConnection()
    monkeypatch.setattr(wrds_mod, "get_connection", lambda: conn)
    wrds_mod.fetch_table("crsp.msf", where_clause="date >= '2020-01-01'", cache_name="msf")
    wrds_mod.fetch_table("crsp.msf", where_clause="date >= '2015-01-01'", cache_name="msf")
    # Filtro distinto: tiene que volver a consultar, no servir el de 2020.
    assert len(conn.calls) == 2


def test_la_clave_ignora_espacios_y_saltos_de_linea():
    a = wrds_mod._cache_key("x", "SELECT *\n  FROM crsp.msf")
    b = wrds_mod._cache_key("x", "SELECT * FROM crsp.msf")
    assert a == b


def test_refresh_ignora_la_cache(monkeypatch, isolated_cache):
    conn = FakeConnection()
    monkeypatch.setattr(wrds_mod, "get_connection", lambda: conn)
    wrds_mod.query("SELECT 1", cache_name="q")
    wrds_mod.query("SELECT 1", cache_name="q", refresh=True)
    assert len(conn.calls) == 2


# ---------------------------------------------------------------------------
#  Defecto 3: .env desde el directorio equivocado y sin tolerar BOM
# ---------------------------------------------------------------------------


def test_env_con_espacios_y_comillas_se_normaliza(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('WRDS_USERNAME = "demo_user"\n', encoding="utf-8")
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    monkeypatch.setattr(wrds_mod, "_env_file", lambda: env)
    assert wrds_mod._read_env_value("WRDS_USERNAME") == "demo_user"


def test_env_escrito_por_powershell_con_bom_se_lee(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("WRDS_USERNAME=demo_user\n", encoding="utf-8-sig")  # BOM de Out-File
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    monkeypatch.setattr(wrds_mod, "_env_file", lambda: env)
    assert wrds_mod._read_env_value("WRDS_USERNAME") == "demo_user"


def test_env_se_busca_en_la_raiz_del_repo_no_en_el_directorio_actual(tmp_path, monkeypatch):
    from sfc_tfsig.paths import ROOT

    monkeypatch.chdir(tmp_path)
    assert wrds_mod._env_file() == ROOT / ".env"


# ---------------------------------------------------------------------------
#  Defecto 4: sin terminal y sin contrasena guardada, fallar en vez de colgarse
# ---------------------------------------------------------------------------


def test_sin_pgpass_y_sin_terminal_falla_con_instrucciones(tmp_path, monkeypatch):
    monkeypatch.setenv("WRDS_USERNAME", "demo_user")
    monkeypatch.setattr(wrds_mod, "_pgpass_path", lambda: tmp_path / "no_existe.conf")
    monkeypatch.setattr(wrds_mod.sys.stdin, "isatty", lambda: False, raising=False)
    _patch_wrds(monkeypatch, FakeConnection())
    with pytest.raises(RuntimeError, match="create_pgpass_file"):
        wrds_mod.get_connection()


def test_sin_usuario_falla_con_instrucciones(monkeypatch, tmp_path):
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    monkeypatch.setattr(wrds_mod, "_env_file", lambda: tmp_path / ".env")
    with pytest.raises(RuntimeError, match="WRDS_USERNAME"):
        wrds_mod.get_connection()


def test_limit_se_fuerza_a_entero(monkeypatch):
    conn = FakeConnection()
    monkeypatch.setattr(wrds_mod, "get_connection", lambda: conn)
    wrds_mod.fetch_table("crsp.msf", limit="5")
    assert conn.calls[0].endswith("LIMIT 5")
