import pandas as pd

from sfc_tfsig.data import cache, wrds as wrds_mod


class FakeConnection:
    def __init__(self, username):
        self.username = username
        self.calls = []

    def raw_sql(self, query):
        self.calls.append(query)
        return pd.DataFrame({"date": ["2024-01-02"], "permno": [10001]})


def test_get_connection_uses_wrds_username_from_env(monkeypatch):
    captured = {}

    def fake_connection(**kwargs):
        captured.update(kwargs)
        return FakeConnection(kwargs["wrds_username"])

    monkeypatch.setenv("WRDS_USERNAME", "demo_user")
    monkeypatch.setattr(wrds_mod, "_import_wrds", lambda: type("W", (), {"Connection": staticmethod(fake_connection)})())

    conn = wrds_mod.get_connection()

    assert conn.username == "demo_user"
    assert captured["wrds_username"] == "demo_user"


def test_fetch_table_uses_cache_when_available(monkeypatch):
    expected = pd.DataFrame({"permno": [10001]})
    captured = {"sql": False}

    class FakeConn:
        def raw_sql(self, query):
            captured["sql"] = True
            return pd.DataFrame({"permno": [99999]})

    monkeypatch.setattr(wrds_mod.cache, "read_frame", lambda *args, **kwargs: expected)
    monkeypatch.setattr(wrds_mod, "get_connection", lambda: FakeConn())

    result = wrds_mod.fetch_table("crsp.msf", cache_name="wrds_test", limit=10)

    assert result.equals(expected)
    assert captured["sql"] is False


def test_fetch_table_caches_when_missing(monkeypatch):
    fetched = pd.DataFrame({"date": ["2024-01-02"], "permno": [10001]})
    written = {}

    def fake_write(name, df):
        written["name"] = name
        written["df"] = df

    monkeypatch.setattr(wrds_mod.cache, "read_frame", lambda *args, **kwargs: None)
    monkeypatch.setattr(wrds_mod.cache, "write_frame", fake_write)
    monkeypatch.setattr(wrds_mod, "get_connection", lambda: FakeConnection("demo_user"))

    result = wrds_mod.fetch_table("crsp.msf", cache_name="wrds_test", limit=10)

    assert result.equals(fetched)
    assert written["name"] == "wrds_test"
    assert written["df"].equals(fetched)


def test_env_value_with_spaces_and_quotes_is_normalized(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text('WRDS_USERNAME = "demo_user"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    value = wrds_mod._read_env_value("WRDS_USERNAME")

    assert value == "demo_user"
