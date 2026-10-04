from sfc_tfsig import console


def test_keep_awake_pide_y_libera(monkeypatch):
    llamadas = []

    class Kernel32:
        def SetThreadExecutionState(self, flags):
            llamadas.append(flags)
            return 0x80000000

    import ctypes

    monkeypatch.setattr(console.os, "name", "nt")
    monkeypatch.setattr(ctypes, "windll", type("W", (), {"kernel32": Kernel32()})(), raising=False)
    with console.keep_awake():
        assert llamadas == [console._ES_CONTINUOUS | console._ES_SYSTEM_REQUIRED | console._ES_DISPLAY_REQUIRED]
    assert llamadas[-1] == console._ES_CONTINUOUS


def test_keep_awake_fuera_de_windows_no_hace_nada(monkeypatch):
    monkeypatch.setattr(console.os, "name", "posix")
    with console.keep_awake():
        pass
