"""Linea de comandos: los subcomandos del modelo multifactor no dependen del motor long/short."""

from __future__ import annotations

import sys

from sfc_tfsig import cli


def test_el_parser_funciona_sin_las_dependencias_del_motor_long_short(monkeypatch):
    # Un None en sys.modules hace que `import quant_engine.app` lance ImportError,
    # que es lo que pasa sin el extra [quant] (rich, statsmodels...).
    monkeypatch.setitem(sys.modules, "quant_engine.app", None)
    args = cli.build_parser().parse_args(["backtest", "--sin-validacion"])
    assert args.comando == "backtest"
    assert args.sin_validacion is True


def test_sin_subcomando_y_sin_motor_long_short_sale_con_mensaje(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "quant_engine.app", None)
    code = cli.main([])
    assert code == 2
    assert "[quant]" in capsys.readouterr().err
