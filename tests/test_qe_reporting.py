"""Reporte del motor long/short: que dice y que no dice cada salida.

Se prueba con un resultado de juguete (SimpleNamespace) con solo los campos que
lee cada funcion: el reporte es presentacion pura y no deberia necesitar red.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from rich.console import Console

from quant_engine.backtest.engine import MODE_A_LABEL
from quant_engine.reporting import export, terminal


def _result(historical=None, robustness=None):
    agreement = pd.DataFrame({"research": ["LONG", "SHORT"], "quant_score": [0.8, 0.3],
                              "agreement": ["STRONG AGREEMENT", "SIGNAL CONFLICT"]}, index=["AAA", "BBB"])
    return SimpleNamespace(
        primary={"method": "risk_parity",
                 "summary": {"sharpe": 1.234, "max_drawdown": -0.1729, "cagr": 0.3336},
                 "ex_ante": {"beta": 0.05, "net": 0.0, "vol_annual": 0.18,
                             "risk_contribution": pd.Series({"AAA": 0.6, "BBB": 0.4})}},
        agreement=agreement,
        scores=pd.DataFrame({"quant_score": [0.8, 0.3]}, index=["AAA", "BBB"]),
        spread={"mean_annual": 0.12, "p_value": 0.03},
        robustness=robustness or {},
        historical=historical,
        warnings=[],
        settings=SimpleNamespace(get=lambda key, default=None: default),
    )


def _text(fn, r) -> str:
    console = Console(record=True, width=200)
    fn(console, r)
    return console.export_text()


def test_resumen_final_no_usa_la_trayectoria_del_modo_a():
    # Auditoria #8. La seccion 21 de la especificacion: la cesta de hoy aplicada
    # al pasado no es evidencia. El resumen citaba su Sharpe (1.23) y el rango
    # de Sharpe de la robustez, que tambien sale del modo A.
    r = _result(robustness={"sharpe_min": 0.815, "sharpe_max": 1.395})
    text = _text(terminal.summary, r)
    assert "1.23" not in text
    assert "0.81" not in text and "1.40" not in text
    assert "not evidence" in text


def test_resumen_final_cita_el_modo_b_cuando_existe():
    hist = {"summary": {"cagr": 0.08, "sharpe": 0.61, "max_drawdown": -0.22, "information_ratio": 0.3}}
    text = _text(terminal.summary, _result(historical=hist))
    assert "Mode B" in text and "0.61" in text


def test_analisis_de_riesgo_rotula_el_modo_a():
    text = _text(terminal.risk_analysis, _result())
    assert "MODE A - HYPOTHETICAL PATH" in text


def test_json_rotula_cada_trayectoria_del_modo_a():
    portfolios = {"risk_parity": {"status": "ok", "summary": {"sharpe": 1.2}, "ex_ante": {}}}
    out = export.portfolios_json(portfolios)
    assert out["risk_parity"]["path"] == MODE_A_LABEL
    assert export.robustness_json({"table": pd.DataFrame()})["path"] == MODE_A_LABEL
    assert export.robustness_json({}) == {}
