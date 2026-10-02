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


def test_robustez_incluye_sensibilidad_al_coste_de_prestamo():
    # La fila de borrow_cost pedida por el comite: 0,25% / 1% / 3% anual sobre
    # el nocional corto, sin cambiar el default del YAML. Mas prestamo, menos
    # Sharpe: los cortos pagan mas y nada mas cambia.
    import numpy as np

    from quant_engine.data import cleaning
    from quant_engine.data.loader import MarketData
    from quant_engine.robustness import run_robustness
    from quant_engine.settings import build_settings

    rng = np.random.default_rng(5)
    cal = pd.bdate_range("2021-01-04", periods=900)
    names = ["L1", "L2", "S1", "S2", "SPY"]
    data = {}
    for t in names:
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(cal))))
        c = pd.Series(close, index=cal)
        data[t] = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6})
    clean = cleaning.align(MarketData(ohlcv=data), "SPY")
    settings = build_settings(("L1", "L2"), ("S1", "S2"), construction="equal_weight", lookback="1y")
    result = SimpleNamespace(settings=settings, clean=clean, longs=["L1", "L2"], shorts=["S1", "S2"],
                             as_of=cal[-1], eval_start=cal[-1] - pd.DateOffset(years=1),
                             rank_stability={})
    table = run_robustness(result)["table"]
    rows = table[table["dimension"] == "borrow_cost"].set_index("variant")
    assert list(rows.index) == ["0.25%", "1.00%", "3.00%"]
    assert rows.loc["0.25%", "sharpe"] > rows.loc["1.00%", "sharpe"] > rows.loc["3.00%", "sharpe"]
    assert settings.borrow_cost == 0.0025
