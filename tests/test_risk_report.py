"""Reporte de riesgo: ficheros, graficos, secciones vacias y formato.

El reporte es lo que lee el comite. Un CSV que no se escribe, un grafico que
revienta o una seccion que imprime "nan" en vez de "n/d" no rompen ningun
calculo, pero dejan al comite sin el dato o con uno que no se entiende.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from conftest import BASE_CONFIG
from risk_helpers import light_risk_config, synthetic_book

from sfc_tfsig import risk_report
from sfc_tfsig.config import config_from_dict
from sfc_tfsig.risk import analysis


@pytest.fixture(scope="module")
def politica():
    return light_risk_config()


@pytest.fixture(scope="module")
def completo(politica):
    m = synthetic_book()
    return analysis.analyze(
        m["weights"], m["sectors"], float(m["nav"].iloc[-1]), m["close"], "SPY", politica,
        strategy_nav=m["nav"], benchmark_nav=m["bench"], as_of=m["close"].index[-1],
        volume_wide=m["volume"], factor_scores=m["scores"], cost_bps=20.0,
        turnover=0.2, rebalances_per_year=12,
    )


@pytest.fixture(scope="module")
def previo(politica):
    m = synthetic_book()
    return analysis.analyze_book(m["weights"], m["sectors"], 100_000.0, m["close"], "SPY",
                                 politica, as_of=m["close"].index[-1])


def test_formato_de_valores_que_faltan():
    for fn in (risk_report._pct, risk_report._num, risk_report._money):
        assert fn(float("nan")) == "n/d"
        assert fn(None) == "n/d"
    assert risk_report._pct(0.1234) == "12.34%"
    assert risk_report._money(1234567.8) == "1,234,568"


def test_el_reporte_no_imprime_nan_ni_inf(completo, politica):
    text = risk_report.build_risk_report(completo, config_from_dict(BASE_CONFIG), politica)
    assert re.search(r"\b(nan|inf)\b", text.lower()) is None


def test_sin_historia_las_secciones_de_historia_se_omiten(previo, politica):
    """Antes de operar no hay backtest: esas secciones no aparecen vacias."""
    assert risk_report.backtest_section(previo) == ""
    assert risk_report.tail_section(previo) == ""
    assert risk_report.bootstrap_section(previo) == ""
    assert risk_report.history_section(previo) == ""
    assert risk_report.factor_section(previo) == ""
    text = risk_report.build_risk_report(previo, config_from_dict(BASE_CONFIG), politica)
    assert "Backtest del VaR" not in text
    assert "Semaforo de limites" in text


def test_la_comprobacion_previa_en_verde_lo_dice(previo):
    green = previo.limits.assign(status="OK")
    view = analysis.RiskAnalysis(**{**previo.__dict__, "limits": green, "status": "OK"})
    text = risk_report.render_pre_trade(view)
    assert "Todos los limites con dato en verde" in text
    assert "| Limite |" not in text


def test_se_escriben_reporte_y_tablas(completo, tmp_path, monkeypatch):
    monkeypatch.setattr(risk_report, "REPORT_DIR", tmp_path)
    path = risk_report.save_risk_report("# prueba\n")
    assert path.parent == tmp_path and path.read_text(encoding="utf-8") == "# prueba\n"

    artifacts = risk_report.save_risk_artifacts(completo)
    for label in ("limites", "contribuciones", "var_cartera", "var_estrategia", "backtest_var",
                  "montecarlo", "choques_covarianza", "var_condicional", "liquidez"):
        assert label in artifacts and artifacts[label].exists()
    limits = pd.read_csv(artifacts["limites"])
    assert set(limits["status"]) <= {"OK", "ALERTA", "EXCEDIDO", "SIN DATO"}
    # Las tablas indexadas por fecha u horizonte conservan el indice.
    assert "horizon_d" in pd.read_csv(artifacts["montecarlo"]).columns


def test_las_tablas_vacias_no_crean_ficheros(previo, tmp_path, monkeypatch):
    monkeypatch.setattr(risk_report, "REPORT_DIR", tmp_path)
    artifacts = risk_report.save_risk_artifacts(previo)
    for label in ("var_estrategia", "backtest_var", "montecarlo", "calibracion", "capacidad"):
        assert label not in artifacts


def test_se_genera_el_grafico(completo, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(risk_report, "REPORT_DIR", tmp_path)
    path = risk_report.save_risk_charts(completo)
    assert path is not None and path.exists() and path.stat().st_size > 10_000


def test_el_grafico_aguanta_un_analisis_sin_historia(previo, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(risk_report, "REPORT_DIR", tmp_path)
    assert risk_report.save_risk_charts(previo).exists()


def test_el_valor_de_un_limite_respeta_su_unidad():
    assert risk_report._limit_value(0.25, "pct") == "25.00%"
    assert risk_report._limit_value(2.5, "days") == "2.50"
    assert risk_report._limit_value(np.nan, "num") == "n/d"


def test_las_columnas_en_dolares_multiplican_por_el_nav():
    table = pd.DataFrame([{"method": "historico", "horizon_d": 1, "confidence": 0.99,
                           "var": 0.02, "es": 0.03}])
    row = risk_report._var_rows(table, 1_000_000.0)[0]
    assert row[3:] == ["2.00%", "20,000", "3.00%", "30,000"]


def test_un_limite_minimo_se_muestra_como_suelo(previo):
    text = risk_report.limits_section(previo)
    line = next(l for l in text.splitlines() if l.startswith("| Numero efectivo de nombres"))
    assert ">= 15.00" in line
    line = next(l for l in text.splitlines() if l.startswith("| Volatilidad ex-ante anual"))
    assert "<= 25.00%" in line
