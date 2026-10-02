"""Ficheros del reporte: dos corridas seguidas no se pisan."""

from __future__ import annotations

import pandas as pd

from sfc_tfsig import orders as orders_mod, report as report_mod, risk_report as risk_report_mod


def test_dos_reportes_seguidos_no_se_pisan(tmp_path, monkeypatch):
    # Con la hora al minuto, la corrida trimestral sobrescribio los ficheros
    # de la mensual lanzada justo antes.
    monkeypatch.setattr(report_mod, "REPORT_DIR", tmp_path)
    primero = report_mod.save_report("mensual")
    segundo = report_mod.save_report("trimestral")
    assert primero != segundo
    assert primero.read_text(encoding="utf-8") == "mensual"
    assert segundo.read_text(encoding="utf-8") == "trimestral"


def test_el_reporte_de_riesgo_tampoco_se_pisa(tmp_path, monkeypatch):
    monkeypatch.setattr(risk_report_mod, "REPORT_DIR", tmp_path)
    primero = risk_report_mod.save_risk_report("a")
    segundo = risk_report_mod.save_risk_report("b")
    assert primero != segundo


def test_las_ordenes_tampoco_se_pisan(tmp_path, monkeypatch):
    monkeypatch.setattr(orders_mod, "REPORT_DIR", tmp_path)
    vacio = pd.DataFrame(columns=["ticker"])
    primero = orders_mod.save_orders(vacio, "a", pd.Timestamp("2026-09-30"))
    segundo = orders_mod.save_orders(vacio, "b", pd.Timestamp("2026-09-30"))
    assert primero["markdown"] != segundo["markdown"]
