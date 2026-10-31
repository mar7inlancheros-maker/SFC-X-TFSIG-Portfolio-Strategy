"""Fechas de sesion de las fichas: festivos de NYSE y de Toronto."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_engine.sessions import NYSECalendar, TSXCalendar, market_of, session_date


def _days(cal, year):
    return {d.date().isoformat() for d in cal.holidays(f"{year}-01-01", f"{year}-12-31")}


def test_festivos_de_nyse_con_sus_observancias():
    # Calendario publicado por NYSE para 2026 y 2027.
    assert _days(NYSECalendar(), 2026) == {
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03",
        "2026-09-07", "2026-11-26", "2026-12-25"}
    assert {"2027-06-18", "2027-07-05", "2027-12-24"} <= _days(NYSECalendar(), 2027)   # sabado -> viernes
    assert "2021-12-31" not in _days(NYSECalendar(), 2021)                              # Ano Nuevo en sabado


def test_festivos_de_toronto():
    assert _days(TSXCalendar(), 2026) == {
        "2026-01-01", "2026-02-16", "2026-04-03", "2026-05-18", "2026-07-01", "2026-08-03", "2026-09-07",
        "2026-10-12", "2026-12-25", "2026-12-28"}
    assert {"2027-12-27", "2027-12-28"} <= _days(TSXCalendar(), 2027)   # Navidad sabado, Boxing Day domingo


@pytest.mark.parametrize("ticker, market", [("ARES", "US"), ("BRK-B", "US"), ("ATZ.TO", "CA"), ("XYZ.V", "CA"),
                                            ("VOD.L", "OTHER")])
def test_bolsa_por_sufijo(ticker, market):
    assert market_of(ticker) == market


def test_la_entrada_es_la_sesion_siguiente_y_se_cuenta_desde_ella():
    d = pd.Timestamp("2026-10-30")                                       # viernes
    assert session_date("ARES", d, 0) == pd.Timestamp("2026-11-02")
    assert session_date("ARES", d, 20) == pd.Timestamp("2026-12-01")     # salta Thanksgiving
    assert session_date("ATZ.TO", d, 20) == pd.Timestamp("2026-11-30")   # Toronto abre en Thanksgiving de EE.UU.
    assert session_date("ARES", pd.Timestamp("2026-11-25"), 0) == pd.Timestamp("2026-11-27")
