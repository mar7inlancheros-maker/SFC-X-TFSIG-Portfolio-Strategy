"""Los comandos `riesgo` y `ordenes` de punta a punta, sin red y sin cache.

Los tests de cada modulo prueban las piezas; estos prueban que la CLI las
conecta: que los argumentos que pasa `cmd_riesgo` existen, que el reporte y los
CSV se escriben, que `--estricto` cambia el codigo de salida y que la hoja de
ordenes lleva el control de riesgo previo. Todo lo que tocaria disco o red se
sustituye por datos sinteticos y un directorio temporal.
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import BASE_CONFIG
from risk_helpers import light_risk_config, synthetic_strategy

from sfc_tfsig import cli, orders, risk_report


@pytest.fixture(scope="module")
def estrategia():
    return synthetic_strategy(BASE_CONFIG)


@pytest.fixture
def entorno(estrategia, tmp_path, monkeypatch):
    cfg, scored, close, volume = estrategia
    monkeypatch.setattr(cli, "_load", lambda path: cfg)
    monkeypatch.setattr(cli, "load_risk_config", lambda path=None: light_risk_config())
    monkeypatch.setattr(cli, "_require_panel", lambda cfg, rebuild, progress=True: scored)
    monkeypatch.setattr(cli, "_price_frames", lambda scored, cfg: (close, volume))
    monkeypatch.setattr(cli, "ensure_dirs", lambda: None)
    monkeypatch.setattr(cli, "load_positions", lambda: pd.Series(dtype="float64"))
    monkeypatch.setattr(risk_report, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(orders, "REPORT_DIR", tmp_path)

    def no_network(*args, **kwargs):
        raise AssertionError("con --sin-red no se descarga nada")

    monkeypatch.setattr(cli, "_stress_prices", no_network)
    return tmp_path


def _report(tmp_path) -> str:
    reports = sorted(p for p in tmp_path.glob("riesgo_*.md"))
    assert len(reports) == 1
    return reports[0].read_text(encoding="utf-8")


def test_riesgo_escribe_reporte_y_tablas(entorno, capsys):
    assert cli.main(["riesgo", "--sin-red"]) == 0
    text = _report(entorno)
    text.encode("ascii")
    for section in ("Semaforo de limites", "Modelo factorial", "Volatilidad condicional",
                    "Backtest del VaR", "Monte Carlo", "VaR estresado", "Calibracion",
                    "Capacidad", "Limitaciones"):
        assert section in text
    written = {p.name.split("_", 3)[-1] for p in entorno.glob("riesgo_*.csv")}
    for label in ("limites.csv", "contribuciones.csv", "var_condicional.csv",
                  "calibracion.csv", "capacidad.csv", "factores.csv"):
        assert label in written
    assert "reporte de riesgo:" in capsys.readouterr().out


def test_estricto_devuelve_3_con_un_limite_excedido(entorno, monkeypatch):
    monkeypatch.setattr(cli, "load_risk_config", lambda path=None: light_risk_config(max_beta=0.01))
    assert cli.main(["riesgo", "--sin-red", "--estricto"]) == 3
    # Sin --estricto, el mismo exceso no cambia el codigo de salida.
    assert cli.main(["riesgo", "--sin-red"]) == 0


def test_con_red_usa_la_historia_larga_del_estres(entorno, estrategia, monkeypatch):
    _, _, close, _ = estrategia
    calls = []

    def stress_prices(tickers, risk_cfg):
        calls.append(sorted(tickers))
        return close

    monkeypatch.setattr(cli, "_stress_prices", stress_prices)
    assert cli.main(["riesgo"]) == 0
    assert len(calls) == 1 and "SPY" in calls[0]


def test_simulaciones_cambia_las_trayectorias_del_bootstrap(entorno):
    assert cli.main(["riesgo", "--sin-red", "--simulaciones", "600"]) == 0
    assert "600 trayectorias" in _report(entorno)


def test_ordenes_lleva_el_control_de_riesgo_previo(entorno):
    assert cli.main(["ordenes", "--capital", "100000"]) == 0
    sheets = list(entorno.glob("ordenes_*.md"))
    assert len(sheets) == 1
    text = sheets[0].read_text(encoding="utf-8")
    assert "Riesgo de la cartera objetivo" in text
    assert "Mayor fuente de riesgo" in text
    text.encode("ascii")


def test_ordenes_sin_riesgo_lo_omite(entorno):
    assert cli.main(["ordenes", "--capital", "100000", "--sin-riesgo"]) == 0
    text = next(entorno.glob("ordenes_*.md")).read_text(encoding="utf-8")
    assert "Riesgo de la cartera objetivo" not in text


def test_politica_de_riesgo_invalida_sale_con_codigo_2(entorno, monkeypatch, capsys):
    from sfc_tfsig.config import ConfigError

    def broken(path=None):
        raise ConfigError("[limits].max_beta = -1.0: un limite debe ser positivo")

    monkeypatch.setattr(cli, "load_risk_config", broken)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["riesgo", "--sin-red"])
    assert exit_info.value.code == 2
    assert "politica de riesgo" in capsys.readouterr().err
