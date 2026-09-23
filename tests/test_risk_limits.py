"""Limites de vigilancia: estados, direccion y claves desconocidas."""

from __future__ import annotations

import copy
import tomllib

import pytest

from sfc_tfsig.config import ConfigError, risk_config_from_dict
from sfc_tfsig.paths import RISK_CONFIG
from sfc_tfsig.risk import limits


@pytest.fixture
def risk_data():
    with RISK_CONFIG.open("rb") as fh:
        return tomllib.load(fh)


def _cfg(data, **limit_values):
    data = copy.deepcopy(data)
    data["limits"] = {"warning_fraction": 0.8, **limit_values}
    return risk_config_from_dict(data)


def test_cada_limite_del_toml_tiene_medida(risk_data):
    limits.validate_limit_keys(risk_config_from_dict(risk_data))


def test_un_limite_sin_medida_se_rechaza_antes_de_calcular(risk_data):
    cfg = _cfg(risk_data, max_ex_ante_vol=0.25, max_sharpe_invertido=1.0)
    with pytest.raises(ConfigError, match="max_sharpe_invertido"):
        limits.validate_limit_keys(cfg)


@pytest.mark.parametrize("value, status", [(0.10, "OK"), (0.21, "ALERTA"), (0.26, "EXCEDIDO")])
def test_limite_maximo(risk_data, value, status):
    cfg = _cfg(risk_data, max_ex_ante_vol=0.25)
    checks = limits.check_limits({"ex_ante_vol": value}, cfg)
    assert checks.iloc[0]["status"] == status


@pytest.mark.parametrize("value, status", [(30.0, "OK"), (17.0, "ALERTA"), (12.0, "EXCEDIDO")])
def test_limite_minimo_funciona_al_reves(risk_data, value, status):
    cfg = _cfg(risk_data, min_effective_names=15.0)
    checks = limits.check_limits({"effective_names": value}, cfg)
    assert checks.iloc[0]["status"] == status


def test_medida_ausente_es_sin_dato_y_no_ok(risk_data):
    cfg = _cfg(risk_data, max_current_drawdown=0.25)
    checks = limits.check_limits({}, cfg)
    assert checks.iloc[0]["status"] == "SIN DATO"
    assert limits.overall_status(checks) == "SIN DATO"


def test_el_estado_general_es_el_peor(risk_data):
    cfg = _cfg(risk_data, max_ex_ante_vol=0.25, max_beta=1.3)
    checks = limits.check_limits({"ex_ante_vol": 0.30, "beta": 1.0}, cfg)
    assert limits.overall_status(checks) == "EXCEDIDO"
    assert checks.set_index("limit").loc["max_ex_ante_vol", "action"] != ""
    assert checks.set_index("limit").loc["max_beta", "action"] == ""
