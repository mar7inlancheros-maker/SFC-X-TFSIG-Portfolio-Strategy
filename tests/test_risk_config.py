"""Validacion de la politica de riesgo (config/risk.toml)."""

from __future__ import annotations

import copy
import tomllib

import pytest

from sfc_tfsig.config import (
    ConfigError,
    load_config,
    load_risk_config,
    risk_config_from_dict,
)
from sfc_tfsig.paths import RISK_CONFIG


@pytest.fixture
def data():
    with RISK_CONFIG.open("rb") as fh:
        return tomllib.load(fh)


def _with(data, path: str, value):
    out = copy.deepcopy(data)
    node = out
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return out


def test_la_politica_del_repo_es_valida_y_de_tipo_riesgo():
    cfg = load_risk_config()
    assert cfg.kind == "risk"
    assert len(cfg.fingerprint) == 16


def test_su_fingerprint_no_toca_el_de_la_estrategia(data):
    before = load_config().fingerprint
    risk_config_from_dict(_with(data, "limits.max_beta", 1.5))
    assert load_config().fingerprint == before


@pytest.mark.parametrize("path, value, message", [
    ("var.confidence", [99], "0.99, no 99"),
    ("var.ewma_lambda", 1.2, "RiskMetrics"),
    ("var.garch.n_paths", 100, "trayectorias"),
    ("montecarlo.parametric.t_dof", 2, "varianza"),
    ("montecarlo.bootstrap.return_haircut", 2.0, "ANUAL"),
    ("exposure.participation", 0.0, "participation"),
    ("liquidity.stress_vol_mult", 0.5, "abaratar"),
    ("stress.history_start", "ayer", "AAAA-MM-DD"),
    ("limits.max_beta", -1.0, "positivo"),
])
def test_valores_imposibles_se_rechazan_con_mensaje(data, path, value, message):
    with pytest.raises(ConfigError, match=message):
        risk_config_from_dict(_with(data, path, value))


def test_escenario_historico_con_fechas_al_reves(data):
    bad = copy.deepcopy(data)
    bad["stress"]["historical"][0]["start"], bad["stress"]["historical"][0]["end"] = (
        bad["stress"]["historical"][0]["end"], bad["stress"]["historical"][0]["start"])
    with pytest.raises(ConfigError, match="anterior"):
        risk_config_from_dict(bad)


def test_factor_desconocido_en_un_escenario(data):
    bad = copy.deepcopy(data)
    bad["stress"]["hypothetical"][0]["factors"] = {"carry": -0.1}
    with pytest.raises(ConfigError, match="carry"):
        risk_config_from_dict(bad)


def test_falta_una_seccion(data):
    bad = copy.deepcopy(data)
    del bad["liquidity"]
    with pytest.raises(ConfigError, match="liquidity"):
        risk_config_from_dict(bad)


def test_replace_valida_como_riesgo_y_no_como_estrategia():
    cfg = load_risk_config()
    other = cfg.replace(**{"montecarlo.bootstrap.n_paths": 1000})
    assert other.kind == "risk"
    assert other.fingerprint != cfg.fingerprint


def test_escenario_historico_de_un_solo_dia_se_rechaza(data):
    bad = copy.deepcopy(data)
    bad["stress"]["historical"][0]["end"] = bad["stress"]["historical"][0]["start"]
    with pytest.raises(ConfigError, match="anterior"):
        risk_config_from_dict(bad)


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_correlacion_de_crisis_fuera_de_0_1_se_rechaza(data, value):
    bad = copy.deepcopy(data)
    bad["stress"]["covariance"][0]["corr_blend"] = value
    with pytest.raises(ConfigError, match="corr_blend"):
        risk_config_from_dict(bad)
