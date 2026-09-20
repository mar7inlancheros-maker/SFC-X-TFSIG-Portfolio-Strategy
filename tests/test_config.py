"""La configuracion: validar antes de tocar datos, y firmar lo que se ejecuta.

Un typo en el TOML no puede convertirse en una cartera rara que nadie sabe
explicar en el comite.
"""

from __future__ import annotations

import copy

import pytest

from sfc_tfsig.config import ConfigError, config_from_dict, load_config
from sfc_tfsig.paths import DEFAULT_CONFIG

from conftest import BASE_CONFIG


def _con(**cambios):
    """BASE_CONFIG con claves sustituidas por ruta con puntos."""
    data = copy.deepcopy(BASE_CONFIG)
    for ruta, valor in cambios.items():
        partes = ruta.split(".")
        nodo = data
        for parte in partes[:-1]:
            nodo = nodo[parte]
        nodo[partes[-1]] = valor
    return data


def test_la_configuracion_del_repo_es_valida():
    """El TOML que viene en el repo tiene que cargar sin tocarlo."""
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.get("meta.name")
    assert len(cfg.fingerprint) == 16


def test_fingerprint_cambia_si_cambia_una_decision_de_inversion():
    base = config_from_dict(BASE_CONFIG)
    # El buffer sube con las posiciones: la validacion exige buffer >= n, y eso
    # es correcto -- 25 posiciones con un amortiguador de 15 seria incoherente.
    otra = base.replace(**{"portfolio.n_positions": 25, "portfolio.buffer_rank": 35})
    assert base.fingerprint != otra.fingerprint


def test_fingerprint_es_estable_entre_cargas_identicas():
    assert config_from_dict(BASE_CONFIG).fingerprint == config_from_dict(BASE_CONFIG).fingerprint


def test_pesos_de_factores_se_normalizan_a_uno():
    cfg = config_from_dict(_con(**{"factors.weights": {"value": 2.0, "quality": 2.0}}))
    assert sum(cfg.factor_weights.values()) == pytest.approx(1.0)
    assert cfg.factor_weights["value"] == pytest.approx(0.5)


def test_un_peso_cero_apaga_el_factor():
    cfg = config_from_dict(_con(**{
        "factors.weights": {"value": 1.0, "quality": 1.0, "momentum": 0.0, "lowvol": 0.0}
    }))
    assert set(cfg.factor_weights) == {"value", "quality"}


def test_factor_desconocido_se_rechaza_con_los_conocidos_en_el_mensaje():
    with pytest.raises(ConfigError, match="momentom"):
        config_from_dict(_con(**{"factors.weights": {"momentom": 1.0}}))


def test_todos_los_pesos_a_cero_es_no_tener_modelo():
    with pytest.raises(ConfigError, match="no hay modelo"):
        config_from_dict(_con(**{
            "factors.weights": {"value": 0.0, "quality": 0.0, "momentum": 0.0, "lowvol": 0.0}
        }))


def test_peso_negativo_se_rechaza():
    with pytest.raises(ConfigError, match="negativo"):
        config_from_dict(_con(**{"factors.weights": {"value": -1.0, "quality": 1.0}}))


def test_topes_imposibles_se_detectan_antes_de_correr_nada():
    """max_weight x n_positions < 1 impide invertir el capital completo."""
    with pytest.raises(ConfigError, match="impiden invertir"):
        config_from_dict(_con(**{"portfolio.max_weight": 0.03,
                                 "portfolio.n_positions": 20}))


def test_buffer_menor_que_las_posiciones_se_rechaza():
    with pytest.raises(ConfigError, match="buffer_rank"):
        config_from_dict(_con(**{"portfolio.buffer_rank": 5, "portfolio.n_positions": 20}))


def test_retardo_de_ejecucion_negativo_es_look_ahead():
    with pytest.raises(ConfigError, match="look-ahead"):
        config_from_dict(_con(**{"calendar.execution_lag_d": -1}))


def test_momentum_con_salto_mayor_que_la_ventana_se_rechaza():
    with pytest.raises(ConfigError, match="lookback_m"):
        config_from_dict(_con(**{"factors.momentum": {"lookback_m": 3, "skip_m": 6}}))


def test_historia_insuficiente_para_momentum_se_rechaza():
    with pytest.raises(ConfigError, match="momentum 12-1"):
        config_from_dict(_con(**{"universe.min_history_months": 6}))


def test_coste_negativo_se_rechaza():
    with pytest.raises(ConfigError, match="no paga dividendos"):
        config_from_dict(_con(**{"costs.spread_bps": -5.0}))


def test_benchmark_vacio_se_rechaza():
    with pytest.raises(ConfigError, match="sin referencia no hay alfa"):
        config_from_dict(_con(**{"backtest.benchmark": "  "}))


def test_clave_inexistente_da_error_con_la_ruta_completa():
    cfg = config_from_dict(BASE_CONFIG)
    with pytest.raises(ConfigError, match="portfolio.inventada"):
        cfg.get("portfolio.inventada")


def test_valor_por_defecto_cuando_se_pide():
    cfg = config_from_dict(BASE_CONFIG)
    assert cfg.get("portfolio.inventada", 42) == 42


def test_coste_total_por_lado_suma_las_tres_patas():
    cfg = config_from_dict(BASE_CONFIG)
    assert cfg.total_cost_bps == pytest.approx(20.0)


def test_la_configuracion_es_inmutable():
    cfg = config_from_dict(BASE_CONFIG)
    with pytest.raises(Exception):
        cfg.fingerprint = "otro"
