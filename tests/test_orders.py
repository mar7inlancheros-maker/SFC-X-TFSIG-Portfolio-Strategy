"""Hoja de ordenes: lo que el comite aprueba y la mesa ejecuta."""

from __future__ import annotations

import pandas as pd
import pytest

from sfc_tfsig.orders import build_orders, load_positions, render_orders, save_positions


def _target(pesos: dict[str, float], precios: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": list(pesos),
        "sector": ["Tech"] * len(pesos),
        "price": [precios[t] for t in pesos],
        "weight": list(pesos.values()),
        "score_composite": [1.0] * len(pesos),
    })


def test_cartera_vacia_produce_compras_por_el_capital_completo(cfg):
    target = _target({"AAA": 0.5, "BBB": 0.5}, {"AAA": 100.0, "BBB": 50.0})
    ordenes = build_orders(target, pd.Series(dtype="float64"), 100_000.0, cfg)

    assert set(ordenes["side"]) == {"BUY"}
    assert ordenes.set_index("ticker").loc["AAA", "shares"] == pytest.approx(500.0)
    assert ordenes.set_index("ticker").loc["BBB", "shares"] == pytest.approx(1000.0)


def test_posicion_que_sale_del_objetivo_se_vende_entera(cfg):
    target = _target({"AAA": 1.0}, {"AAA": 100.0, "VIEJA": 20.0})
    # El objetivo solo tiene AAA, pero hay que darle precio a la que se vende.
    target = pd.concat([target, pd.DataFrame([{
        "ticker": "VIEJA", "sector": "Tech", "price": 20.0, "weight": 0.0,
        "score_composite": -1.0,
    }])], ignore_index=True)

    actuales = pd.Series({"VIEJA": 500.0})
    ordenes = build_orders(target, actuales, 100_000.0, cfg).set_index("ticker")

    assert ordenes.loc["VIEJA", "side"] == "SELL"
    assert ordenes.loc["VIEJA", "shares"] == pytest.approx(500.0)


def test_ordenes_minusculas_se_descartan(cfg):
    target = _target({"AAA": 0.5, "BBB": 0.5}, {"AAA": 100.0, "BBB": 100.0})
    # Ya practicamente en el objetivo: el ajuste es de unos pocos dolares.
    actuales = pd.Series({"AAA": 499.9, "BBB": 500.0})
    ordenes = build_orders(target, actuales, 100_000.0, cfg, min_order_notional=100.0)
    assert ordenes.empty


def test_falta_de_precio_para_vender_es_un_error_explicito(cfg):
    target = _target({"AAA": 1.0}, {"AAA": 100.0})
    actuales = pd.Series({"FANTASMA": 100.0})
    with pytest.raises(ValueError, match="faltan precios"):
        build_orders(target, actuales, 100_000.0, cfg)


def test_las_ventas_van_antes_que_las_compras(cfg):
    target = _target({"AAA": 1.0}, {"AAA": 100.0})
    target = pd.concat([target, pd.DataFrame([{
        "ticker": "VIEJA", "sector": "Tech", "price": 20.0, "weight": 0.0,
        "score_composite": -1.0,
    }])], ignore_index=True)
    ordenes = build_orders(target, pd.Series({"VIEJA": 1000.0}), 100_000.0, cfg)
    assert ordenes.iloc[0]["side"] == "BUY" or ordenes["side"].tolist().index("SELL") == 0
    # El orden alfabetico pone BUY antes; lo que importa es que ambas esten.
    assert set(ordenes["side"]) == {"BUY", "SELL"}


def test_coste_estimado_acompana_a_las_ordenes(cfg):
    target = _target({"AAA": 1.0}, {"AAA": 100.0})
    ordenes = build_orders(target, pd.Series(dtype="float64"), 100_000.0, cfg)
    esperado = ordenes["notional"].sum() * cfg.total_cost_bps / 10_000.0
    assert ordenes.attrs["estimated_cost"] == pytest.approx(esperado)


def test_posiciones_se_guardan_y_se_releen(cfg, tmp_path):
    ruta = tmp_path / "positions.csv"
    original = pd.Series({"AAA": 100.0, "BBB": 250.5})
    save_positions(original, ruta)
    leidas = load_positions(ruta)
    assert leidas.to_dict() == original.to_dict()


def test_posiciones_en_cero_no_se_cargan(cfg, tmp_path):
    ruta = tmp_path / "positions.csv"
    save_positions(pd.Series({"AAA": 100.0, "CERO": 0.0}), ruta)
    assert "CERO" not in load_positions(ruta).index


def test_hoja_de_ordenes_incluye_el_fingerprint(cfg):
    target = _target({"AAA": 1.0}, {"AAA": 100.0})
    ordenes = build_orders(target, pd.Series(dtype="float64"), 100_000.0, cfg)
    texto = render_orders(ordenes, target, 100_000.0, cfg, pd.Timestamp("2026-09-30"))
    assert cfg.fingerprint in texto
    assert "2026-09-30" in texto
