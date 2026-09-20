"""Construccion de cartera: seleccion, pesos y restricciones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig.portfolio import apply_caps, build_portfolio, select_names, turnover


def _scored(n=40, sectors=None):
    sectors = sectors or (["Tech"] * 20 + ["Banca"] * 20)
    return pd.DataFrame({
        "date": pd.to_datetime(["2023-01-31"] * n),
        "ticker": [f"T{i:02d}" for i in range(n)],
        "sector": sectors[:n],
        "price": np.linspace(10, 200, n),
        "volatility": np.linspace(0.15, 0.60, n),
        "score_composite": np.linspace(2.0, -2.0, n),  # T00 el mejor
    })


def test_seleccion_toma_los_mejores_sin_cartera_previa():
    seleccion = select_names(_scored(), n_positions=10, buffer_rank=15)
    assert len(seleccion) == 10
    assert seleccion["ticker"].tolist()[0] == "T00"


def test_amortiguador_evita_vender_por_un_puesto():
    """Una posicion en el puesto 12 se mantiene si el buffer llega a 15."""
    scored = _scored()
    seleccion = select_names(scored, n_positions=10, buffer_rank=15, held={"T11"})
    assert "T11" in seleccion["ticker"].tolist()
    # Y desplaza justo a uno de los ultimos entrantes, no a los mejores.
    assert "T00" in seleccion["ticker"].tolist()


def test_amortiguador_no_retiene_lo_que_se_deterioro_de_verdad():
    seleccion = select_names(_scored(), n_positions=10, buffer_rank=15, held={"T30"})
    assert "T30" not in seleccion["ticker"].tolist()


def test_topes_por_nombre_y_sector_se_respetan(cfg):
    scored = _scored()
    cartera = build_portfolio(scored, cfg)
    assert cartera["weight"].max() <= float(cfg.get("portfolio.max_weight")) + 1e-9
    por_sector = cartera.groupby("sector")["weight"].sum()
    assert por_sector.max() <= float(cfg.get("portfolio.max_sector_w")) + 1e-9


def test_pesos_suman_el_capital_invertido(make_cfg):
    """Con sectores suficientes, se invierte todo menos el colchon de caja."""
    cfg = make_cfg(**{"portfolio.cash_buffer": 0.03})
    sectores = ["Tech", "Banca", "Salud", "Energia", "Consumo"] * 8
    cartera = build_portfolio(_scored(sectors=sectores), cfg)
    assert cartera["weight"].sum() == pytest.approx(0.97, abs=1e-9)


def test_tope_sectorial_imposible_deja_caja_en_vez_de_incumplirse(make_cfg):
    """Dos sectores y techo del 40%: 2 x 40% = 80%. El 20% restante es caja.

    Lo que NO puede pasar es que el modelo devuelva 50/50 y se salte el limite.
    """
    cfg = make_cfg(**{"portfolio.cash_buffer": 0.0})
    cartera = build_portfolio(_scored(sectors=["Tech"] * 20 + ["Banca"] * 20), cfg)
    por_sector = cartera.groupby("sector")["weight"].sum()
    assert por_sector.max() <= 0.40 + 1e-9
    assert cartera["weight"].sum() == pytest.approx(0.80, abs=1e-9)


def test_tope_sectorial_redistribuye_sin_perder_capital():
    """Todo de un sector: el exceso tiene que ir a alguna parte, no evaporarse."""
    pesos = pd.Series([0.4, 0.3, 0.2, 0.1], index=list("ABCD"))
    sectores = pd.Series(["Energia", "Energia", "Tech", "Salud"], index=list("ABCD"))
    acotados = apply_caps(pesos, sectores, max_weight=0.30, max_sector_weight=0.40)
    assert acotados.sum() == pytest.approx(1.0)
    assert acotados.max() <= 0.30 + 1e-9
    assert acotados.groupby(sectores).sum().max() <= 0.40 + 1e-9


def test_ponderacion_equal_reparte_igual(make_cfg):
    cfg = make_cfg(**{"portfolio.weighting": "equal", "portfolio.max_sector_w": 1.0,
                      "portfolio.max_weight": 0.5})
    cartera = build_portfolio(_scored(), cfg)
    assert cartera["weight"].std() == pytest.approx(0.0, abs=1e-12)


def test_inverse_vol_da_mas_peso_a_lo_menos_volatil(make_cfg):
    cfg = make_cfg(**{"portfolio.weighting": "inverse_vol", "portfolio.max_sector_w": 1.0,
                      "portfolio.max_weight": 0.5})
    cartera = build_portfolio(_scored(), cfg).set_index("ticker")
    # T00 (vol 0.15) debe pesar mas que T09 (vol mas alta) entre los elegidos.
    assert cartera.loc["T00", "weight"] > cartera.loc["T09", "weight"]


def test_score_tilt_nunca_produce_peso_negativo(make_cfg):
    cfg = make_cfg(**{"portfolio.weighting": "score_tilt", "portfolio.n_positions": 30,
                      "portfolio.buffer_rank": 35, "portfolio.max_sector_w": 1.0,
                      "portfolio.max_weight": 0.5})
    cartera = build_portfolio(_scored(), cfg)
    assert (cartera["weight"] > 0).all()


def test_turnover_completo_es_uno():
    antes = pd.Series({"A": 0.5, "B": 0.5})
    despues = pd.Series({"C": 0.5, "D": 0.5})
    assert turnover(antes, despues) == pytest.approx(1.0)


def test_turnover_sin_cambios_es_cero():
    pesos = pd.Series({"A": 0.5, "B": 0.5})
    assert turnover(pesos, pesos) == pytest.approx(0.0)
