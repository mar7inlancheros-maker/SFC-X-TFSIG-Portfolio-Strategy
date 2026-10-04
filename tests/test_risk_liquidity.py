"""Coste de liquidacion y capacidad con la ley de la raiz cuadrada."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig.risk import liquidity


def _libro():
    w = pd.Series({"A": 0.5, "B": 0.5})
    adv = pd.Series({"A": 1_000_000.0, "B": 4_000_000.0})
    vol = pd.Series({"A": 0.02, "B": 0.02})
    return w, adv, vol


def test_el_impacto_sigue_la_raiz_del_tamano():
    w, adv, vol = _libro()
    detail, _ = liquidity.liquidation_cost(w, 400_000.0, adv, vol, cost_bps=0.0, impact_coef=1.0)
    # A: 200k contra 1M diario -> 0.02 x raiz(0.2); B: 200k contra 4M -> 0.02 x raiz(0.05).
    assert detail.loc["A", "impact"] == pytest.approx(0.02 * np.sqrt(0.2))
    assert detail.loc["B", "impact"] == pytest.approx(0.02 * np.sqrt(0.05))


def test_la_crisis_encarece_la_salida():
    w, adv, vol = _libro()
    _, normal = liquidity.liquidation_cost(w, 400_000.0, adv, vol, cost_bps=20.0, impact_coef=1.0)
    _, crisis = liquidity.liquidation_cost(w, 400_000.0, adv, vol, cost_bps=20.0, impact_coef=1.0,
                                           adv_mult=0.5, vol_mult=2.0, spread_mult=3.0)
    assert crisis > 2.0 * normal
    # El coste fijo se cobra siempre: 20 bps de todo el libro como minimo.
    assert normal >= 0.002


def test_sin_volumen_solo_se_cobra_el_coste_fijo():
    w = pd.Series({"A": 1.0})
    _, cost = liquidity.liquidation_cost(w, 1e6, pd.Series({"A": np.nan}), pd.Series({"A": 0.02}),
                                         cost_bps=20.0, impact_coef=1.0)
    assert cost == pytest.approx(0.002)


def test_el_impacto_por_dolar_crece_con_la_raiz_del_nav():
    w, adv, vol = _libro()
    curve = liquidity.capacity_curve(w, adv, vol, turnover=0.2, rebalances_per_year=12,
                                     cost_bps=20.0, impact_coef=1.0, navs=[1e6, 4e6])
    # Cuadruplicar el NAV duplica el impacto por dolar.
    assert curve.loc[4e6, "impact_annual"] == pytest.approx(2.0 * curve.loc[1e6, "impact_annual"])
    # El coste fijo por dolar no depende del tamano.
    assert curve.loc[4e6, "fixed_annual"] == pytest.approx(curve.loc[1e6, "fixed_annual"])


def test_el_nav_del_presupuesto_cuadra_con_la_curva():
    w, adv, vol = _libro()
    nav = liquidity.capacity_at_budget(w, adv, vol, turnover=0.2, rebalances_per_year=12,
                                       impact_coef=1.0, budget=0.01)
    curve = liquidity.capacity_curve(w, adv, vol, turnover=0.2, rebalances_per_year=12,
                                     cost_bps=0.0, impact_coef=1.0, navs=[nav])
    assert curve["impact_annual"].iloc[0] == pytest.approx(0.01)


def test_en_crisis_el_coste_fijo_se_multiplica():
    """Sin impacto, la crisis cuesta exactamente el multiplo del coste fijo."""
    w, adv, vol = _libro()
    _, normal = liquidity.liquidation_cost(w, 400_000.0, adv, vol, cost_bps=20.0, impact_coef=0.0)
    _, crisis = liquidity.liquidation_cost(w, 400_000.0, adv, vol, cost_bps=20.0, impact_coef=0.0,
                                           spread_mult=3.0)
    assert crisis == pytest.approx(3.0 * normal)


def test_la_capacidad_opera_dos_veces_la_rotacion_en_cada_rebalanceo():
    """Rotacion = nocional operado / NAV / 2, igual que en el backtest: cada
    rebalanceo opera 2 x rotacion del NAV y paga el coste fijo sobre eso."""
    w, adv, vol = _libro()
    curve = liquidity.capacity_curve(w, adv, vol, turnover=0.2, rebalances_per_year=12,
                                     cost_bps=20.0, impact_coef=1.0, navs=[1e6])
    assert curve.loc[1e6, "fixed_annual"] == pytest.approx(2 * 0.2 * 12 * 20 / 10_000)
