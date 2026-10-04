"""Coste de salir: liquidacion en estres y capacidad con impacto de mercado.

`exposure.liquidity` cuenta DIAS para deshacer cada posicion. Esto pone precio
a esos dias. Dos preguntas:

1. **Cuanto cuesta liquidar todo el libro manana**, en condiciones normales y
   en una crisis (volumen a la mitad, volatilidad doble, spread triple). Es la
   perdida que se anade a la del mercado cuando hay que vender a la fuerza.
2. **Hasta que capital aguanta la estrategia.** El backtest cobra 20 bps fijos
   por lado a cualquier escala. Con el impacto de mercado, el coste crece con
   la raiz del tamano de la orden; a partir de cierto NAV se come el exceso.

**Ley de la raiz cuadrada** (Almgren et al. 2005; Toth et al. 2011; es la
regularidad empirica mas robusta de la microestructura): el impacto de una
orden de tamano Q en una accion con volumen diario V y volatilidad diaria
sigma es Y x sigma x raiz(Q / V), con Y del orden de 1. Se aplica sobre el
tamano TOTAL de la orden aunque se reparta en varios dias: el impacto no se
recupera entre sesiones de la misma orden.

Funciones puras.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _impact(order: pd.Series, adv: pd.Series, daily_vol: pd.Series, coef: float) -> pd.Series:
    ratio = (order / adv.where(adv > 0)).clip(lower=0.0)
    return coef * daily_vol * np.sqrt(ratio)


def liquidation_cost(
    weights: pd.Series,
    nav: float,
    adv: pd.Series,
    daily_vol: pd.Series,
    *,
    cost_bps: float,
    impact_coef: float,
    adv_mult: float = 1.0,
    vol_mult: float = 1.0,
    spread_mult: float = 1.0,
) -> tuple[pd.DataFrame, float]:
    """Coste de vender toda la cartera. Devuelve (detalle, coste total / NAV).

    `cost_bps` es el coste fijo por lado del backtest (comision, spread y
    slippage). En estres se multiplica entero por `spread_mult`: en una crisis
    lo que se ensancha es sobre todo el spread, y separar las piezas seria
    precision que el dato no tiene.
    """
    w = weights.astype(float)
    value = w * nav
    stressed_adv = adv.reindex(w.index) * adv_mult
    stressed_vol = daily_vol.reindex(w.index) * vol_mult
    fixed = cost_bps / 10_000.0 * spread_mult
    impact = _impact(value, stressed_adv, stressed_vol, impact_coef)
    unit_cost = fixed + impact
    detail = pd.DataFrame({
        "value": value,
        "adv": stressed_adv,
        "impact": impact,
        "unit_cost": unit_cost,
        "cost": value * unit_cost,
    }).sort_values("cost", ascending=False)
    # Sin volumen medible se cobra solo el coste fijo: el impacto no se puede
    # estimar, y el reporte ya avisa de la falta de volumen.
    total = float((value * fixed).sum() + (value * impact).fillna(0.0).sum())
    return detail, total / nav if nav else float("nan")


def capacity_curve(
    weights: pd.Series,
    adv: pd.Series,
    daily_vol: pd.Series,
    *,
    turnover: float,
    rebalances_per_year: float,
    cost_bps: float,
    impact_coef: float,
    navs,
) -> pd.DataFrame:
    """Coste anual de rebalancear frente al NAV, con impacto de mercado.

    Se supone que cada rebalanceo opera 2 x rotacion x NAV repartido como los
    pesos actuales: una aproximacion, porque en la realidad se opera mas en los
    nombres que entran y salen. La columna que importa es `impact_annual`: el
    coste que el backtest NO cobra y que aparece al crecer.
    """
    w = weights.astype(float)
    rows = []
    for nav in navs:
        order = 2.0 * turnover * w * float(nav)
        impact = _impact(order, adv.reindex(w.index), daily_vol.reindex(w.index), impact_coef).fillna(0.0)
        per_rebalance = float((order * impact).sum()) / float(nav)
        fixed = float(order.sum()) / float(nav) * cost_bps / 10_000.0
        rows.append({
            "nav": float(nav),
            "fixed_annual": fixed * rebalances_per_year,
            "impact_annual": per_rebalance * rebalances_per_year,
            "total_annual": (fixed + per_rebalance) * rebalances_per_year,
            "max_participation": float((order / adv.reindex(w.index)).max()),
        })
    return pd.DataFrame(rows).set_index("nav")


def capacity_at_budget(weights: pd.Series, adv: pd.Series, daily_vol: pd.Series, *,
                       turnover: float, rebalances_per_year: float, impact_coef: float,
                       budget: float) -> float:
    """NAV al que el impacto anual iguala `budget` (fraccion del NAV al ano).

    El impacto por dolar crece con la raiz del NAV, asi que hay solucion
    cerrada: impacto(NAV) = impacto(1) x raiz(NAV).
    """
    unit = capacity_curve(weights, adv, daily_vol, turnover=turnover,
                          rebalances_per_year=rebalances_per_year, cost_bps=0.0,
                          impact_coef=impact_coef, navs=[1.0])["impact_annual"].iloc[0]
    if not np.isfinite(unit) or unit <= 0:
        return float("nan")
    return float((budget / unit) ** 2)
