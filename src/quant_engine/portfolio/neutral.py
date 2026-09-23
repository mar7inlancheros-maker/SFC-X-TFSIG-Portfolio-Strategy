"""Neutralidad de beta y de sector.

**Neutral en dolares no es neutral al mercado.** Una cartera 100/100 con largos
de beta 1,3 y cortos de beta 0,8 tiene beta neta +0,5: la mitad de la
exposicion de un fondo indexado, vestida de long/short.

Beta neutral por escalado de patas, manteniendo las proporciones internas de
cada una. Con beta media larga b_L y corta b_S (medias ponderadas) y bruta G:

    L b_L = S b_S   y   L + S = G     =>   L = G b_S / (b_L + b_S),  S = G - L

La neta deja de ser cero: es el precio de anular la beta. Se reportan las dos.

Neutral por sector: minimiza la suma de cuadrados de la exposicion neta por
sector, con penalizacion pequena por alejarse del equiponderado para que el
problema tenga solucion unica. Un sector presente en UNA sola pata no se puede
neutralizar sin sacar esos nombres: queda residual, y se reporta cual.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from .constraints import ConstructionParams, cap_leg, exposures
from .construction import _leg_constraints, _solve, equal_weight


def portfolio_beta(weights: pd.Series, betas: pd.Series) -> float:
    return float((weights * betas.reindex(weights.index)).sum())


def beta_neutral_legs(weights: pd.Series, betas: pd.Series, params: ConstructionParams) -> tuple[pd.Series, str]:
    w = weights.fillna(0.0)
    long_w, short_w = w[w > 0], -w[w < 0]
    b = betas.reindex(w.index)
    b_long = float((long_w * b[long_w.index]).sum() / long_w.sum())
    b_short = float((short_w * b[short_w.index]).sum() / short_w.sum())
    if b_long <= 0 or b_short <= 0:
        return w, "beta media no positiva en una pata: no se puede neutralizar por escalado"

    long_total = params.gross * b_short / (b_long + b_short)
    short_total = params.gross - long_total
    note = ""
    try:
        new_long = cap_leg(long_total * long_w / long_w.sum(), long_total, params.max_position)
        new_short = cap_leg(short_total * short_w / short_w.sum(), short_total, params.max_position)
    except ValueError as exc:
        return w, f"beta neutral imposible con el tope por nombre: {exc}"
    out = pd.Series(0.0, index=w.index)
    out[new_long.index] = new_long
    out[new_short.index] = -new_short
    if abs(portfolio_beta(out, betas)) > 1e-6:
        note = "el tope por nombre deja beta residual"
    return out, note


def sector_exposure(weights: pd.Series, sectors: dict[str, str]) -> pd.Series:
    s = pd.Series({t: sectors.get(t, "Unknown") for t in weights.index})
    return weights.groupby(s).sum().sort_values()


def sector_neutral(signs: pd.Series, sectors: dict[str, str], params: ConstructionParams,
                   penalty: float = 0.05) -> tuple[pd.Series | None, str, list[str]]:
    names = list(signs.index)
    labels = pd.Series({t: sectors.get(t, "Unknown") for t in names})
    base = equal_weight(signs.astype(float), params).to_numpy(float)
    w = cp.Variable(len(names))
    net_by_sector = []
    for sector in labels.unique():
        idx = np.where(labels.to_numpy() == sector)[0]
        net_by_sector.append(cp.sum(w[idx]))
    objective = cp.sum_squares(cp.hstack(net_by_sector)) + penalty * cp.sum_squares(w - base)
    prob = cp.Problem(cp.Minimize(objective), _leg_constraints(w, signs.astype(float), params))
    status = _solve(prob)
    if w.value is None:
        return None, f"sector neutral sin solucion ({status})", []

    one_sided = []
    for sector, group in signs.groupby(labels):
        if (group > 0).all() or (group < 0).all():
            one_sided.append(sector)
    note = ""
    if one_sided:
        note = f"sectores con nombres en una sola pata, no neutralizables: {', '.join(sorted(one_sided))}"
    return pd.Series(w.value, index=names).where(lambda s: s.abs() > 1e-10, 0.0), note, one_sided


def describe(weights: pd.Series, betas: pd.Series) -> dict[str, float]:
    e = exposures(weights)
    e["beta"] = portfolio_beta(weights, betas)
    return e
