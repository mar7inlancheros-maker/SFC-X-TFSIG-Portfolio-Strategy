"""Validacion del Quant Score: predice algo, o solo ordena?

Los veredictos del plan (CONFIRM, REDUCE, NO TRADE, FLIP) descansan en el Quant
Score. Este modulo recalcula EL MISMO score en cada fin de mes pasado, con los
datos que habia ese dia, y mide si ordenaba bien los retornos del mes
siguiente: el IC de rangos (Spearman) entre score y retorno futuro, fecha a
fecha.

    IC medio > 0 y t >= 2   el score ordeno bien en este universo
    |t| < 2                 sin evidencia: el score ordena, no predice
    IC medio < 0 y t <= -2  el score ordeno al reves

**Universo**: la canasta mas los pares de referencia (`scoring.reference_peers`).
Con 5 o 10 nombres un IC mensual es casi ruido; con ~55 empieza a decir algo.
Los pares son las grandes de HOY: el universo tiene sesgo de supervivencia, y
eso lo hace optimista, no pesimista.

**Todos los componentes con historia**: precio, valor y calidad (trimestres de
Compustat por `rdq`), analistas (resumenes mensuales de IBES) e interes corto
(con 14 dias de retraso de publicacion). Sin WRDS solo se validan los de
precio: el score validado es el del YAML renormalizado sin los que faltan, y
el resultado dice cuales.

**Sin look-ahead**: las caracteristicas de la fecha `d` salen de precios hasta
`d` y de trimestres publicados (`rdq`) hasta `d`. El retorno futuro empieza en
el cierre de `d`. Hay un test que altera los precios posteriores a `d` y exige
que el score de `d` no cambie.

Las cuentas de IC y su t de Newey-West son las del modelo multifactor
(`sfc_tfsig.validation`): un IC significa lo mismo en los dos motores.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from sfc_tfsig.validation import ic_summary, information_coefficient

from . import composite


def month_end_dates(calendar: pd.DatetimeIndex, first: pd.Timestamp, last: pd.Timestamp,
                    max_dates: int) -> list[pd.Timestamp]:
    """Ultima sesion de cada mes entre `first` y `last`, las `max_dates` mas recientes.

    Fin de mes del calendario COMPLETO: un mes cortado por `last` no aporta su
    dia 17 como si fuera su cierre.
    """
    if calendar.empty:
        return []
    ends = pd.DatetimeIndex(pd.Series(calendar, index=calendar).groupby([calendar.year, calendar.month]).max().values)
    # El ultimo mes del calendario puede estar en curso: no tiene cierre todavia.
    last_day = calendar[-1]
    if last_day != last_day + pd.offsets.BMonthEnd(0):
        ends = ends[:-1]
    return list(ends[(ends >= first) & (ends <= last)])[-max_dates:]


def forward_returns(close: pd.DataFrame, dates: list[pd.Timestamp], horizon: int) -> pd.DataFrame:
    """Retorno de cada nombre desde el cierre de `d` hasta `horizon` sesiones despues.

    NaN si falta cualquiera de los dos precios: un nombre que deja de cotizar
    sale de esa fecha (sesgo de supervivencia, declarado arriba).
    """
    fwd = close.shift(-horizon) / close - 1.0
    return fwd.reindex(pd.DatetimeIndex(dates))


def validate(features_at: Callable[[pd.Timestamp], pd.DataFrame], fwd: pd.DataFrame,
             weights: dict[str, float], *, min_names: int = 20, horizon: int = 21) -> dict[str, object]:
    """IC del score y de cada componente, y spread de terciles, fecha a fecha.

    `features_at(d)` devuelve las caracteristicas (nombres x columnas) con datos
    hasta `d`. Funcion pura: los tests la llaman con datos sinteticos.
    """
    rows = []
    for d in fwd.index:
        feats = features_at(d)
        if feats is None or feats.empty:
            continue
        scored = composite.quant_score(feats, weights)
        block = scored.drop(columns=["rank"]).join(fwd.loc[d].rename("forward_return"), how="inner")
        block["date"] = d
        rows.append(block.reset_index(names="ticker"))
    if not rows:
        return {"available": False, "reason": "sin fechas con datos suficientes"}
    panel = pd.concat(rows, ignore_index=True)

    # Con horizonte de un mes y fechas mensuales no hay solape; con mas, si.
    lags = max(0, math.ceil(horizon / 21) - 1)
    ic = information_coefficient(panel, score_col="quant_score", min_names=min_names)
    if len(ic) < 6:
        return {"available": False,
                "reason": f"solo {len(ic)} meses con {min_names} nombres o mas: hacen falta 6"}
    summary = ic_summary(ic, lags=lags)

    comps = {}
    for name in composite.COMPONENTS:
        if name not in panel or panel[name].notna().sum() == 0:
            continue
        c_ic = information_coefficient(panel, score_col=name, min_names=min_names)
        if len(c_ic) >= 6:
            s = ic_summary(c_ic, lags=lags)
            comps[name] = {"ic_mean": s["mean"], "ic_t": s["t_stat"], "hit_rate": s["hit_rate"],
                           "months": int(s["n"]), "weight": float(weights.get(name, 0.0))}

    spreads = {}
    for d, g in panel.groupby("date"):
        pair = g[["quant_score", "forward_return"]].dropna()
        if len(pair) < min_names:
            continue
        cut = pair["quant_score"].quantile([1 / 3, 2 / 3])
        top = pair.loc[pair["quant_score"] >= cut.iloc[1], "forward_return"].mean()
        bottom = pair.loc[pair["quant_score"] <= cut.iloc[0], "forward_return"].mean()
        spreads[d] = top - bottom
    spread = pd.Series(spreads).sort_index()
    spread_s = ic_summary(spread, lags=lags) if len(spread) >= 6 else {}

    active = {k for k, v in weights.items() if float(v) > 0}
    validated = sorted(c for c in active if c in comps)
    return {
        "available": True,
        "months": int(summary["n"]), "first": ic.index.min(), "last": ic.index.max(),
        "horizon_d": horizon,
        "avg_names": float(panel.groupby("date")["forward_return"].count().mean()),
        "ic_mean": summary["mean"], "ic_t": summary["t_stat"], "ic_hit": summary["hit_rate"],
        "ic_ir": summary["annualized_ir"],
        "spread_mean": float(spread.mean()) if len(spread) else float("nan"),
        "spread_t": spread_s.get("t_stat", float("nan")),
        "components": pd.DataFrame(comps).T,
        "validated": validated,
        "not_validated": sorted(c for c in active if c not in comps),
        "ic": ic,
    }


def evidence(result: dict[str, object] | None, min_t: float) -> str:
    """'predictive', 'none', 'inverse' o 'unknown' (sin validacion)."""
    if not result or not result.get("available"):
        return "unknown"
    t = result.get("ic_t")
    if t is None or not np.isfinite(t):
        return "unknown"
    if t >= min_t:
        return "predictive"
    if t <= -min_t:
        return "inverse"
    return "none"
