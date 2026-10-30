"""Supuestos del DCF a partir de la historia de la empresa, el consenso y el YAML.

Cada supuesto sale de una regla escrita aqui, y la regla queda anotada en el
resultado (`notes`): un DCF automatico que no dice de donde sale cada numero
no se puede discutir en un comite.

| Supuesto | Regla por defecto |
|---|---|
| Crecimiento ano 1 (y 2) | Consenso de ventas de IBES si existe; si no, media del ultimo ano y la CAGR de 3 anos, acotada a [-20%, +40%] |
| Crecimiento anos siguientes | Linea recta hasta el crecimiento terminal en el ano N |
| Margen EBIT | Se mantiene el actual (persistencia); escenarios +- la dispersion historica. Con margen negativo, el mejor margen positivo propio o se avisa |
| Impuesto | Efectivo de 3 anos subiendo al marginal del pais en el ano N |
| D&A, CapEx | Mediana de 3 anos sobre ventas |
| Circulante | Mediana de 3 anos de variacion de circulante / variacion de ventas, acotada |
| Terminal g | `terminal_growth` del YAML, nunca por encima de la tasa libre de riesgo |
| RONIC | WACC + `ronic_excess` (0: el crecimiento del terminal no crea valor) |

Por que el margen se mantiene y no "revierte a la media": revertir a la
historia propia hundia a las empresas que mejoraron de verdad (PLTR paso del
-40% al 32%), y revertir a la del sector exige datos de sector que no siempre
hay. La persistencia es el supuesto neutral; el escenario bajista lleva la
reversion. Un analista que sepa mas lo pone en `valuation.overrides`.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .dcf import Assumptions, converge, fade, with_shift

DEFAULTS: dict[str, object] = {
    "years": 10,
    "margin_years": 5,
    "terminal_growth": 0.025,
    "erp": 0.045,
    "ronic_excess": 0.0,
    "sales_to_capital": 5.0,
    "start_growth_bounds": [-0.20, 0.40],
    "marginal_tax": {"USD": 0.25, "CAD": 0.265, "EUR": 0.25, "GBP": 0.25, "default": 0.25},
    "scenario_weights": {"bear": 0.25, "base": 0.50, "bull": 0.25},
    "scenario_wacc_shift": 0.01,
    "scenario_g_shift": 0.005,
}


def setting(cfg: dict, key: str):
    return cfg.get(key, DEFAULTS[key]) if cfg else DEFAULTS[key]


def _recent(series: pd.Series, n: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna().iloc[-n:]


def derive(hist: pd.DataFrame) -> dict[str, float]:
    """Ratios historicos que alimentan los supuestos. `hist`: anual, del mas antiguo al mas reciente."""
    rev = pd.to_numeric(hist["revenue"], errors="coerce")
    ebit = pd.to_numeric(hist["ebit"], errors="coerce")
    growth = rev.pct_change(fill_method=None)
    margin = ebit / rev.where(rev > 0)
    out: dict[str, float] = {
        "revenue": float(rev.iloc[-1]),
        "ebit": float(ebit.iloc[-1]),
        "margin": float(margin.iloc[-1]),
        "last_growth": float(growth.iloc[-1]) if len(growth.dropna()) else np.nan,
        "years_of_history": int(rev.notna().sum()),
    }
    valid = rev.dropna()
    if len(valid) >= 4 and valid.iloc[-4] > 0 and valid.iloc[-1] > 0:
        out["cagr3"] = float((valid.iloc[-1] / valid.iloc[-4]) ** (1 / 3) - 1)
    else:
        out["cagr3"] = np.nan
    positive = margin[margin > 0]
    out["best_positive_margin"] = float(positive.max()) if len(positive) else np.nan
    out["margin_median5"] = float(_recent(margin, 5).median()) if len(_recent(margin, 5)) else np.nan
    out["growth_sd"] = float(np.clip(_recent(growth, 5).std(), 0.03, 0.25)) if len(_recent(growth, 5)) > 2 else 0.08
    out["margin_sd"] = float(np.clip(_recent(margin, 5).std(), 0.01, 0.10)) if len(_recent(margin, 5)) > 2 else 0.03

    def ratio(num: str, den: pd.Series, lo: float, hi: float, default: float) -> float:
        if num not in hist:
            return default
        r = _recent(pd.to_numeric(hist[num], errors="coerce") / den.where(den > 0), 3)
        return float(np.clip(r.median(), lo, hi)) if len(r) else default

    out["da_pct"] = ratio("da", rev, 0.0, 0.30, 0.03)
    out["capex_pct"] = ratio("capex", rev, 0.0, 0.40, out["da_pct"])
    out["sbc_pct"] = ratio("sbc", rev, 0.0, 0.60, np.nan)
    pretax = pd.to_numeric(hist.get("pretax_income"), errors="coerce") if "pretax_income" in hist else None
    if pretax is not None and "tax_expense" in hist:
        eff = _recent(pd.to_numeric(hist["tax_expense"], errors="coerce") / pretax.where(pretax > 0), 3)
        out["effective_tax"] = float(np.clip(eff.median(), 0.0, 0.40)) if len(eff) else np.nan
    else:
        out["effective_tax"] = np.nan
    if {"current_assets", "current_liabilities", "cash"} <= set(hist.columns):
        debt_cur = (pd.to_numeric(hist["debt_current"], errors="coerce").fillna(0.0)
                    if "debt_current" in hist else 0.0)
        nwc = (hist["current_assets"] - hist["cash"]) - (hist["current_liabilities"] - debt_cur)
        drev = rev.diff()
        r = _recent(nwc.diff() / drev.where(drev.abs() > 0.01 * rev.abs()), 3)
        out["nwc_pct"] = float(np.clip(r.median(), -0.30, 0.50)) if len(r) else 0.10
        out["nwc"] = float(nwc.iloc[-1]) if pd.notna(nwc.iloc[-1]) else np.nan
    else:
        out["nwc_pct"], out["nwc"] = 0.10, np.nan
    last = hist.iloc[-1]
    val = lambda k: float(last[k]) if k in last and pd.notna(last[k]) else 0.0  # noqa: E731
    ic = val("equity") + val("debt") + val("minority") - val("cash")
    out["invested_capital"] = ic if np.isfinite(ic) else np.nan
    # Ventas por unidad de capital invertido, mediana de 3 anos: cuanto capital
    # pide de verdad este negocio para vender un dolar mas.
    caps = []
    for _, row in hist.iloc[-3:].iterrows():
        v = lambda k, r=row: float(r[k]) if k in r and pd.notna(r[k]) else 0.0  # noqa: E731
        capital = v("equity") + v("debt") + v("minority") - v("cash")
        if capital > 0 and v("revenue") > 0:
            caps.append(v("revenue") / capital)
    out["sales_to_capital"] = float(np.median(caps)) if caps else np.nan
    out["roic"] = (out["ebit"] * (1 - (out["effective_tax"] if np.isfinite(out["effective_tax"]) else 0.25))
                   / ic) if ic and ic > 0 else np.nan
    return out


def start_growth(d: dict, consensus: list[float] | None, bounds: tuple[float, float]) -> tuple[list[float], str]:
    """Crecimiento de los primeros anos y de donde sale."""
    if consensus:
        return [float(g) for g in consensus[:2]], "IBES consensus revenue estimates"
    parts = [g for g in (d.get("last_growth"), d.get("cagr3")) if g is not None and np.isfinite(g)]
    if not parts:
        return [0.0], "no revenue history: zero growth assumed"
    g = float(np.clip(np.mean(parts), *bounds))
    src = ("average of last-year growth and 3-year CAGR" if len(parts) == 2 else "last-year growth")
    return [g], f"{src}, bounded to [{bounds[0]:+.0%}, {bounds[1]:+.0%}]"


def build(hist: pd.DataFrame, coc: dict[str, float], currency: str, cfg: dict | None = None,
          consensus: list[float] | None = None, overrides: dict | None = None) -> dict[str, object]:
    """Escenarios base, bajista y alcista, con las notas de cada supuesto."""
    cfg = cfg or {}
    o = overrides or {}
    years = int(o.get("years", setting(cfg, "years")))
    d = derive(hist)
    notes: list[str] = []

    rf = coc["rf"]
    g_term = float(o.get("terminal_growth", min(float(setting(cfg, "terminal_growth")), rf)))
    if g_term > rf + 1e-12:
        notes.append(f"terminal growth {g_term:.1%} is above the risk-free rate {rf:.1%}: the company "
                     "would outgrow the economy forever")
    taxes = setting(cfg, "marginal_tax")
    marginal = float(taxes.get(currency, taxes.get("default", 0.25)) if isinstance(taxes, dict) else taxes)

    bounds = tuple(setting(cfg, "start_growth_bounds"))
    if "growth" in o:
        g0 = o["growth"] if isinstance(o["growth"], list) else [o["growth"]]
        start, src = [float(x) for x in g0][:2], "analyst override"
    else:
        start, src = start_growth(d, consensus, bounds)
    if len(start) >= 2 and years > 2:
        growth = np.concatenate([[start[0]], fade(start[1], g_term, years - 1)])
    else:
        growth = fade(start[0], g_term, years)
    notes.append(f"growth: year 1 {growth[0]:+.1%} ({src}), straight line to {g_term:.1%} in year {years}")

    current = d["margin"]
    if "target_margin" in o:
        target, msrc = float(o["target_margin"]), "analyst override"
    elif np.isfinite(current) and current > 0:
        target, msrc = current, "current margin held (persistence)"
    elif np.isfinite(d["best_positive_margin"]):
        target, msrc = d["best_positive_margin"], "negative margin today: back to its best positive margin"
    else:
        target, msrc = current, "never profitable: no margin to converge to"
    target = float(min(target, 0.60))
    margin_years = int(o.get("margin_years", setting(cfg, "margin_years")))
    margin = converge(current if np.isfinite(current) else 0.0, target, years, margin_years)
    notes.append(f"EBIT margin: {current:.1%} now, {target:.1%} by year {margin_years} ({msrc})")

    eff = d["effective_tax"] if np.isfinite(d["effective_tax"]) else marginal
    eff = min(eff, marginal)
    tax = converge(eff, marginal, years, years)
    notes.append(f"tax: effective {eff:.1%} rising to the marginal {marginal:.1%} by year {years}")

    da_pct = float(o.get("da_pct", d["da_pct"]))
    capex_pct = float(o.get("capex_pct", d["capex_pct"]))
    nwc_pct = float(o.get("nwc_pct", d["nwc_pct"]))
    notes.append(f"reinvestment: D&A {da_pct:.1%} and CapEx {capex_pct:.1%} of revenue, working capital "
                 f"{nwc_pct:.0%} of each new dollar of revenue (3-year medians)")

    wacc = float(o.get("wacc", coc["wacc"]))
    ronic = wacc + float(o.get("ronic_excess", setting(cfg, "ronic_excess")))
    notes.append(f"terminal: g {g_term:.1%}, RONIC {ronic:.1%} (reinvests {g_term / ronic:.0%} of NOPAT)"
                 if ronic > 0 else f"terminal: g {g_term:.1%}")
    if np.isfinite(d.get("sbc_pct", np.nan)) and d["sbc_pct"] > 0.10:
        notes.append(f"stock-based pay is {d['sbc_pct']:.0%} of revenue: GAAP EBIT already expenses it, so it "
                     "is not added back")

    floor_s2c = float(setting(cfg, "sales_to_capital"))
    own = d.get("sales_to_capital", np.nan)
    if "sales_to_capital" in o:
        s2c, s2c_src = float(o["sales_to_capital"]), "analyst override"
    elif np.isfinite(own) and own > floor_s2c:
        s2c, s2c_src = float(min(own, 25.0)), f"its own 3-year sales-to-capital ({own:.1f}x, capped at 25x)"
    else:
        s2c, s2c_src = floor_s2c, "the default floor"
    notes.append(f"growth needs capital: at least 1 of net reinvestment for every {s2c:.1f} of new revenue "
                 f"({s2c_src})")
    base = Assumptions(growth=growth, margin=margin, tax=tax, da_pct=da_pct, capex_pct=capex_pct,
                       nwc_pct=nwc_pct, terminal_growth=g_term, wacc=wacc, ronic=ronic, sales_to_capital=s2c,
                       label="base", notes=tuple(notes))
    excess = abs(growth[0] - g_term)
    gshift = max(0.5 * excess, 0.02)
    mshift = max(d["margin_sd"], 0.02)
    ws, gs = float(setting(cfg, "scenario_wacc_shift")), float(setting(cfg, "scenario_g_shift"))
    bear = with_shift(base, growth_shift=-gshift, margin_shift=-mshift, wacc_shift=ws, g_shift=-gs, label="bear")
    bull = with_shift(base, growth_shift=gshift, margin_shift=mshift, wacc_shift=-ws,
                      g_shift=min(gs, max(rf - g_term, 0.0)), label="bull")
    bull = replace(bull, margin=np.minimum(bull.margin, 0.60))
    return {"cases": {"bear": bear, "base": base, "bull": bull}, "derived": d,
            "weights": dict(setting(cfg, "scenario_weights")), "marginal_tax": marginal,
            "scenario_shifts": {"growth": gshift, "margin": mshift, "wacc": ws, "g": gs}}
