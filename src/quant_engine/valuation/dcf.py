"""DCF del flujo libre de la empresa (FCFF). Funciones puras.

Misma estructura que los modelos en Excel del club (PLTR, CNL): ventas, EBIT,
impuestos, NOPAT, D&A, CapEx, variacion del capital circulante, FCFF,
descuento a mitad de ano, valor terminal de Gordon, puente a valor por accion
y sensibilidades. Lo que cambia es lo que un comite de inversion pediria:

**1. Valor terminal coherente con el crecimiento.** Crecer al g para siempre
exige reinvertir. El FCFF del ano N+1 es

    FCFF(N+1) = NOPAT(N) x (1 + g) x (1 - g / RONIC)

con RONIC (rentabilidad del capital NUEVO) = WACC por defecto: en equilibrio
competitivo el crecimiento del terminal no crea valor. El Excel de PLTR
crecia al 3% perpetuo con un CapEx del 0,6% de las ventas: eso es un ROIC
infinito y era el 71% del valor.

**2. Diez anos con convergencia, no cinco con un escalon.** El crecimiento
baja en linea recta del ano 1 al terminal en el ano N; el margen converge a su
objetivo en `margin_years`; el impuesto efectivo sube al marginal (las bases
imponibles negativas se agotan). Pasar de crecer 18% en el ano 5 a 3% en el 6
no lo hace ninguna empresa.

**3. Crecer cuesta capital tambien en los anos explicitos.** La reinversion
neta (CapEx - D&A + circulante) nunca baja de lo nuevo vendido entre
`sales_to_capital`: cada 5 dolares de ventas nuevas exigen al menos 1 de
capital. Con los ratios historicos de PLTR (D&A por encima del CapEx y
circulante negativo), crecer un 40% al ano salia gratis.

**4. Mitad de ano tambien en el terminal.** Con flujos a mitad de ano, el
valor terminal se descuenta N - 0,5 anos: los flujos del ano N+1 en adelante
tambien llegan a mitad de cada ano. Es la convencion del Excel del club, y es
la consistente.

**5. Todo vectorizado.** La misma funcion valora un escenario, una rejilla de
sensibilidad, 5.000 sorteos de Monte Carlo o la busqueda del DCF inverso. No
hay dos formulas que puedan discrepar.

Unidades: millones de la moneda en que reporta la empresa; tasas en
fraccion (0,08 = 8%).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

# Escala de rating sintetico por cobertura de intereses (EBIT / gasto
# financiero), estructura de Damodaran. Diferenciales aproximados de enero de
# 2025 sobre el bono del Tesoro: se revisan una vez al ano en
# `valuation.rating_spreads` del YAML si hace falta.
# (cobertura minima empresa grande, cobertura minima empresa pequena, rating, diferencial)
RATING_TABLE: tuple[tuple[float, float, str, float], ...] = (
    (8.50, 12.50, "AAA", 0.0059),
    (6.50, 9.50, "AA", 0.0070),
    (5.50, 7.50, "A+", 0.0092),
    (4.25, 6.00, "A", 0.0107),
    (3.00, 4.50, "A-", 0.0121),
    (2.50, 4.00, "BBB", 0.0147),
    (2.25, 3.50, "BB+", 0.0186),
    (2.00, 3.00, "BB", 0.0238),
    (1.75, 2.50, "B+", 0.0286),
    (1.50, 2.00, "B", 0.0336),
    (1.25, 1.50, "B-", 0.0400),
    (0.80, 1.25, "CCC", 0.0575),
    (0.65, 0.80, "CC", 0.0800),
    (0.20, 0.50, "C", 0.1000),
    (-np.inf, -np.inf, "D", 0.1200),
)
LARGE_FIRM_USD_MM = 5_000.0


@dataclass(frozen=True)
class Capital:
    """Precio y estructura de capital, en millones de la moneda de reporte."""

    price: float        # por accion, en la moneda de reporte
    shares: float       # millones, diluidas
    cash: float         # caja e inversiones a corto
    debt: float         # deuda financiera y arrendamientos financieros
    minority: float = 0.0
    preferred: float = 0.0

    @property
    def market_cap(self) -> float:
        return self.price * self.shares

    @property
    def net_debt(self) -> float:
        return self.debt - self.cash


@dataclass(frozen=True)
class Assumptions:
    """Un escenario completo. Vectores de longitud `years` (ano 1 = el siguiente)."""

    growth: np.ndarray
    margin: np.ndarray
    tax: np.ndarray
    da_pct: float
    capex_pct: float
    nwc_pct: float          # variacion del circulante por cada unidad de variacion de ventas
    terminal_growth: float
    wacc: float
    ronic: float
    sales_to_capital: float = 5.0   # tope de ventas nuevas por cada unidad de capital nuevo
    label: str = "base"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def years(self) -> int:
        return len(self.growth)


def blume(beta: float) -> float:
    """Beta ajustada de Blume (1975): las betas revierten hacia 1."""
    return 0.67 * beta + 0.33


def synthetic_rating(coverage: float, market_cap_usd: float,
                     table: tuple = RATING_TABLE) -> tuple[str, float]:
    """Rating y diferencial de credito por cobertura de intereses.

    Sin gasto financiero (cobertura infinita) es AAA. Las empresas pequenas
    necesitan mas cobertura para el mismo rating.
    """
    if coverage is None or np.isnan(coverage):
        coverage = np.inf
    small = market_cap_usd < LARGE_FIRM_USD_MM
    for large_min, small_min, rating, spread in table:
        if coverage >= (small_min if small else large_min):
            return rating, spread
    return table[-1][2], table[-1][3]


def cost_of_capital(rf: float, beta_raw: float, erp: float, marginal_tax: float, ebit: float,
                    interest: float, cap: Capital, market_cap_usd: float,
                    table: tuple = RATING_TABLE) -> dict[str, float | str]:
    """WACC: CAPM con beta de Blume y deuda con rating sintetico, a pesos de mercado."""
    beta = blume(beta_raw)
    ke = rf + beta * erp
    coverage = ebit / interest if interest and interest > 0 else np.inf
    rating, spread = synthetic_rating(coverage, market_cap_usd, table)
    kd = rf + spread
    e, d = cap.market_cap, max(cap.debt, 0.0)
    we = e / (e + d) if e + d > 0 else 1.0
    wacc = we * ke + (1 - we) * kd * (1 - marginal_tax)
    return {"rf": rf, "beta_raw": beta_raw, "beta": beta, "erp": erp, "cost_of_equity": ke,
            "coverage": coverage, "rating": rating, "spread": spread, "cost_of_debt": kd,
            "after_tax_cost_of_debt": kd * (1 - marginal_tax), "marginal_tax": marginal_tax,
            "weight_equity": we, "weight_debt": 1 - we, "wacc": wacc}


def fade(start: float, end: float, years: int, hold: int = 1) -> np.ndarray:
    """`hold` anos en `start` (al menos el primero) y en linea recta hasta `end` en el ultimo.

    Con 10 anos y hold 1: el ano 1 crece `start`, el 10 crece `end`, y cada ano
    intermedio baja lo mismo.
    """
    if years <= 1:
        return np.array([float(end)])
    hold = min(max(hold, 1), years - 1)
    t = np.arange(1, years + 1)
    w = np.clip((t - hold) / (years - hold), 0.0, 1.0)
    return start + (end - start) * w


def converge(start: float, target: float, years: int, by_year: int) -> np.ndarray:
    """De `start` (ano 0) a `target` en linea recta hasta `by_year`, luego plano."""
    t = np.arange(1, years + 1)
    w = np.clip(t / max(by_year, 1), 0, 1)
    return start + (target - start) * w


def _paths(base_revenue: float, growth: np.ndarray, margin: np.ndarray, tax: np.ndarray,
           da_pct: np.ndarray, capex_pct: np.ndarray, nwc_pct: np.ndarray,
           sales_to_capital: np.ndarray | float = np.inf) -> dict[str, np.ndarray]:
    """Proyeccion vectorizada: cada fila un escenario o un sorteo, cada columna un ano."""
    growth = np.atleast_2d(growth)
    revenue = base_revenue * np.cumprod(1 + growth, axis=1)
    previous = np.concatenate([np.full((revenue.shape[0], 1), base_revenue), revenue[:, :-1]], axis=1)
    ebit = revenue * np.atleast_2d(margin)
    taxes = np.maximum(ebit, 0) * np.atleast_2d(tax)
    nopat = ebit - taxes
    da = revenue * np.reshape(da_pct, (-1, 1))
    capex = revenue * np.reshape(capex_pct, (-1, 1))
    dnwc = (revenue - previous) * np.reshape(nwc_pct, (-1, 1))
    floor = np.maximum(revenue - previous, 0.0) / np.reshape(sales_to_capital, (-1, 1))
    extra = np.maximum(floor - (capex - da + dnwc), 0.0)
    fcff = nopat + da - capex - dnwc - extra
    return {"revenue": revenue, "ebit": ebit, "taxes": taxes, "nopat": nopat, "da": da,
            "capex": capex, "dnwc": dnwc, "growth_capital": extra, "fcff": fcff}


def value_vector(base_revenue: float, growth, margin, tax, da_pct, capex_pct, nwc_pct,
                 terminal_growth, wacc, ronic, cap: Capital, sales_to_capital=np.inf) -> dict[str, np.ndarray]:
    """Valor por accion de N escenarios a la vez. Escalares o vectores de longitud N."""
    growth = np.atleast_2d(np.asarray(growth, dtype=float))
    n, years = growth.shape
    vec = lambda v: np.broadcast_to(np.asarray(v, dtype=float), (n,)).astype(float)  # noqa: E731
    wacc, g, ronic = vec(wacc), vec(terminal_growth), vec(ronic)
    margin = np.broadcast_to(np.atleast_2d(np.asarray(margin, dtype=float)), (n, years))
    tax = np.broadcast_to(np.atleast_2d(np.asarray(tax, dtype=float)), (n, years))
    p = _paths(base_revenue, growth, margin, tax, vec(da_pct), vec(capex_pct), vec(nwc_pct), vec(sales_to_capital))
    periods = np.arange(1, years + 1) - 0.5
    factor = (1 + wacc[:, None]) ** -periods[None, :]
    pv_fcff = (p["fcff"] * factor).sum(axis=1)
    reinvest = np.where(ronic > 0, g / ronic, 0.0)
    fcff_next = p["nopat"][:, -1] * (1 + g) * (1 - reinvest)
    spread = wacc - g
    tv = np.where(spread > 0, fcff_next / np.where(spread > 0, spread, np.nan), np.nan)
    pv_tv = tv * (1 + wacc) ** -(years - 0.5)
    ev = pv_fcff + pv_tv
    equity = ev + cap.cash - cap.debt - cap.minority - cap.preferred
    # Responsabilidad limitada: una accion no vale menos que cero aunque la deuda
    # supere al negocio. El patrimonio negativo se conserva para el aviso.
    per_share = np.maximum(equity, 0.0) / cap.shares if cap.shares > 0 else np.full(n, np.nan)
    return {"pv_fcff": pv_fcff, "tv": tv, "pv_tv": pv_tv, "ev": ev, "equity": equity,
            "per_share": per_share, "paths": p, "factor": factor}


def value(base_revenue: float, a: Assumptions, cap: Capital, invested_capital: float | None = None,
          fiscal_end: pd.Timestamp | None = None) -> dict[str, object]:
    """Un escenario: valor, tabla anual y comprobaciones."""
    out = value_vector(base_revenue, a.growth, a.margin, a.tax, a.da_pct, a.capex_pct, a.nwc_pct,
                       a.terminal_growth, a.wacc, a.ronic, cap, a.sales_to_capital)
    p = {k: v[0] for k, v in out["paths"].items()}
    years = a.years
    labels = ([f"FY{(fiscal_end.year + t)}E" for t in range(1, years + 1)] if fiscal_end is not None
              else [f"Y{t}" for t in range(1, years + 1)])
    table = pd.DataFrame({
        "revenue": p["revenue"], "growth": a.growth, "ebit": p["ebit"], "margin": a.margin,
        "tax_rate": a.tax, "nopat": p["nopat"], "da": p["da"], "capex": p["capex"], "dnwc": p["dnwc"],
        "growth_capital": p["growth_capital"], "fcff": p["fcff"],
        "period": np.arange(1, years + 1) - 0.5, "discount": out["factor"][0],
        "pv_fcff": p["fcff"] * out["factor"][0],
    }, index=labels)
    if invested_capital is not None and np.isfinite(invested_capital):
        reinvestment = p["capex"] - p["da"] + p["dnwc"] + p["growth_capital"]
        capital = invested_capital + np.cumsum(reinvestment)
        opening = np.concatenate([[invested_capital], capital[:-1]])
        table["invested_capital"] = capital
        table["roic"] = np.where(opening > 0, p["nopat"] / np.where(opening > 0, opening, np.nan), np.nan)
    ev, pv_tv = float(out["ev"][0]), float(out["pv_tv"][0])
    per_share = float(out["per_share"][0])
    ebitda_n = p["ebit"][-1] + p["da"][-1]
    result = {
        "label": a.label, "table": table, "pv_fcff": float(out["pv_fcff"][0]), "tv": float(out["tv"][0]),
        "pv_tv": pv_tv, "ev": ev, "equity": float(out["equity"][0]), "per_share": per_share,
        "upside": per_share / cap.price - 1 if cap.price > 0 else np.nan,
        "tv_share": pv_tv / ev if ev > 0 else np.nan,
        "terminal_ev_ebitda": float(out["tv"][0]) / ebitda_n if ebitda_n > 0 else np.nan,
        "wacc": a.wacc, "terminal_growth": a.terminal_growth, "ronic": a.ronic,
        # Con caja neta y poco activo (software), el capital invertido es casi
        # cero y el ROIC sale en miles por ciento sin que nada este mal.
        "capital_is_material": bool(invested_capital is not None and np.isfinite(invested_capital)
                                    and invested_capital > 0.05 * base_revenue),
    }
    result["flags"] = checks(result, a, cap)
    return result


def checks(result: dict, a: Assumptions, cap: Capital) -> list[str]:
    """Lo que un comite preguntaria antes de creerse el numero."""
    flags = []
    if a.wacc - a.terminal_growth < 0.02:
        flags.append(f"WACC {a.wacc:.1%} is within 2 points of terminal growth {a.terminal_growth:.1%}: "
                     "the terminal value explodes; do not use")
    if a.margin[-1] <= 0:
        flags.append("negative operating margin in the terminal year: a DCF of a business that never "
                     "earns money is not meaningful; set a target margin or use another method")
    tv_share = result.get("tv_share")
    if tv_share is not None and np.isfinite(tv_share) and tv_share > 0.8:
        flags.append(f"terminal value is {tv_share:.0%} of enterprise value: the price rests on years 11+")
    if result.get("equity", 0) <= 0:
        flags.append("equity value is zero or negative: debt exceeds the value of the operations, so the shares "
                     "are worth only their option value (shown as 0)")
    roic = result["table"].get("roic") if result.get("capital_is_material") else None
    # 150% y no 100%: Apple ronda el 90% de verdad. Lo que se busca es el ROIC
    # absurdo de un CapEx que no acompana al crecimiento.
    if roic is not None and np.nanmax(np.abs(roic.to_numpy())) > 1.5:
        flags.append(f"ROIC reaches {np.nanmax(roic.to_numpy()):.0%} in the forecast: either the capital base "
                     "is tiny or CapEx and working capital do not keep up with the growth; check both")
    mult = result.get("terminal_ev_ebitda")
    if mult is not None and np.isfinite(mult) and not 3 <= mult <= 30:
        flags.append(f"implied terminal EV/EBITDA {mult:.1f}x is outside 3-30x")
    return flags


def with_shift(a: Assumptions, growth_shift: float = 0.0, margin_shift: float = 0.0, wacc_shift: float = 0.0,
               g_shift: float = 0.0, label: str | None = None) -> Assumptions:
    """El mismo escenario desplazado. El crecimiento se desplaza solo en el exceso
    sobre el terminal: el ultimo ano sigue llegando al g del terminal."""
    years = a.years
    weight = np.linspace(1.0, 0.0, years) if years > 1 else np.zeros(1)
    return replace(a, growth=a.growth + growth_shift * weight, margin=a.margin + margin_shift,
                   wacc=a.wacc + wacc_shift, terminal_growth=a.terminal_growth + g_shift,
                   ronic=max(a.ronic + wacc_shift, 1e-6) if a.ronic == a.wacc else a.ronic,
                   label=label or a.label)


def sensitivity(base_revenue: float, a: Assumptions, cap: Capital, rows: dict[str, list[float]],
                cols: dict[str, list[float]]) -> pd.DataFrame:
    """Rejilla de valor por accion. `rows`/`cols`: {parametro: desplazamientos}."""
    (rname, rvals), (cname, cvals) = next(iter(rows.items())), next(iter(cols.items()))
    grid = pd.DataFrame(index=pd.Index(rvals, name=rname), columns=pd.Index(cvals, name=cname), dtype=float)
    for r in rvals:
        for c in cvals:
            s = with_shift(a, **{f"{rname}_shift": r, f"{cname}_shift": c})
            v = value_vector(base_revenue, s.growth, s.margin, s.tax, s.da_pct, s.capex_pct, s.nwc_pct,
                             s.terminal_growth, s.wacc, s.ronic, cap, s.sales_to_capital)["per_share"][0]
            grid.loc[r, c] = v if s.wacc - s.terminal_growth >= 0.01 else np.nan
    return grid


def monte_carlo(base_revenue: float, a: Assumptions, cap: Capital, *, growth_sd: float, margin_sd: float,
                wacc_sd: float = 0.01, g_halfwidth: float = 0.005, n: int = 5000, seed: int = 0,
                min_spread: float = 0.02) -> dict[str, float | np.ndarray]:
    """Distribucion del valor por accion si los supuestos clave son inciertos.

    Crecimiento (en el exceso sobre el terminal) y margen objetivo se desplazan
    con la dispersion historica de la propia empresa; WACC normal con 1 punto
    de desviacion; g uniforme +-0,5 puntos. Los sorteos con WACC - g menor que
    `min_spread` se recortan a ese minimo: el terminal no puede explotar.
    """
    rng = np.random.default_rng(seed)
    years = a.years
    weight = np.linspace(1.0, 0.0, years) if years > 1 else np.zeros(1)
    gs = rng.normal(0.0, growth_sd, n)
    ms = rng.normal(0.0, margin_sd, n)
    w = np.maximum(a.wacc + rng.normal(0.0, wacc_sd, n), 0.03)
    g = a.terminal_growth + rng.uniform(-g_halfwidth, g_halfwidth, n)
    g = np.minimum(g, w - min_spread)
    growth = a.growth[None, :] + gs[:, None] * weight[None, :]
    margin = a.margin[None, :] + ms[:, None]
    ronic = w if a.ronic == a.wacc else np.full(n, a.ronic)
    v = value_vector(base_revenue, growth, margin, a.tax, a.da_pct, a.capex_pct, a.nwc_pct, g, w, ronic,
                     cap, a.sales_to_capital)["per_share"]
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    return {"n": len(v), "mean": float(v.mean()), "p05": float(np.quantile(v, 0.05)),
            "p25": float(np.quantile(v, 0.25)), "p50": float(np.median(v)),
            "p75": float(np.quantile(v, 0.75)), "p95": float(np.quantile(v, 0.95)),
            "prob_above_price": float((v > cap.price).mean()), "values": v}


def reverse_dcf(base_revenue: float, a: Assumptions, cap: Capital, lo: float = -0.6,
                hi: float = 2.0, tol: float = 1e-6) -> dict[str, float | str]:
    """Que crecimiento descuenta el precio de hoy, con el resto de supuestos fijos.

    Se busca el desplazamiento del exceso de crecimiento (el mismo perfil de
    convergencia) que iguala el valor al precio. El valor crece con el
    crecimiento mientras el margen supere el coste de la reinversion; si no, el
    precio no se alcanza y se dice.
    """
    def gap(shift: float) -> float:
        s = with_shift(a, growth_shift=shift)
        return float(value_vector(base_revenue, s.growth, s.margin, s.tax, s.da_pct, s.capex_pct, s.nwc_pct,
                                  s.terminal_growth, s.wacc, s.ronic, cap, s.sales_to_capital)["per_share"][0]
                     - cap.price)

    f_lo, f_hi = gap(lo), gap(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        side = "above" if f_hi < 0 and f_lo < 0 else "below"
        return {"solved": False, "note": f"the price is {side} anything this model gives for growth paths "
                                         f"from {lo:+.0%} to {hi:+.0%} over the base case"}
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = gap(mid)
        if abs(f_mid) < tol * max(cap.price, 1) or hi - lo < tol:
            break
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    s = with_shift(a, growth_shift=mid)
    cagr = float(np.prod(1 + s.growth) ** (1 / a.years) - 1)
    return {"solved": True, "growth_shift": mid, "implied_year1_growth": float(s.growth[0]),
            "implied_cagr": cagr, "base_cagr": float(np.prod(1 + a.growth) ** (1 / a.years) - 1),
            "years": a.years}


def scenarios(base_revenue: float, cases: dict[str, Assumptions], weights: dict[str, float], cap: Capital,
              invested_capital: float | None = None, fiscal_end: pd.Timestamp | None = None) -> dict:
    """Valor de cada escenario y el valor ponderado por probabilidad."""
    results = {k: value(base_revenue, a, cap, invested_capital, fiscal_end) for k, a in cases.items()}
    w = {k: float(weights.get(k, 0.0)) for k in results}
    total = sum(w.values()) or 1.0
    expected = sum(results[k]["per_share"] * w[k] for k in results) / total
    return {"cases": results, "weights": {k: v / total for k, v in w.items()}, "expected": expected,
            "expected_upside": expected / cap.price - 1 if cap.price > 0 else np.nan}
