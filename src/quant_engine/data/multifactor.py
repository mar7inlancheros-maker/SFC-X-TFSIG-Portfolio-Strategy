"""El score del modelo multifactor (sfc_tfsig) como evidencia para el plan.

El Quant Score del motor long/short no ha demostrado que prediga (IC ~0 en su
validacion). El compuesto del modelo multifactor si: en su backtest de 2012 a
2026, sobre ~1.200 empresas al mes, IC de rangos 0,024 con t de Newey-West 3,0
y quintiles casi perfectamente ordenados. Este modulo lee su ultimo panel
puntuado (de disco, sin red) y dice en que percentil esta cada nombre.

La evidencia NO se escribe a mano: se lee del ultimo reporte de backtest
(`output/reports/backtest_*.md`). Sin reporte, o con un IC sin significancia,
el percentil se muestra pero no mueve la conviccion ni los FLIP.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

FACTOR_COLUMNS = ("score_value", "score_quality", "score_momentum", "score_lowvol")
STALE_DAYS = 45            # el panel es mensual: con mas de mes y medio, se avisa
MIN_EVIDENCE_T = 2.0       # t del IC del multifactor para que cuente como evidencia
EXTREME_QUINTILE = 0.20    # Q1 (>= 0,80 de percentil) o Q5 (<= 0,20)


def evidence_from_report(text: str, name: str = "") -> dict | None:
    """IC, su t y el spread de quintiles, leidos de un reporte de backtest."""
    def grab(pattern: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    ic = grab(r"\| IC medio \| (-?[\d.]+) \|")
    t = grab(r"\| t \(Newey-West\) \| (-?[\d.]+) \|")
    if ic is None or t is None:
        return None
    return {"ic": ic, "ic_t": t, "months": grab(r"\| Meses evaluados \| (\d+) \|"),
            "spread": grab(r"Spread Q1-Q5: (-?[\d.]+)% mensual"), "spread_t": grab(r"mensual, t = (-?[\d.]+)"),
            "report": name, "validated": t >= MIN_EVIDENCE_T}


def latest_evidence(report_dir: Path) -> dict | None:
    # Solo reportes oficiales con fecha (backtest_AAAAMMDD_HHMM.md): un
    # contrafactual o una copia con otro nombre no puede colarse como evidencia.
    reports = sorted(p for p in report_dir.glob("backtest_*.md")
                     if re.fullmatch(r"backtest_\d{8}_\d{4}\.md", p.name))
    if not reports:
        return None
    return evidence_from_report(reports[-1].read_text(encoding="utf-8"), reports[-1].name)


def _panel_ticker(ticker: str, available: set[str]) -> str | None:
    """Ticker del panel (estilo SEC) para un ticker de Yahoo. Toronto: el listado de EE.UU."""
    t = ticker.upper()
    if t in available:
        return t
    if "." in t:
        base = t[: t.rindex(".")]
        for cand in (base, base.replace(".", "-")):
            if cand in available:
                return cand
    return None


def scores_from_panel(panel: pd.DataFrame, tickers: list[str], as_of: pd.Timestamp) -> dict:
    """Percentil del compuesto del multifactor de cada ticker en la ultima fecha. Funcion pura."""
    if panel is None or panel.empty or "score_composite" not in panel:
        return {"available": False, "reason": "no scored multifactor panel: run python main.py panel"}
    date = pd.Timestamp(panel["date"].max())
    last = panel[panel["date"] == date].dropna(subset=["score_composite"])
    pct = last.set_index("ticker")["score_composite"].rank(pct=True)
    factors = last.set_index("ticker")[[c for c in FACTOR_COLUMNS if c in last]]
    rows, missing = {}, {}
    available = set(pct.index)
    for t in tickers:
        key = _panel_ticker(t, available)
        if key is None:
            missing[t] = ("not in the multifactor universe on that date (it filters by listing, liquidity, size "
                          "and requires value and quality data)")
            continue
        composite = float(last.set_index("ticker").loc[key, "score_composite"])
        rows[t] = {"panel_ticker": key, "percentile": float(pct[key]), "composite": composite,
                   **{c.replace("score_", ""): float(factors.loc[key, c]) for c in factors.columns
                      if pd.notna(factors.loc[key, c])}}
    return {"available": True, "date": date, "universe": len(last), "rows": rows, "missing": missing,
            "stale": (as_of - date).days > STALE_DAYS}


# ---------------------------------------------------------------------------
#  Nombres fuera del universo: puntuados contra el, como referencia
# ---------------------------------------------------------------------------

_MONEY = ("revenue", "cogs", "net_income", "operating_income", "interest_expense", "ocf", "capex", "equity",
          "assets", "liabilities", "debt_long", "debt_short", "cash", "current_assets", "current_liabilities")


def exclusion_reasons(row: pd.Series, cfg, liquidity_cutoff: float | None = None) -> list[str]:
    """Reglas del universo del multifactor que el nombre no cumple en esa fecha.

    `liquidity_cutoff`: volumen en dolares del ultimo nombre que entro esa fecha.
    El universo se queda con los `universe.max_names` mas liquidos; sin esta
    regla, Gold.com (18,6 M USD al dia, por debajo del corte) salia como "cambio
    de ticker", que era falso.
    """
    g = lambda k: float(row[k]) if k in row and pd.notna(row[k]) else np.nan  # noqa: E731
    rules = [("price", "universe.min_price", "price {v:,.2f} USD below the {m:,.2f} minimum"),
             ("dollar_volume", "universe.min_dollar_volume", "trades {v:,.0f} USD a day, below {m:,.0f}"),
             ("market_cap", "universe.min_market_cap", "market cap {v:,.0f} USD, below {m:,.0f}"),
             ("history_months", "universe.min_history_months", "{v:.0f} months of history, below {m:.0f}")]
    out = []
    for col, key, text in rules:
        m = cfg.get(key)
        v = g(col)
        if m is not None and np.isfinite(v) and v < float(m):
            out.append(text.format(v=v, m=float(m)))
    dv = g("dollar_volume")
    if liquidity_cutoff is not None and np.isfinite(dv) and dv < liquidity_cutoff and not out:
        out.append(f"trades {dv:,.0f} USD a day, below the {liquidity_cutoff:,.0f} of the last of the "
                   f"{int(cfg.get('universe.max_names'))} most liquid names the model keeps (universe.max_names)")
    return out


def fundamentals_row(ticker: str, date: pd.Timestamp, wrds=None) -> tuple[pd.Series | None, str]:
    """Conceptos fundamentales point-in-time en `date`, en las unidades del panel.

    Primero la SEC (lo mismo que usa el multifactor); si no presenta alli
    (Toronto sin listado en EE.UU.), el TTM de Compustat con WRDS.
    """
    from sfc_tfsig.data import sec  # noqa: PLC0415

    t = ticker.upper()
    candidates = [t] + ([t[: t.rindex(".")]] if "." in t else [])
    try:
        listing = sec.listed_companies().drop_duplicates("ticker").set_index("ticker")["cik"]
        cik = next((listing[c] for c in candidates if c in listing.index), None)
    except Exception:  # noqa: BLE001
        cik = None
    if cik is not None:
        visible = sec.as_of(sec.company_observations(int(cik)), date)
        if len(visible):
            return visible.iloc[0], "SEC XBRL"
    if wrds is not None and getattr(wrds, "available", False) and t in wrds.ids.index and not wrds.fundq.empty:
        from . import wrds_data  # noqa: PLC0415

        ttm = wrds_data.ttm_panel(wrds.fundq, wrds.ids.loc[[t]], date)
        if len(ttm):
            r = ttm.iloc[0]
            row = pd.Series({k: float(r[k]) * 1e6 for k in _MONEY if k in r and pd.notna(r[k])})
            row["shares"] = float(r["shares_mm"]) * 1e6 if pd.notna(r.get("shares_mm")) else np.nan
            row["shares_growth"] = r.get("shares_growth")
            row["last_period_end"] = r.get("fiscal_period_end")
            return row, f"Compustat (WRDS, {r.get('currency', 'USD')})"
    return None, "no fundamentals at the SEC (and none from WRDS)"


def price_factors(close_usd: pd.Series, volume: pd.Series, date: pd.Timestamp, cfg) -> dict[str, float]:
    """Precio, momentum, volatilidad, volumen en dolares e historia en `date`, como el panel."""
    from sfc_tfsig.data import prices as prices_mod  # noqa: PLC0415

    c = close_usd.dropna()
    c = c[c.index <= date]
    if c.empty:
        return {}
    wide = c.to_frame("x")
    month = pd.Timestamp(date).to_period("M")

    def monthly(frame: pd.DataFrame) -> float:
        rows = frame.loc[frame.index.to_period("M") == month, "x"]
        return float(rows.iloc[-1]) if len(rows) else np.nan

    mom = prices_mod.momentum(wide, lookback_m=int(cfg.get("factors.momentum.lookback_m")),
                              skip_m=int(cfg.get("factors.momentum.skip_m")))
    vol = prices_mod.realized_volatility(wide, window_d=int(cfg.get("factors.lowvol.lookback_d")))
    dv = (c * volume.reindex(c.index)).iloc[-63:].median() if volume is not None else np.nan
    return {"price": float(c.iloc[-1]), "momentum": monthly(mom), "volatility": monthly(vol),
            "dollar_volume": float(dv) if pd.notna(dv) else np.nan,
            "history_months": (pd.Timestamp(date) - c.index[0]).days / 30.4375}


def score_outside(panel: pd.DataFrame, extra: pd.DataFrame, cfg) -> pd.DataFrame:
    """Scores del multifactor de nombres fuera del universo, contra la seccion cruzada de su fecha.

    Las mismas funciones del modelo (`financials.compute_all`,
    `composite.build_scores`): se anaden las filas a la ultima fecha y se
    puntua todo junto. El percentil se mide solo contra los nombres del
    universo. Funcion sin red: recibe el panel y las filas ya armadas.
    """
    from sfc_tfsig import financials  # noqa: PLC0415
    from sfc_tfsig.factors.composite import build_scores  # noqa: PLC0415

    date = pd.Timestamp(panel["date"].max())
    base = panel[panel["date"] == date].drop(columns=[c for c in panel.columns if c.startswith("score_")])
    rows = financials.compute_all(extra.assign(date=date).reset_index(drop=True))
    combined = pd.concat([base, rows], ignore_index=True)
    scored = build_scores(combined, cfg)
    universe = scored.iloc[: len(base)]["score_composite"].dropna()
    out = scored.iloc[len(base):].set_index("ticker")
    comp = out["score_composite"]
    out["percentile"] = [float((universe < c).mean() + 0.5 * (universe == c).mean()) if pd.notna(c) else np.nan
                         for c in comp]
    return out[["percentile", "score_composite", *[c for c in FACTOR_COLUMNS if c in out]]]


def outside_universe(tickers: list[str], panel: pd.DataFrame, cfg, clean, sectors: dict[str, str],
                     wrds=None) -> dict[str, dict]:
    """Por que cada nombre no esta en el universo, y donde quedaria si estuviera."""
    date = pd.Timestamp(panel["date"].max())
    last = panel[panel["date"] == date]
    cutoff = (float(last["dollar_volume"].min())
              if "dollar_volume" in last and len(last) >= int(cfg.get("universe.max_names")) else None)
    rows, notes = [], {}
    for t in tickers:
        fund, source = fundamentals_row(t, date, wrds)
        px = price_factors(clean.close[t], clean.volume.get(t), date, cfg) if t in clean.close else {}
        if fund is None or not px:
            notes[t] = {"reasons": [source if fund is None else "no prices on that date"], "scored": False}
            continue
        row = pd.concat([fund, pd.Series(px)])
        row["ticker"], row["sector"] = t, sectors.get(t, "Unknown")
        row["market_cap"] = row["price"] * row["shares"] if pd.notna(row.get("shares")) else np.nan
        rows.append(row)
        notes[t] = {"reasons": exclusion_reasons(row, cfg, cutoff), "source": source, "scored": True}
    if rows:
        scored = score_outside(panel, pd.DataFrame(rows), cfg)
        for t in scored.index:
            s = scored.loc[t]
            notes[t].update({"percentile": s["percentile"], "composite": s["score_composite"],
                             **{c.replace("score_", ""): float(s[c]) for c in FACTOR_COLUMNS
                                if c in s and pd.notna(s[c])}})
            if pd.isna(s["score_composite"]):
                notes[t]["reasons"].append("no value or no quality score: the model requires both "
                                           "(require_fundamental_score)")
    for t, n in notes.items():
        if n.get("scored") and not n["reasons"]:
            n["reasons"].append("not in the model's universe list on that date (listing, exchange or a "
                                "ticker change)")
    return notes


def load(tickers: list[str], as_of: pd.Timestamp, clean=None, sectors: dict[str, str] | None = None,
         wrds=None) -> dict:
    """Panel del multifactor de disco, evidencia del ultimo reporte y los nombres
    fuera del universo puntuados como referencia. Nunca lanza."""
    try:
        from sfc_tfsig import cli as mf_cli  # noqa: PLC0415
        from sfc_tfsig.config import load_config  # noqa: PLC0415
        from sfc_tfsig.data import cache  # noqa: PLC0415
        from sfc_tfsig.paths import REPORT_DIR  # noqa: PLC0415

        cfg = load_config(None)
        panel = cache.read_frame(mf_cli._panel_cache_name(cfg))  # noqa: SLF001
        out = scores_from_panel(panel, tickers, as_of)
        out["evidence"] = latest_evidence(REPORT_DIR)
        if out.get("available") and clean is not None and out["missing"]:
            try:
                out["outside"] = outside_universe(list(out["missing"]), panel, cfg, clean, sectors or {}, wrds)
            except Exception as exc:  # noqa: BLE001 -- sin la referencia, el resto sigue
                out["outside_error"] = f"{type(exc).__name__}: {exc}"
        return out
    except Exception as exc:  # noqa: BLE001 -- el plan sigue sin el multifactor
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


RELATIVE_BAND = 0.5   # z del factor de valor: por encima, barato frente al sector; por debajo de -0,5, caro


def relative_value(mf: dict | None, ticker: str) -> dict | None:
    """Valor relativo del nombre: el z sectorial del factor de valor del multifactor.

    Rendimientos (beneficio, flujo de caja, libros, ventas) frente a los de su
    sector, winsorizados. Es la contraparte relativa del DCF, que es absoluto.
    """
    if not mf or not mf.get("available"):
        return None
    info = mf.get("rows", {}).get(ticker) or (mf.get("outside") or {}).get(ticker) or {}
    z = info.get("value")
    if z is None or not np.isfinite(z):
        return None
    label = "CHEAP" if z > RELATIVE_BAND else "EXPENSIVE" if z < -RELATIVE_BAND else "IN LINE"
    return {"z": float(z), "label": label, "in_universe": ticker in mf.get("rows", {})}


def agreement(direction: int, percentile: float | None) -> int:
    """+1 si el multifactor respalda la direccion en su quintil extremo, -1 si la
    contradice ahi, 0 en medio o sin dato."""
    if percentile is None or not np.isfinite(percentile) or direction == 0:
        return 0
    if percentile >= 1 - EXTREME_QUINTILE:
        return 1 if direction > 0 else -1
    if percentile <= EXTREME_QUINTILE:
        return 1 if direction < 0 else -1
    return 0


# Lo que hizo la regla del plan segun el multifactor (`tools/qe_rule_studies.py
# multifactor`, jul-2023 a jun-2026, 43.196 operaciones). Si cambia la regla,
# se vuelve a correr y se actualiza aqui; las fichas lo citan tal cual.
RULE_EVIDENCE = {
    (1, 1): "on the plan's rule, longs in its Q1 made +0.31R against +0.17R for neutral ones (t 2.4)",
    (1, -1): "on the plan's rule, longs in its Q5 made +0.12R against +0.17R for neutral ones (not significant)",
    (-1, 1): ("but it does not raise conviction: on the plan's rule, shorts in its Q5 did no better than neutral "
              "ones (+0.02R, t 0.3) and still lost -0.10R; it is a model for picking longs"),
    (-1, -1): ("on the plan's rule, shorts against its Q1 lost -0.16R on average (t -2.3), 0.04R worse than "
               "neutral ones (t -1.5)"),
}


def conviction_step(direction: int, percentile: float | None) -> int:
    """Escalon de conviccion (y respaldo de un FLIP) que el multifactor puede dar.

    Medido con la regla del plan (`tools/qe_rule_studies.py multifactor`: 36
    cierres de mes jul-2023 a jun-2026, 600 nombres mas liquidos, 43.196
    operaciones), frente al grupo neutral:
    - largo en Q1: +0,132R, t 2,38 -> sube;
    - corto contra Q1: -0,043R, t -1,46 (con la regla de los largos era
      -0,106R, t -3,91) -> baja igual (prudente: solo resta);
    - largo contra Q5: -0,056R, t -0,74 -> baja igual (prudente: solo resta);
    - corto en Q5: +0,015R, t 0,28, y en absoluto -0,099R -> NO sube. El
      multifactor es un modelo de largos: su Q5 no encuentra buenos cortos.
    """
    agree = agreement(direction, percentile)
    return 0 if (direction < 0 and agree > 0) else agree


def usable(mf: dict | None) -> bool:
    """El multifactor cuenta como evidencia: panel al dia y validado en su reporte."""
    return bool(mf and mf.get("available") and not mf.get("stale")
                and (mf.get("evidence") or {}).get("validated"))
