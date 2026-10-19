"""Senal clara por nombre: que hacer, por que, y como ejecutarlo.

El plan (`trade_plan.build_plan`) decide direccion, niveles y tamano. Este
modulo lo convierte en una ficha por nombre que se puede leer y operar sin
abrir el resto del reporte:

- **Senal**: LONG, SHORT o NO TRADE, si va con research o contra, y la
  conviccion.
- **Por que**: los componentes que empujan a favor y en contra, con los numeros
  que hay detras (no z-scores): rendimiento por beneficio, ROE, revision de
  estimaciones, interes corto, momentum.
- **Fundamentales** del nombre, con aviso si un valor no es creible.
- **Como ejecutarlo**, paso a paso: orden de entrada, stop, objetivos, salida
  por tiempo. En un corto, antes: localizar el prestamo, coste, riesgo de
  squeeze, margen y dividendos. Si research decia LONG y el dato dice SHORT, la
  ficha explica como hacer el corto (y al reves).

Funciones puras: no descargan nada. Solo presentan numeros del plan y del
analisis; si un numero no esta, la ficha lo dice en vez de inventarlo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data.validation import MIN_SESSIONS

EXCHANGE = {".TO": "TSX", ".V": "TSX Venture", ".NE": "Cboe Canada", ".CN": "CSE", ".L": "LSE",
            ".PA": "Euronext Paris", ".DE": "Xetra", ".AS": "Euronext Amsterdam", ".SW": "SIX",
            ".T": "Tokyo", ".HK": "HKEX", ".AX": "ASX"}

# Reg T: un corto exige depositar el 50% de su valor ademas de lo que se cobra
# al vender. El mantenimiento lo fija el broker y sube en nombres volatiles.
SHORT_INITIAL_MARGIN = 0.5
# Por encima de esto de prestamo anual, la expectativa del backtest (que usa el
# coste del YAML) deja de valer: se avisa de no abrir el corto sin recalcular.
HARD_TO_BORROW = 0.05


def _pct(v, d: int = 1) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v * 100:.{d}f}%"


def _signed_pct(v, d: int = 1) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v * 100:+.{d}f}%"


def _num(v, d: int = 2) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.{d}f}"


def exchange_of(ticker: str) -> str:
    t = ticker.upper()
    if "." in t:
        return EXCHANGE.get(t[t.rindex("."):], "its listing exchange")
    return "its US listing"


def squeeze_risk(si_pct: float | None, days_to_cover: float | None) -> str:
    """Riesgo de squeeze de un corto por el interes corto existente."""
    si = si_pct if si_pct is not None and pd.notna(si_pct) else None
    dtc = days_to_cover if days_to_cover is not None and pd.notna(days_to_cover) else None
    if si is None and dtc is None:
        return "UNKNOWN"
    if (si or 0) >= 0.20 or (dtc or 0) >= 8:
        return "HIGH"
    if (si or 0) >= 0.10 or (dtc or 0) >= 4:
        return "MEDIUM"
    return "LOW"


def facts(ticker: str, features: pd.DataFrame, fundamentals: pd.DataFrame,
          liquidity: pd.DataFrame | None = None) -> pd.Series:
    """Todos los numeros de un nombre en una fila: caracteristicas y fundamentales."""
    parts = []
    for frame in (features, fundamentals, liquidity):
        if frame is not None and not frame.empty and ticker in frame.index:
            parts.append(frame.loc[ticker])
    if not parts:
        return pd.Series(dtype=object)
    row = pd.concat(parts)
    return row[~row.index.duplicated(keep="first")]


def fundamentals_lines(row: pd.Series) -> list[str]:
    """Fundamentales en lenguaje claro, con aviso si un valor no es creible."""
    g = row.get
    out = []
    ey = g("earnings_yield")
    if ey is not None and pd.notna(ey):
        pe = f" (P/E {1 / ey:.0f})" if ey > 0.005 else " (losses)" if ey < 0 else ""
        out.append(f"earnings yield {_pct(ey)}{pe}" + (" - verify" if abs(ey) > 0.5 else ""))
    fcf = g("fcf_yield")
    if fcf is not None and pd.notna(fcf):
        out.append(f"FCF yield {_pct(fcf)}" + (" - verify: working capital or one-off" if abs(fcf) > 0.5 else ""))
    if pd.notna(g("roe", np.nan)):
        out.append(f"ROE {_pct(g('roe'), 0)}" + (" - verify: tiny or negative equity" if abs(g("roe")) > 2 else ""))
    if pd.notna(g("gross_profitability", np.nan)):
        out.append(f"gross profit / assets {_pct(g('gross_profitability'), 0)}")
    if pd.notna(g("debt_to_equity", np.nan)):
        out.append(f"debt / equity {_num(g('debt_to_equity'), 1)}")
    if pd.notna(g("eps_revision_3m", np.nan)):
        word = "raised" if g("eps_revision_3m") > 0 else "cut" if g("eps_revision_3m") < 0 else "unchanged"
        out.append(f"EPS estimates {word} {_signed_pct(g('eps_revision_3m'))} in 3 months")
    if pd.notna(g("meanrec", np.nan)):
        n = f", {int(g('numrec'))} analysts" if pd.notna(g("numrec", np.nan)) else ""
        out.append(f"consensus {g('meanrec'):.1f} (1 strong buy - 5 sell{n})")
    if pd.notna(g("si_pct_float", np.nan)):
        dtc = f", {g('days_to_cover'):.1f} days to cover" if pd.notna(g("days_to_cover", np.nan)) else ""
        out.append(f"short interest {_pct(g('si_pct_float'))} of shares{dtc}")
    if pd.notna(g("mom_12_1", np.nan)):
        out.append(f"12-1 momentum {_signed_pct(g('mom_12_1'), 0)}, 1y return {_signed_pct(g('ret_1y'), 0)}")
    if pd.notna(g("vol_annual", np.nan)):
        out.append(f"volatility {_pct(g('vol_annual'), 0)}, beta {_num(g('beta'))}")
    if pd.notna(g("adv_usd", np.nan)):
        out.append(f"trades ~{g('adv_usd') / 1e6:,.0f}M USD a day")
    return out


def _component_evidence(name: str, row: pd.Series) -> str:
    """Los numeros detras de un componente del Quant Score."""
    g = row.get
    text = {
        "momentum": f"12-1 momentum {_signed_pct(g('mom_12_1'), 0)}, 6m {_signed_pct(g('mom_6m'), 0)}, "
                    f"3m {_signed_pct(g('mom_3m'), 0)}",
        "value": f"earnings yield {_pct(g('earnings_yield'))}, FCF yield {_pct(g('fcf_yield'))}, "
                 f"book/price {_num(g('book_to_price'))}",
        "quality": f"ROE {_pct(g('roe'), 0)}, gross profit/assets {_pct(g('gross_profitability'), 0)}, "
                   f"debt/equity {_num(g('debt_to_equity'), 1)}",
        "analyst": f"EPS estimates {_signed_pct(g('eps_revision_3m'))} in 3m, consensus {_num(g('meanrec'), 1)}",
        "short_interest": f"{_pct(g('si_pct_float'))} of shares sold short",
        "risk_adjusted_return": f"1y Sharpe {_num(g('sharpe_1y'))}",
        "volatility": f"volatility {_pct(g('vol_annual'), 0)}",
        "mean_reversion": f"{_num(g('zscore_20d'), 1)} sd from its 20-day mean",
        "beta": f"beta {_num(g('beta'))}",
        "liquidity": f"~{g('adv_usd') / 1e6:,.0f}M USD a day" if pd.notna(g("adv_usd", np.nan)) else "liquidity",
        "statistical": f"trend quality {_num(g('trend_quality'))}",
    }
    return text.get(name, name)


def why_lines(drivers: str, row: pd.Series) -> list[str]:
    """'momentum +0.28, value -0.22' -> 'momentum (+0.28): 12-1 momentum +45%, ...'."""
    out = []
    for item in [d.strip() for d in str(drivers or "").split(",") if d.strip()]:
        name, _, contrib = item.rpartition(" ")
        if name:
            out.append(f"{name.replace('_', ' ')} ({contrib}): {_component_evidence(name, row)}")
    return out


def execution_flags(stats: dict) -> list[str]:
    """Lo de la calidad de datos que cambia la ejecucion: el precio de entrada y el ATR.

    Los dias rellenados por festivos de otra bolsa no estan: son esperables en
    un nombre de Toronto y no cambian la orden de hoy.
    """
    g = stats.get
    out = []
    if (g("last_gap") or 0) >= 3:
        out.append(f"last price is {int(g('last_gap'))} sessions old: entry and levels are stale")
    if abs(g("last_jump") or 0) > 0.25:
        out.append(f"last close moved {g('last_jump'):+.0%} in a day: confirm it is the real price before trading")
    if (g("split_like") or 0) > 0:
        out.append(f"{int(g('split_like'))} split-like jumps in the history: check the split adjustment")
    if (g("extreme_returns") or 0) > 0:
        out.append(f"{int(g('extreme_returns'))} daily moves above 50%: ATR and odds include them")
    if (g("ohlc_repaired") or 0) > 0:
        out.append(f"{int(g('ohlc_repaired'))} bars with an inconsistent high/low were repaired")
    return out


def exit_date(levels_date: pd.Timestamp, horizon_d: int) -> pd.Timestamp:
    return pd.Timestamp(levels_date) + pd.offsets.BDay(horizon_d)


def execution_steps(ticker: str, side: int, shares: int, current: int, row: pd.Series, fx: float,
                    capital: float, horizon_d: int, levels_date: pd.Timestamp, borrow_cost: float,
                    short_facts: dict | None = None, hedge: dict | None = None,
                    earnings: pd.Timestamp | None = None) -> list[str]:
    """Pasos para abrir y gestionar la posicion, en el orden en que se hacen."""
    entry, stop, tp1, tp2, atr_v = (float(row[k]) for k in ("entry", "stop", "tp1", "tp2", "atr"))
    ccy = row.get("currency", "USD") or "USD"
    half = shares // 2
    rest = shares - half
    loss = shares * abs(entry - stop) * fx
    steps: list[str] = []
    if side < 0:
        sf = short_facts or {}
        known = sf.get("squeeze", "UNKNOWN") != "UNKNOWN"
        si_text = (f"Short interest {_pct(sf.get('si_pct'))} of shares, {_num(sf.get('days_to_cover'), 1)} days to "
                   f"cover: squeeze risk {sf['squeeze']}" if known else
                   "Short interest unknown here (needs WRDS; Compustat has US listings only): ask the broker")
        steps.append(
            f"Locate first: ask the broker to borrow {ticker}. {si_text}. The plan "
            f"assumes a borrow fee of {borrow_cost:.2%} a year; if the broker quotes more than {HARD_TO_BORROW:.0%} "
            "(hard to borrow), the expectation above no longer holds: do not short without recomputing.")
    if current and np.sign(current) != side:
        steps.append(f"Close the current {'long' if current > 0 else 'short'} first: "
                     f"{'SELL' if current > 0 else 'BUY TO COVER'} {abs(current):,} at market.")
    verb, exit_verb = ("BUY", "SELL") if side > 0 else ("SELL SHORT", "BUY TO COVER")
    where = f"on {exchange_of(ticker)}" + (f", in {ccy}" if ccy != "USD" else "")
    steps.append(f"{verb} {shares:,} {ticker} {where}: limit {entry:,.2f} (last close), day order. If it opens "
                 f"outside {entry - atr_v:,.2f} - {entry + atr_v:,.2f} (1 ATR), do not chase: entry, R and size "
                 "change; recompute.")
    stop_word = "SELL STOP" if side > 0 else "BUY STOP"
    steps.append(f"Right after the fill: {stop_word} {shares:,} at {stop:,.2f}, good till cancelled"
                 + (" (a short's stop is ABOVE the entry)" if side < 0 else "") +
                 f". If hit: about -{loss:,.0f} USD ({loss / capital:.2%} of capital), more on a gap.")
    steps.append(f"TP1: {exit_verb} LIMIT {half:,} at {tp1:,.2f}, good till cancelled. When it fills, move the stop "
                 f"on the other {rest:,} to {entry:,.2f} (break-even).")
    steps.append(f"TP2: {exit_verb} LIMIT {rest:,} at {tp2:,.2f}, good till cancelled; cancel the stop when it fills "
                 "(one-cancels-other if the broker has it).")
    steps.append(f"Time exit: if nothing is hit by {exit_date(levels_date, horizon_d):%Y-%m-%d} "
                 f"({horizon_d} sessions), close the rest at market.")
    if side < 0:
        steps.append(f"Margin: about {SHORT_INITIAL_MARGIN * shares * entry * fx:,.0f} USD of own equity (Reg T: "
                     "50% of the short value on top of the sale proceeds); the broker can raise it on a volatile "
                     "name. Any dividend paid while short is owed by you.")
    if ccy != "USD":
        steps.append(f"Prices are in {ccy}; size and risk are in USD at {fx:,.4f} USD per {ccy}.")
    if hedge:
        steps.append(f"Optional market hedge: this position alone carries beta {hedge['beta']:+.2f}, like "
                     f"{hedge['beta'] * shares * entry * fx * side:+,.0f} USD of {hedge['ticker']}. To keep only the "
                     f"view on {ticker}: {hedge['action']} {hedge['shares']:,} {hedge['ticker']} at ~{hedge['price']:,.2f}.")
    if earnings is not None and pd.notna(earnings):
        if pd.Timestamp(earnings) <= exit_date(levels_date, horizon_d):
            steps.append(f"Earnings on {pd.Timestamp(earnings):%Y-%m-%d}, before the time exit: a gap can jump the "
                         "stop. Consider half size, or wait for the report.")
    else:
        steps.append("Next earnings date unknown: check it before opening; an earnings gap jumps the stop.")
    return steps


def standalone_hedge(beta: float, shares: int, side: int, entry: float, fx: float, bench: str,
                     bench_price: float) -> dict | None:
    """Benchmark que deja una posicion suelta sin beta de mercado."""
    if not (bench_price > 0) or beta is None or not np.isfinite(beta) or shares <= 0:
        return None
    exposure = beta * shares * entry * fx * side
    n = round(abs(exposure) / bench_price)
    if n == 0:
        return None
    return {"beta": float(beta), "ticker": bench, "shares": n, "price": float(bench_price),
            "action": "SELL SHORT" if exposure > 0 else "BUY"}


def headline(t: str, row: pd.Series, p) -> str:
    """Una frase: que hacer con el nombre y por que."""
    research, score = row["research"], row["aligned_score"]
    if row["signal"] == "NO TRADE":
        if row.get("score_verdict") == "FLIP":
            return (f"NO TRADE {t}: data points against research {research} (aligned score {score:+.2f}), but "
                    "going the other way needs evidence that the score predicts. Review the thesis.")
        return (f"NO TRADE {t}: data contradicts research {research} (aligned score {score:+.2f}), not enough to "
                "take the other side. Review the thesis.")
    against = row.get("against_research")
    lead = f"{row['signal']} {t}: " + (f"AGAINST research ({research})" if against else f"with research ({research})")
    size = ""
    if row["verdict"] == "REDUCE":
        size = f"; data is neutral, so {p.reduce_size:.0%} size"
    elif row.get("unproven_flip"):
        size = f"; the score has not shown it predicts here, so {p.reduce_size:.0%} size"
    return f"{lead}. Conviction {row['conviction']} (aligned score {score:+.2f}){size}."


def build_cards(result) -> dict[str, dict]:
    """Una ficha por nombre de la canasta, en el orden del plan, mas los excluidos."""
    tp = result.trade_plan
    if not tp:
        return {}
    plan, orders, p = tp["plan"], tp["orders"], tp["params"]
    s = result.settings
    capital = s.initial_capital
    bench = s.benchmark
    bench_close = result.clean.close[bench].dropna() if bench in result.clean.close else pd.Series(dtype=float)
    bench_price = float(bench_close.iloc[-1]) if len(bench_close) else float("nan")
    earnings = getattr(result, "earnings", None) or {}
    cards: dict[str, dict] = {}
    for t, row in plan.iterrows():
        row_facts = facts(t, result.features, result.fundamentals, getattr(result, "liquidity", None))
        side = {"LONG": 1, "SHORT": -1}.get(row["signal"], 0)
        current = int(orders.loc[t, "current"]) if t in orders.index else 0
        booked = int(abs(orders.loc[t, "target"])) if (t in orders.index and tp.get("has_book")) else 0
        shares, sized_by = (booked, "book") if booked else (int(row.get("solo_shares", 0)), "standalone")
        flags = execution_flags(result.clean.stats.get(t, {}))
        card = {
            "ticker": t, "research": row["research"], "signal": row["signal"], "verdict": row["verdict"],
            "against_research": bool(row.get("against_research")), "conviction": row.get("conviction", ""),
            "aligned_score": row["aligned_score"], "headline": headline(t, row, p),
            "why_for": why_lines(row.get("drivers_for", ""), row_facts),
            "why_against": why_lines(row.get("drivers_against", ""), row_facts),
            "fundamentals": fundamentals_lines(row_facts), "data_flags": flags,
            "shares": shares, "sized_by": sized_by, "steps": [],
            "size_mult": p.reduce_size if (row["verdict"] == "REDUCE" or row.get("unproven_flip")) else 1.0,
            "earnings": earnings.get(t),
        }
        if side and shares > 0:
            short_facts = {"si_pct": row_facts.get("si_pct_float"), "days_to_cover": row_facts.get("days_to_cover")}
            short_facts["squeeze"] = squeeze_risk(short_facts["si_pct"], short_facts["days_to_cover"])
            beta = row_facts.get("beta")
            hedge = (standalone_hedge(float(beta) if beta is not None and pd.notna(beta) else float("nan"),
                                      shares, side, float(row["entry"]), float(row["fx"]), bench, bench_price)
                     if sized_by == "standalone" else None)
            card["steps"] = execution_steps(t, side, shares, current, row, float(row["fx"]), capital, p.horizon_d,
                                            tp.get("levels_date", result.as_of), s.borrow_cost, short_facts,
                                            hedge, earnings.get(t))
            card["notional"] = shares * float(row["entry"]) * float(row["fx"])
            card["beta"] = float(beta) if beta is not None and pd.notna(beta) else None
            card["risk_pct"] = shares * abs(float(row["entry"]) - float(row["stop"])) * float(row["fx"]) / capital
        cards[t] = card
    for t in result.excluded:
        why = "insufficient data"
        if t in result.quality.index:
            n = result.quality.loc[t, "sessions"]
            why = (f"only {int(n)} sessions of price history; the analysis needs {MIN_SESSIONS}" if n else
                   str(result.quality.loc[t, "notes"]) or "no price data")
        cards[t] = {"ticker": t, "research": "LONG" if t in s.longs else "SHORT", "signal": "NOT ANALYZED",
                    "headline": f"NOT ANALYZED {t}: {why}.", "why_for": [], "why_against": [], "fundamentals": [],
                    "data_flags": [], "steps": [], "shares": 0, "conviction": "", "against_research": False}
    return cards


def standalone_total(cards: dict[str, dict], bench: str, bench_price: float) -> dict | None:
    """Todas las posiciones sueltas juntas: nocional, beta en dolares, riesgo y una sola cobertura.

    Sin libro long/short, cada nombre se dimensiona por su cuenta. Sumadas son
    una apuesta direccional: hay que verla entera antes de abrirlas.
    """
    live = [c for c in cards.values() if c.get("steps") and c.get("sized_by") == "standalone"]
    if not live:
        return None
    signed = [c["notional"] * (1 if c["signal"] == "LONG" else -1) for c in live]
    betas = [c.get("beta") if c.get("beta") is not None and pd.notna(c.get("beta")) else 1.0 for c in live]
    beta_usd = sum(n * b for n, b in zip(signed, betas, strict=True))
    n_bench = round(abs(beta_usd) / bench_price) if bench_price > 0 else 0
    return {"names": [c["ticker"] for c in live], "long": sum(n for n in signed if n > 0),
            "short": -sum(n for n in signed if n < 0), "net": sum(signed), "beta_usd": beta_usd,
            "risk_pct": sum(c["risk_pct"] for c in live), "hedge_shares": n_bench,
            "hedge_action": "SELL SHORT" if beta_usd > 0 else "BUY", "bench": bench, "bench_price": bench_price}


def card_size_note(card: dict, max_risk: float) -> str:
    """De donde sale el tamano de la ficha."""
    if not card.get("shares"):
        return ""
    if card.get("sized_by") == "book":
        return "size from the long/short book"
    mult = card.get("size_mult", 1.0)
    cut = f" x {mult:.0%}" if mult < 1 else ""
    return (f"standalone size: no long/short book to share, so the stop costs {max_risk:.0%} of capital{cut}; "
            "this is a directional position")
