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
                    earnings: pd.Timestamp | None = None, earnings_move: dict | None = None) -> list[str]:
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
            size = ""
            if earnings_move:
                stop_pct = abs(entry - stop) / entry
                verdict = ("larger than" if earnings_move["median"] > stop_pct else "within")
                size = (f" Its typical earnings-day move is {earnings_move['median']:.1%} (1 in 4 above "
                        f"{earnings_move['p75']:.1%}; {earnings_move['source']}), {verdict} the "
                        f"{stop_pct:.1%} to the stop.")
            steps.append(f"Earnings on {pd.Timestamp(earnings):%Y-%m-%d}, before the time exit: a gap can jump the "
                         f"stop.{size} Consider half size, or wait for the report.")
    else:
        steps.append("Next earnings date unknown: check it before opening; an earnings gap jumps the stop.")
    return steps


def borrow_tier(si_pct: float | None) -> tuple[str, float] | None:
    """Tramo probable del coste de prestamo por el interes en corto.

    Regla general, no un dato del broker: la mayoria de las acciones se presta
    a tasa general (~0,25-0,5% al ano) y las muy vendidas en corto se encarecen
    (D'Avolio, 2002). Devuelve (descripcion, tasa central del tramo).
    """
    if si_pct is None or not np.isfinite(si_pct):
        return None
    if si_pct < 0.05:
        return "general collateral (about 0.25-0.5% a year)", 0.004
    if si_pct < 0.15:
        return "possibly 'warm' (about 0.5-5% a year)", 0.02
    return "likely hard to borrow (5% a year or more)", 0.08


def time_window(ticker: str, row: pd.Series, side: int, shares: int, fx: float, levels_date: pd.Timestamp,
                horizon_d: int, borrow_cost: float, ex_dividend: pd.Timestamp | None = None,
                si_pct: float | None = None) -> list[str]:
    """Rango de tiempo de la posicion: cuando abrir, cuanto suele durar, cuando
    revisar, cuando cerrar si o si, y lo que cuesta mantenerla.

    Todo sale de la misma regla abierta cada dia de los ultimos anos en este
    valor (`trade_plan.holding_window`), no de una regla general.
    """
    start = pd.Timestamp(levels_date) + pd.offsets.BDay(1)
    day = lambda n: start + pd.offsets.BDay(int(round(n)))  # noqa: E731
    lines = [f"Open on {start:%Y-%m-%d} (next session). If it is not filled within 2 sessions, recompute the "
             "levels: entry and R have moved."]
    p25, p75, med = row.get("hold_p25"), row.get("hold_p75"), row.get("hold_median")
    if p25 is not None and pd.notna(p25):
        lines.append(f"Typical holding for this rule on {ticker}: {p25:.0f}-{p75:.0f} sessions, from "
                     f"{day(p25):%Y-%m-%d} to {day(p75):%Y-%m-%d} (median {med:.0f}).")
    review, late = row.get("review_day"), row.get("late_tp1_rate")
    if review is not None and pd.notna(review):
        tp1 = row.get("tp1_median_days")
        cover = "cover" if side < 0 else "sell"
        head = (f"TP1 usually comes in about {tp1:.0f} sessions. Review on {day(review):%Y-%m-%d} "
                f"({review:.0f} sessions): if TP1 has not been reached, ")
        oos_line = ""
        imp = row.get("oos_ts_improvement")
        if imp is not None and pd.notna(imp):
            oos_line = (f" Out of sample (review day picked on trades before {pd.Timestamp(row['oos_ts_split']):%Y-%m},"
                        f" tested after): E(R) {row['oos_ts_exp_r_plain']:+.2f}R without the time stop, "
                        f"{row['oos_ts_exp_r_time_stop']:+.2f}R with it.")
        if row.get("cut_at_review") and pd.notna(late):
            ts, plain = row.get("bt_exp_r_time_stop"), row.get("bt_exp_r")
            gain = (f" With that time stop the rule's E(R) on {ticker} was {ts:+.2f}R against {plain:+.2f}R "
                    "without it (in-sample: the review day comes from the same trades)."
                    if ts is not None and pd.notna(ts) and plain is not None and pd.notna(plain) else "")
            lines.append(head + f"{cover} the position: only {late:.0%} of past trades still open then reached it, "
                         "below the break-even for TP1." + gain + oos_line)
        elif pd.notna(late):
            lines.append(head + f"keep the stop and hold: {late:.0%} of past trades still open then reached it later."
                         + oos_line)
    hard = day(horizon_d)
    lines.append(f"Hard exit: {'cover' if side < 0 else 'close'} whatever is left by {hard:%Y-%m-%d} "
                 f"({horizon_d} sessions), whatever the price.")
    if side < 0:
        notional = shares * float(row["entry"]) * fx
        if med is not None and pd.notna(med) and notional > 0:
            lines.append(f"Borrow over the typical holding: about {notional * borrow_cost * med / 252:,.0f} USD at the "
                         f"assumed {borrow_cost:.2%} a year (about {notional * borrow_cost * 21 / 252:,.0f} USD per "
                         "extra month).")
            tier = borrow_tier(si_pct)
            if tier and tier[1] > borrow_cost:
                lines.append(f"With {si_pct:.1%} of the shares sold short, borrow is {tier[0]}: at {tier[1]:.0%} the "
                             f"typical holding would cost about {notional * tier[1] * med / 252:,.0f} USD. Get the "
                             "real fee from the broker before opening.")
        if ex_dividend is not None and pd.notna(ex_dividend) and start <= pd.Timestamp(ex_dividend) <= hard:
            lines.append(f"Ex-dividend on {pd.Timestamp(ex_dividend):%Y-%m-%d}, inside the window: a short open that "
                         "day pays the dividend; it is not in the expectation above.")
    return lines


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
    elif row.get("flip_backed_by_multifactor"):
        size = "; full size because the validated multifactor model backs the other side"
    return f"{lead}. Conviction {row['conviction']} (aligned score {score:+.2f}){size}."


def _earnings_move(result, ticker: str) -> dict | None:
    """Movimiento tipico de resultados, con las fechas de Compustat si WRDS las trajo."""
    from .factors.technical import earnings_moves  # noqa: PLC0415

    dates = None
    w = getattr(result, "wrds", None)
    if w is not None and w.available and not w.fundq.empty and ticker in w.ids.index:
        dates = w.fundq.loc[w.fundq["gvkey"] == w.ids.loc[ticker, "gvkey"], "rdq"]
    local = result.clean.local_prices(ticker)[0].dropna()
    return earnings_moves(local.pct_change(fill_method=None), dates)


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
        from .factors import technical  # noqa: PLC0415
        from .valuation.engine import summary  # noqa: PLC0415

        close_l, high_l, low_l = result.clean.local_prices(t)
        vol = result.clean.volume.get(t)
        tech = technical.technical_view(close_l, high_l, low_l, vol)
        card["technical"] = technical.describe(tech)
        card["timing"] = technical.timing_advice(tech, side, float(row["entry"]), float(row["stop"]),
                                                 float(row["tp1"])) if side else []
        val = (getattr(result, "valuation", None) or {}).get(t)
        from .data.multifactor import relative_value  # noqa: PLC0415

        card["valuation"], card["valuation_vs_signal"] = valuation_lines(
            row["signal"], summary(val) if val else None, relative_value(getattr(result, "multifactor", None), t))
        mf_line = multifactor_line(t, row, getattr(result, "multifactor", None))
        if mf_line:
            card["valuation"] = [mf_line, *card["valuation"]]
        if side and shares > 0:
            short_facts = {"si_pct": row_facts.get("si_pct_float"), "days_to_cover": row_facts.get("days_to_cover")}
            short_facts["squeeze"] = squeeze_risk(short_facts["si_pct"], short_facts["days_to_cover"])
            beta = row_facts.get("beta")
            hedge = (standalone_hedge(float(beta) if beta is not None and pd.notna(beta) else float("nan"),
                                      shares, side, float(row["entry"]), float(row["fx"]), bench, bench_price)
                     if sized_by == "standalone" else None)
            card["steps"] = execution_steps(t, side, shares, current, row, float(row["fx"]), capital, p.horizon_d,
                                            tp.get("levels_date", result.as_of), s.borrow_cost, short_facts,
                                            hedge, earnings.get(t), _earnings_move(result, t))
            # Rango de tiempo para todo lo que se opera; en los cortos lleva
            # ademas prestamo y dividendos, porque ahi el tiempo cuesta dinero.
            if side:
                card["time_window"] = time_window(t, row, side, shares, float(row["fx"]),
                                                  tp.get("levels_date", result.as_of), p.horizon_d, s.borrow_cost,
                                                  (getattr(result, "ex_dividend", None) or {}).get(t),
                                                  short_facts.get("si_pct"))
            card["notional"] = shares * float(row["entry"]) * float(row["fx"])
            card["beta"] = float(beta) if beta is not None and pd.notna(beta) else None
            card["risk_pct"] = shares * abs(float(row["entry"]) - float(row["stop"])) * float(row["fx"]) / capital
        cards[t] = card
    for t in result.excluded:
        cards[t] = limited_card(result, t, p)
    return cards


LIMITED_MIN_SESSIONS = 30


def limited_card(result, t: str, p) -> dict:
    """Ficha de un nombre excluido del analisis: sin score, sin backtest y sin tamano.

    Con 30 sesiones o mas (una salida a bolsa reciente) se dan los niveles por
    ATR del lado de research, la lectura tecnica y el DCF si hay estados
    financieros: lo que se puede medir sin inventar lo que no.
    """
    from .factors import technical  # noqa: PLC0415
    from .trade_plan import atr, levels  # noqa: PLC0415
    from .valuation.engine import summary  # noqa: PLC0415

    s = result.settings
    research = "LONG" if t in s.longs else "SHORT"
    sessions = int(result.quality.loc[t, "sessions"]) if t in result.quality.index else 0
    why = (f"only {sessions} sessions of price history; the full analysis needs {MIN_SESSIONS}" if sessions else
           (str(result.quality.loc[t, "notes"]) if t in result.quality.index else "") or "no price data")
    card = {"ticker": t, "research": research, "signal": "NOT ANALYZED", "headline": f"NOT ANALYZED {t}: {why}.",
            "why_for": [], "why_against": [], "fundamentals": [], "data_flags": [], "steps": [], "shares": 0,
            "conviction": "", "against_research": False}
    if sessions < LIMITED_MIN_SESSIONS or t not in result.clean.close:
        return card
    close_l, high_l, low_l = result.clean.local_prices(t)
    close = close_l.dropna()
    high = high_l.reindex(close.index).fillna(close)
    low = low_l.reindex(close.index).fillna(close)
    a = atr(high, low, close).dropna()
    side = 1 if research == "LONG" else -1
    ccy = result.clean.currency.get(t, "USD")
    if len(a):
        lv = levels(float(close.iloc[-1]), float(a.iloc[-1]), side, p)
        card["fundamentals"] = [f"levels for research's {research} if taken by judgment ({ccy}): entry "
                                f"{lv['entry']:,.2f}, stop {lv['stop']:,.2f} ({lv['stop_pct']:.1%}), TP1 "
                                f"{lv['tp1']:,.2f}, TP2 {lv['tp2']:,.2f}"]
    tech = technical.technical_view(close_l, high_l, low_l, result.clean.volume.get(t))
    card["technical"] = technical.describe(tech) if tech.get("available") else []
    val = (getattr(result, "valuation", None) or {}).get(t)
    from .data.multifactor import relative_value  # noqa: PLC0415

    card["valuation"], _ = valuation_lines(research, summary(val) if val else None,
                                           relative_value(getattr(result, "multifactor", None), t))
    card["headline"] = (f"LIMITED VIEW {t}: {why}. No Quant Score, rule backtest or size: the levels, technicals "
                        "and DCF below inform a judgment call, they are not a model signal.")
    return card


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


def relative_check(stance: str, rel: dict | None) -> str:
    """El DCF (absoluto) frente al factor de valor del multifactor (relativo al sector)."""
    if not rel:
        return ""
    head = (f"Relative value (multifactor value factor, sector z {rel['z']:+.2f}"
            + ("" if rel["in_universe"] else ", name outside its universe") + f"): {rel['label'].lower()} against "
            "its sector on earnings, cash-flow, book and sales yields")
    dcf_cheap = {"UNDERVALUED": 1, "OVERVALUED": -1}.get(stance, 0)
    rel_cheap = {"CHEAP": 1, "EXPENSIVE": -1}.get(rel["label"], 0)
    if dcf_cheap and rel_cheap:
        verdict = ("agrees with the DCF" if dcf_cheap == rel_cheap else
                   "disagrees with the DCF: absolute and relative valuation point opposite ways, so treat both "
                   "with caution")
        return f"{head}; {verdict}."
    return head + "."


def valuation_lines(signal: str, val: dict | None, rel: dict | None = None) -> tuple[list[str], str]:
    """El DCF del nombre en frases, y si respalda o contradice la senal."""
    if not val:
        line = relative_check("", rel)
        return ([line] if line else []), ""
    if not val.get("available"):
        return [f"DCF not available: {val.get('reason')}"], ""
    ccy = val.get("trading_currency", "USD")
    cases = val["value_cases"]
    lines = [f"DCF ({val['source']}, FY {val['fiscal_end']}): base {cases['base']:,.2f} {ccy} vs price "
             f"{val['price']:,.2f} ({val['upside_base']:+.0%}): {val['stance']}. "
             f"Bear / bull {cases['bear']:,.2f} / {cases['bull']:,.2f}, probability-weighted "
             f"{val['value_expected']:,.2f}; "
             f"WACC {val['wacc']:.1%}, terminal g {val['terminal_growth']:.1%}"
             + (f", terminal value {val['tv_share']:.0%} of EV." if val.get("tv_share") is not None
                and np.isfinite(val["tv_share"]) else ", enterprise value not positive.")]
    if val.get("implied_cagr") is not None:
        lines.append(f"The price implies {val['implied_cagr']:+.1%} revenue growth a year for {val['years']} years; "
                     f"the base case assumes {val['base_cagr']:+.1%}.")
    elif val.get("reverse_note"):
        lines.append(f"Reverse DCF: {val['reverse_note']}.")
    if val.get("mc_prob_above_price") is not None:
        lines.append(f"Monte Carlo: value above the price in {val['mc_prob_above_price']:.0%} of draws.")
    lines.extend(f"DCF check: {f}" for f in val.get("flags", []))
    stance = val["stance"]
    rel_line = relative_check(stance, rel)
    if rel_line:
        lines.append(rel_line)
    if val.get("critical"):
        return lines, "valuation is not decision-grade for this name: see the DCF checks"
    agree = ""
    if signal in ("LONG", "SHORT") and stance in ("UNDERVALUED", "OVERVALUED"):
        backs = (signal == "LONG") == (stance == "UNDERVALUED")
        agree = (f"valuation {'supports' if backs else 'contradicts'} the {signal}: the DCF says "
                 f"{stance.lower()}")
    elif signal in ("LONG", "SHORT"):
        agree = "valuation is neutral for this trade: the DCF is within the fair-value band"
    return lines, agree


def multifactor_line(t: str, row: pd.Series, mf: dict | None) -> str:
    """Donde esta el nombre en el modelo multifactor validado, y si respalda la senal."""
    if not mf or not mf.get("available"):
        return ""
    info = mf.get("rows", {}).get(t)
    if not info:
        out = (mf.get("outside") or {}).get(t)
        if out and out.get("scored") and pd.notna(out.get("percentile")):
            parts = ", ".join(f"{k} {out[k]:+.2f}" for k in ("value", "quality", "momentum", "lowvol") if k in out)
            return (f"Multifactor model: outside its universe ({'; '.join(out['reasons'])}). Scored against it for "
                    f"reference only: {out['percentile']:.0%} percentile ({parts}); not used as evidence, the "
                    "validation covers the universe only.")
        if out:
            return f"Multifactor model: outside its universe ({'; '.join(out['reasons'])}); no score."
        reason = mf.get("missing", {}).get(t)
        return f"Multifactor model: {reason}." if reason else ""
    ev = mf.get("evidence") or {}
    proof = (f"validated: monthly IC {ev['ic']:.3f}, t {ev['ic_t']:.1f}, over {ev['months']:.0f} months"
             if ev.get("ic") is not None else "no backtest report to validate it")
    parts = ", ".join(f"{k} {info[k]:+.2f}" for k in ("value", "quality", "momentum", "lowvol") if k in info)
    q = 5 - min(int(info["percentile"] * 5), 4)
    side = {"LONG": 1, "SHORT": -1}.get(row.get("signal"), 0)
    from .data.multifactor import agreement  # noqa: PLC0415

    verdict = {1: "backs the trade", -1: "contradicts the trade", 0: "neutral for the trade"}[agreement(side, info["percentile"])]
    stale = " (panel more than 45 days old: not used as evidence)" if mf.get("stale") else ""
    return (f"Multifactor model ({proof}): {info['percentile']:.0%} percentile of {mf['universe']:,} names on "
            f"{pd.Timestamp(mf['date']):%Y-%m-%d}, quintile Q{q} ({parts}); {verdict if side else 'no trade'}.{stale}")


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
