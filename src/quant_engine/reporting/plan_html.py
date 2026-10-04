"""Informe del plan de operacion: una pagina HTML autocontenida.

Se abre en el navegador y se imprime a PDF desde ahi (Ctrl+P). CSS y graficos
van dentro del fichero (PNG en base64): se puede mandar por correo sin una
carpeta de imagenes al lado.

Solo presenta lo que calculo `trade_plan.build_plan`. Si un numero no esta en
el plan, no se inventa aqui.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..analysis import AnalysisResult  # noqa: E402

LONG_COLOR, SHORT_COLOR, TP_COLOR, INK = "#2f6db5", "#c0392b", "#1e8449", "#1b1b1b"

_BADGE = {
    "BUY": "long", "FLIP TO LONG": "long",
    "SELL SHORT": "short", "FLIP TO SHORT": "short",
    "SELL": "trim", "BUY TO COVER": "trim",
    "HOLD": "hold", "NO TRADE": "none", "NOT ANALYZED": "none",
}

_VERDICT_TEXT = {
    "CONFIRM": "Data supports research: trade as proposed.",
    "REDUCE": "Data is neutral: trade at reduced size.",
    "NO TRADE": "Data contradicts research: do not open; review the thesis.",
    "FLIP": "Data strongly contradicts research: the model takes the other side.",
}

CSS = """
:root { --ink:#1b1b1b; --muted:#5f6368; --line:#e3e5e8; --bg:#ffffff; --panel:#f6f7f9;
        --long:#2f6db5; --short:#c0392b; --tp:#1e8449; --trim:#b9770e; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:14px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:32px 24px 48px; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-0.01em; }
h2 { font-size:17px; margin:36px 0 10px; padding-bottom:6px; border-bottom:1px solid var(--line); }
.sub { color:var(--muted); margin:0; }
.meta { display:flex; flex-wrap:wrap; gap:6px 18px; color:var(--muted); font-size:13px; margin-top:10px; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin-top:22px; }
.kpi { background:var(--panel); border-radius:8px; padding:12px 14px; }
.kpi .v { font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }
.kpi .l { color:var(--muted); font-size:12px; }
.wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { padding:7px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th { font-size:12px; color:var(--muted); font-weight:600; background:var(--panel); }
td.l, th.l { text-align:left; }
td.wrapcell { white-space:normal; text-align:left; min-width:220px; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }
.b-long { background:#e3edf9; color:var(--long); }
.b-short { background:#f9e3e1; color:var(--short); }
.b-trim { background:#fbf0dc; color:var(--trim); }
.b-hold { background:#eceff1; color:#37474f; }
.b-none { background:#f3f3f3; color:#8a8a8a; }
.up { color:var(--tp); } .down { color:var(--short); } .muted { color:var(--muted); }
.note { background:var(--panel); border-left:3px solid var(--long); padding:10px 14px; border-radius:4px; }
.warn { border-left-color:var(--trim); }
.alert { border-left-color:var(--short); background:#fdecea; margin-top:18px; }
.charts { display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:14px; }
.charts figure { margin:0; border:1px solid var(--line); border-radius:8px; padding:8px; }
.charts img { width:100%; height:auto; display:block; }
ul { padding-left:20px; } li { margin:3px 0; }
footer { margin-top:36px; color:var(--muted); font-size:12px; }
@media print { main { padding:0; } h2 { break-after:avoid; } figure, tr { break-inside:avoid; } }
"""


def _money(v) -> str:
    return "" if v is None or pd.isna(v) else f"{v:,.0f}"


def _price(v) -> str:
    return "" if v is None or pd.isna(v) else f"{v:,.2f}"


def _pct(v, d: int = 1) -> str:
    return "" if v is None or pd.isna(v) else f"{v * 100:.{d}f}%"


def _badge(signal: str) -> str:
    return f'<span class="badge b-{_BADGE.get(signal, "none")}">{escape(signal)}</span>'


def _odds(p: float, breakeven: float) -> str:
    if p is None or pd.isna(p):
        return ""
    cls = "up" if p > breakeven else "down"
    return f'<span class="{cls}">{p:.0%}</span>'


def _chart(r: AnalysisResult, ticker: str, o: pd.Series) -> str:
    """Precio de los ultimos 6 meses con entrada, stop y objetivos."""
    close = r.clean.close[ticker].dropna().iloc[-126:]
    side = 1 if o["target"] > 0 else -1
    color = LONG_COLOR if side > 0 else SHORT_COLOR
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ax.plot(close.index, close.values, color=INK, lw=1.2)
    x0, x1 = close.index[-1], close.index[-1] + pd.Timedelta(days=45)
    ax.fill_between([x0, x1], o["price"], o["stop"], color=SHORT_COLOR, alpha=0.10, lw=0)
    ax.fill_between([x0, x1], o["price"], o["tp2"], color=TP_COLOR, alpha=0.08, lw=0)
    for level, label, c, ls in ((o["price"], "entry", "#555555", "-"), (o["stop"], "stop", SHORT_COLOR, "--"),
                                (o["tp1"], "TP1", TP_COLOR, "--"), (o["tp2"], "TP2", TP_COLOR, ":")):
        ax.hlines(level, close.index[0], x1, colors=c, linestyles=ls, lw=1)
        ax.text(x1, level, f" {label} {level:,.2f}", va="center", fontsize=7.5, color=c)
    ax.set_xlim(close.index[0], x1 + pd.Timedelta(days=30))
    ax.set_title(f"{ticker}  {o['signal']}  -> {int(o['target']):+,} sh", fontsize=10, color=color, loc="left")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build(r: AnalysisResult) -> str:
    tp = r.trade_plan
    s = r.settings
    plan, orders, p = tp["plan"], tp["orders"], tp["params"]
    hedge = tp.get("hedge")
    levels_date = pd.Timestamp(tp.get("levels_date", r.as_of))
    wrds_ok = bool(r.wrds and r.wrds.available)

    to_send = orders[~orders["signal"].isin(["HOLD", "NO TRADE", "NOT ANALYZED"])]
    beta_after = hedge["beta_after"] if hedge else tp["beta"]
    n_corr = int((plan["verdict"] != "CONFIRM").sum())

    out: list[str] = []
    out.append(f"<h1>Trade Plan</h1><p class='sub'>Long/short research basket &middot; levels from the "
               f"{levels_date:%Y-%m-%d} close</p>")
    out.append("<div class='meta'>"
               f"<span>Research LONG: <b>{escape(' '.join(r.longs))}</b></span>"
               f"<span>Research SHORT: <b>{escape(' '.join(r.shorts))}</b></span>"
               f"<span>Capital: {_money(s.initial_capital)}</span>"
               f"<span>Construction: {escape(s.construction)}</span>"
               f"<span>Benchmark: {escape(s.benchmark)}</span>"
               f"<span>Data: Yahoo prices{' + WRDS (Compustat, IBES, short interest, Fama-French)' if wrds_ok else ''}</span>"
               "</div>")

    missing = tp.get("wrds_missing")
    if missing:
        out.append(
            f"<p class='note alert'><b>Price-only plan: WRDS was unavailable.</b> Value, quality, analyst and "
            f"short-interest components ({missing['weight']:.0%} of the Quant Score) are missing, so the "
            f"verdicts can differ from a run with WRDS. Reason: {escape(str(missing['reason']))}</p>")

    kpis = [
        (str(len(to_send)), "orders to send"),
        (f"{n_corr} / {len(plan)}", "research signals corrected"),
        (f"{tp['gross']:.2f} / {tp['net']:+.2f}", "gross / net exposure"),
        (f"{tp['beta']:+.2f} &rarr; {beta_after:+.2f}", "book beta, before &rarr; after hedge"),
        (_pct(tp.get("risk_at_stops"), 1), "capital lost if every stop hits"),
    ]
    out.append("<div class='kpis'>" + "".join(
        f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>" for v, l in kpis) + "</div>")

    # 1. Ordenes
    out.append("<h2>1. Orders</h2>")
    if not tp.get("has_positions"):
        out.append("<p class='muted'>No current positions were given: every order opens from zero.</p>")
    rows = []
    for t, o in orders.iterrows():
        qty = f"{int(o['qty']):,}" if o["qty"] else "&ndash;"
        is_hedge = o.get("verdict") == "HEDGE"
        rows.append(
            f"<tr><td class='l'><b>{escape(t)}</b></td><td class='l'>{_badge(o['signal'])}</td><td>{qty}</td>"
            f"<td>{int(o['current']):,} &rarr; {int(o['target']):,}</td><td>{_price(o.get('price'))}</td>"
            f"<td>{'' if is_hedge else _price(o.get('stop'))}</td><td>{'' if is_hedge else _price(o.get('tp1'))}</td>"
            f"<td>{'' if is_hedge else _price(o.get('tp2'))}</td>"
            f"<td>{'beta hedge' if is_hedge else _pct(o.get('risk_capital_pct'), 2)}</td></tr>")
    out.append("<div class='wrap'><table><tr><th class='l'>Ticker</th><th class='l'>Signal</th><th>Qty</th>"
               "<th>Now &rarr; target</th><th>Last</th><th>Stop</th><th>TP1</th><th>TP2</th>"
               "<th>Risk at stop</th></tr>" + "".join(rows) + "</table></div>")
    out.append("<p class='muted'>Orders at market next session. Stop and targets apply to the whole resulting "
               "position. Take 50% off at TP1 and move the stop to entry; the rest exits at TP2 or at that stop.</p>")

    # 2. Correcciones
    out.append("<h2>2. What the model changed, and why</h2>")
    rows = []
    for t, row in plan.iterrows():
        rows.append(
            f"<tr><td class='l'><b>{escape(t)}</b></td><td class='l'>{escape(row['research'])}</td>"
            f"<td class='l'>{escape(row['verdict'])}</td><td>{row['aligned_score']:+.2f}</td>"
            f"<td class='wrapcell'><span class='up'>{escape(row.get('drivers_for', '') or '')}</span></td>"
            f"<td class='wrapcell'><span class='down'>{escape(row.get('drivers_against', '') or '')}</span></td>"
            f"<td class='wrapcell'>{escape(_VERDICT_TEXT.get(row['verdict'], ''))}</td></tr>")
    out.append("<div class='wrap'><table><tr><th class='l'>Ticker</th><th class='l'>Research</th>"
               "<th class='l'>Verdict</th><th>Aligned score</th><th class='l'>Pushes for</th>"
               "<th class='l'>Pushes against</th><th class='l'>Meaning</th></tr>" + "".join(rows) + "</table></div>")
    out.append(f"<p class='muted'>Aligned score = Quant Score &times; (+1 if research says LONG, &minus;1 if SHORT). "
               f"&ge; {p.weak:+.2f} confirm; between {p.conflict:+.2f} and {p.weak:+.2f} reduce to "
               f"{p.reduce_size:.0%}; up to {p.flip:+.2f} no trade; below {p.flip:+.2f} flip. Drivers are "
               "the components with the largest weighted contribution for and against the research direction.</p>")

    # 3. Niveles y probabilidades
    be1, be2 = 1 / (1 + p.tp_r[0]), 1 / (1 + p.tp_r[1])
    out.append("<h2>3. Levels, size and historical odds</h2>")
    rows = []
    for t, row in plan.iterrows():
        rows.append(
            f"<tr><td class='l'><b>{escape(t)}</b></td><td class='l'>{_badge(row['action'])}</td>"
            f"<td>{_price(row['entry'])}</td><td>{_price(row['stop'])}</td><td>{_pct(row['stop_pct'])}</td>"
            f"<td>{_price(row['tp1'])}</td><td>{_price(row['tp2'])}</td><td>{int(row['shares']):,}</td>"
            f"<td>{_money(row['notional'])}</td><td>{row['weight'] * 100:+.1f}%</td>"
            f"<td>{_odds(row['p_tp1_before_stop'], be1)}</td><td>{_odds(row['p_tp2_before_stop'], be2)}</td></tr>")
    out.append("<div class='wrap'><table><tr><th class='l'>Ticker</th><th class='l'>Plan</th><th>Entry</th>"
               "<th>Stop</th><th>Stop %</th><th>TP1</th><th>TP2</th><th>Shares</th><th>Notional</th>"
               "<th>Weight</th><th>P(TP1)</th><th>P(TP2)</th></tr>" + "".join(rows) + "</table></div>")
    capped = list(plan.index[plan["size_capped_by_risk"]])
    out.append(
        f"<p class='muted'>Stop = {p.stop_atr} &times; ATR(14) from entry. R = entry-to-stop distance; TP1 = "
        f"{p.tp_r[0]}R, TP2 = {p.tp_r[1]}R. Size = portfolio weight with the corrected directions, capped so a "
        f"stop costs at most {p.max_risk_per_trade:.0%} of capital"
        + (f" (cap binds on {escape(', '.join(capped))})" if capped else "") + ". "
        f"P(TP) = share of past entries in this stock over the last {p.history_d // 252} years that reached the "
        f"target before the stop within {p.horizon_d} sessions. Green if above break-even for that target "
        f"({be1:.0%} for TP1, {be2:.0%} for TP2), red if below. A historical base rate, not a forecast.</p>")

    # 4. Cobertura
    out.append("<h2>4. Market exposure</h2>")
    if hedge:
        out.append(
            f"<p class='note warn'><b>Beta hedge: {escape(hedge['action'])} {hedge['shares']:,} "
            f"{escape(hedge['ticker'])}</b> at {_price(hedge['price'])} (notional {_money(hedge['notional'])}). "
            f"Book beta {hedge['beta_before']:+.2f} &rarr; {hedge['beta_after']:+.2f}. The corrections and risk caps "
            "changed the book the optimizer built; without the hedge it would carry market exposure nobody chose.</p>")
    elif not p.hedge_beta:
        # Sin cobertura por configuracion, la beta se queda: decirlo, no
        # presentarlo como "dentro del umbral".
        out.append(f"<p class='note warn'>Beta hedge disabled (<code>trade_plan.hedge_beta: false</code>): the "
                   f"book carries beta {tp['beta']:+.2f} as is.</p>")
    elif abs(tp["beta"]) < p.hedge_threshold:
        out.append(f"<p class='note'>Book beta {tp['beta']:+.2f}: inside &plusmn;{p.hedge_threshold:.2f}, no hedge "
                   "needed.</p>")
    else:
        out.append(f"<p class='note warn'>Book beta {tp['beta']:+.2f} is not hedged: no usable price for "
                   f"{escape(str(r.settings.benchmark))}.</p>")
    outside = list(orders.index[orders["signal"] == "NOT ANALYZED"])
    if outside:
        out.append(f"<p class='note warn'>Positions outside the ticker list ({escape(', '.join(outside))}) are "
                   "left as they are and are not part of the beta hedge: their market exposure is on top of "
                   "the beta above.</p>")
    if tp.get("note"):
        out.append(f"<p class='note warn'>{escape(tp['note'])}</p>")

    # 5. Graficos
    live = orders[(orders["target"] != 0) & orders["stop"].notna()] if "stop" in orders else orders.iloc[0:0]
    if len(live):
        out.append("<h2>5. Price and levels, last six months</h2><div class='charts'>")
        for t, o in live.iterrows():
            out.append(f"<figure><img alt='{escape(t)} levels' src='data:image/png;base64,{_chart(r, t, o)}'>"
                       "</figure>")
        out.append("</div>")

    # 6. Limites
    out.append("<h2>6. Read before executing</h2><ul>"
               "<li>Entry is the last close. A gap at the open moves the entry and R; recompute size if it is large.</li>"
               "<li>A stop is not guaranteed: an earnings gap jumps it. Check earnings dates before opening.</li>"
               f"<li>{_pct(tp.get('risk_at_stops'), 1)} is the loss if every stop hits. In a market crash "
               "correlations rise and several stops hit together.</li>"
               "<li>Short availability and borrow cost are not checked here.</li>"
               "<li>With few names the Quant Score is noisy: it ranks, it does not fine-tune.</li>"
               "</ul>")
    out.append(f"<footer>Generated {datetime.now():%Y-%m-%d %H:%M} &middot; config fingerprint {escape(s.fingerprint)} "
               "&middot; Mechanical output of the rules in config/quant_engine.yaml. Not investment advice.</footer>")

    title = f"Trade Plan {levels_date:%Y-%m-%d}"
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{escape(title)}</title><style>{CSS}</style></head><body><main>"
            + "".join(out) + "</main></body></html>")


def write(r: AnalysisResult, path: Path) -> Path:
    path.write_text(build(r), encoding="utf-8")
    return path
