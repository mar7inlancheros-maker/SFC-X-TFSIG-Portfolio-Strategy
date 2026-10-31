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
.sub3 { font-size:15px; margin:22px 0 8px; }
.card { border:1px solid var(--line); border-radius:8px; padding:12px 16px; margin:12px 0; }
.card h3 { margin:0 0 6px; font-size:16px; display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.card .hl { margin:4px 0 8px; }
.card .why { margin:4px 0; font-size:13px; }
.card ol { margin:6px 0 0; padding-left:22px; } .card ol li { margin:4px 0; font-size:13px; }
.conv { font-size:12px; color:var(--muted); }
footer { margin-top:36px; color:var(--muted); font-size:12px; }
@media print {
  main { padding:0; } h2, h3 { break-after:avoid; } figure, tr, .kpi, .note { break-inside:avoid; }
  .wrap { overflow:visible; break-inside:avoid; }
  table { font-size:11px; } th, td { padding:5px 6px; white-space:normal; }
  td.wrapcell { min-width:0; }
}
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
    """Precio de los ultimos 6 meses con entrada, stop y objetivos (moneda local, como los niveles)."""
    close = r.clean.local_prices(ticker)[0].dropna().iloc[-126:]
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


def _pnl_chart(pnl: pd.Series) -> str:
    """Histograma del P&L del plan abierto en cada fecha historica."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.hist(pnl.values, bins=40, color=LONG_COLOR, alpha=0.85)
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(float(pnl.mean()), color=TP_COLOR, lw=1.5, ls="--", label=f"mean {pnl.mean():+,.0f}")
    ax.axvline(float(pnl.quantile(0.05)), color=SHORT_COLOR, lw=1, ls=":", label=f"5% {pnl.quantile(0.05):+,.0f}")
    ax.set_title("Plan P&L if opened on each past day (USD)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _equity_chart(r: AnalysisResult) -> str | None:
    """Retorno acumulado de la canasta como cartera frente al benchmark."""
    bt = (r.primary or {}).get("backtest")
    if bt is None or len(bt.nav) < 2:
        return None
    nav = bt.nav / bt.nav.iloc[0] - 1
    bench = (1 + r.clean.returns[r.settings.benchmark].reindex(bt.nav.index).fillna(0.0)).cumprod() - 1
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(nav.index, nav.values * 100, color=LONG_COLOR, lw=1.4, label=f"Basket ({r.primary.get('method')})")
    ax.plot(bench.index, bench.values * 100, color="#7f7f7f", lw=1.2, label=str(r.settings.benchmark))
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_title("Cumulative return, % (Mode A, hypothetical path)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_SIGNAL_BADGE = {"LONG": "BUY", "SHORT": "SELL SHORT", "NO TRADE": "NO TRADE", "NOT ANALYZED": "NOT ANALYZED"}


def _sentences(lines: list[str]) -> str:
    """Lineas de la ficha en un parrafo: cada una cierra con punto, o se pegan
    ("...the value per share DCF check: ...")."""
    return " ".join(x.rstrip() if x.rstrip().endswith((".", "!", "?")) else x.rstrip() + "." for x in lines if x)


def _signal_cards(r: AnalysisResult, h2) -> list[str]:
    """Una ficha por nombre: senal, por que, fundamentales y pasos de ejecucion."""
    from ..playbook import (  # noqa: PLC0415
        build_cards,
        card_size_note,
        standalone_total,
    )

    cards = build_cards(r)
    if not cards:
        return []
    tp = r.trade_plan
    p = tp["params"]
    bench = r.settings.benchmark
    bench_close = r.clean.close[bench].dropna()
    total = standalone_total(cards, bench, float(bench_close.iloc[-1]) if len(bench_close) else float("nan"),
                             (r.trade_plan or {}).get("hedge"))
    out = [h2("Signals: what to do, why, and how")]
    errors = tp.get("integrity_errors")
    if errors:
        out.append("<p class='note alert'><b>Plan integrity check failed: do not send these orders until "
                   "fixed.</b><br>" + "<br>".join(escape(e) for e in errors) + "</p>")
    elif errors is not None:
        out.append("<p class='muted'>Integrity check passed: prices, sides, shares = notional / price, risk caps, "
                   "orders and hedge all reconcile.</p>")
    for t, c in cards.items():
        conv = f"<span class='conv'>conviction {escape(c['conviction'])}</span>" if c.get("conviction") else ""
        side = (f"<span class='conv'>{'against' if c.get('against_research') else 'with'} research "
                f"{escape(c['research'])}</span>" if c["signal"] in ("LONG", "SHORT") else "")
        badge = _badge(_SIGNAL_BADGE.get(c["signal"], "NO TRADE"))
        block = [f"<div class='card'><h3>{escape(t)} {badge}{side}{conv}</h3><p class='hl'>{escape(c['headline'])}</p>"]
        if c["why_for"]:
            block.append(f"<p class='why'><span class='up'>Backs research {escape(c['research'])}:</span> "
                         + escape("; ".join(c["why_for"])) + "</p>")
        if c["why_against"]:
            block.append(f"<p class='why'><span class='down'>Against research {escape(c['research'])}:</span> "
                         + escape("; ".join(c["why_against"])) + "</p>")
        if c["fundamentals"]:
            block.append("<p class='why'><b>Fundamentals:</b> " + escape(" | ".join(c["fundamentals"])) + "</p>")
        if c.get("valuation"):
            block.append("<p class='why'><b>Valuation:</b> " + escape(_sentences(c["valuation"]))
                         + (f" <b>{escape(c['valuation_vs_signal'][0].upper() + c['valuation_vs_signal'][1:])}.</b>"
                            if c.get("valuation_vs_signal") else "") + "</p>")
        if c.get("technical"):
            block.append("<p class='why'><b>Technical:</b> " + escape(_sentences(c["technical"])) + "</p>")
        if c.get("timing"):
            block.append("<p class='why'><b>Timing:</b> " + escape(_sentences(c["timing"])) + "</p>")
        if c["data_flags"]:
            block.append("<p class='why down'><b>Data to verify:</b> " + escape("; ".join(c["data_flags"])) + "</p>")
        if c.get("steps"):
            block.append(f"<p class='why'><b>Size:</b> {c['shares']:,} shares, ~{c['notional']:,.0f} USD, stop costs "
                         f"{c['risk_pct']:.2%} of capital ({escape(card_size_note(c, p.max_risk_per_trade))}).</p>")
            block.append("<p class='why'><b>How to execute:</b></p><ol>"
                         + "".join(f"<li>{escape(s)}</li>" for s in c["steps"]) + "</ol>")
        if c.get("time_window"):
            block.append(f"<p class='why'><b>Time window ({'short' if c['signal'] == 'SHORT' else 'long'}):</b></p><ul>"
                         + "".join(f"<li>{escape(s)}</li>" for s in c["time_window"]) + "</ul>")
        block.append("</div>")
        out.append("".join(block))
    if total:
        out.append(f"<p class='note warn'><b>All standalone positions together</b> "
                   f"({escape(', '.join(total['names']))}): long {total['long']:,.0f} USD, short "
                   f"{total['short']:,.0f} USD, net {total['net']:+,.0f} USD, beta-weighted{' (1-year betas)' if total['hedge_in_plan'] else ''} {total['beta_usd']:+,.0f} "
                   f"USD of {escape(bench)}, {total['risk_pct']:.2%} of capital if every stop hits. "
                   + ("The plan's orders already hedge them together: " if total["hedge_in_plan"]
                      else "One hedge for all of them instead of one per name: ")
                   + f"{escape(total['hedge_action'])} {total['hedge_shares']:,} "
                   f"{escape(bench)} at ~{total['bench_price']:,.2f}.</p>")
    return out


def _valuation_section(r: AnalysisResult, h2) -> list[str]:
    """Tabla del DCF de cada nombre y de donde sale cada supuesto."""
    from ..valuation.engine import summary  # noqa: PLC0415

    vals = {t: summary(v) for t, v in (getattr(r, "valuation", None) or {}).items()}
    if not vals:
        return []
    out = [h2("Valuation: automated DCF")]
    rows, missing = [], []
    for t, v in vals.items():
        if not v["available"]:
            missing.append(f"{t}: {v['reason']}")
            continue
        c = v["value_cases"]
        cls = "up" if v["stance"] == "UNDERVALUED" else "down" if v["stance"] == "OVERVALUED" else ""
        implied = f"{v['implied_cagr'] * 100:+.1f}%" if v.get("implied_cagr") is not None else "n/a"
        rows.append(
            f"<tr><td class='l'><b>{escape(t)}</b> <span class='muted'>{escape(v['trading_currency'])}</span></td>"
            f"<td>{_price(v['price'])}</td><td>{_price(c['bear'])}</td><td>{_price(c['base'])}</td>"
            f"<td>{_price(c['bull'])}</td><td><b>{_price(v['value_expected'])}</b></td>"
            f"<td class='{cls}'>{v['upside_base'] * 100:+.0f}%</td><td class='l {cls}'>{escape(v['stance'])}</td>"
            f"<td>{_pct(v['wacc'])}</td><td>{_pct(v['tv_share'], 0) or 'n/a'}</td>"
            f"<td>{_pct(v.get('mc_prob_above_price'), 0)}</td><td>{implied}</td></tr>")
    if rows:
        out.append("<div class='wrap'><table><tr><th class='l'>Ticker</th><th>Price</th><th>Bear</th><th>Base</th>"
                   "<th>Bull</th><th>Weighted</th><th>Base upside</th><th class='l'>Stance</th><th>WACC</th>"
                   "<th>TV % EV</th><th>P(value &gt; price)</th><th>Implied 10y CAGR</th></tr>" + "".join(rows)
                   + "</table></div>")
        out.append("<p class='muted'>Free cash flow to the firm over 10 years: growth fades in a straight line to "
                   "the terminal rate, margins converge to their target in 5, the tax rate rises to the marginal "
                   "one, and growth needs capital (at least 1 of net reinvestment per 5 of new revenue). "
                   "Terminal value reinvests g/RONIC of NOPAT, with RONIC = WACC. WACC from CAPM with a "
                   "Blume-adjusted beta and a synthetic-rating cost of debt, at market weights; mid-year "
                   "discounting. Scenarios weighted 25/50/25. Implied CAGR: the revenue growth the price needs "
                   "with every other assumption unchanged. Stance from the base case; NOT DECISION-GRADE when "
                   "a critical check fails (thin margin, negative equity, WACC too close to g, unreliable share "
                   "count or debt, revenue jump). The DCF informs; it does not change the verdicts.</p>")
        for t, v in vals.items():
            if v["available"]:
                items = "".join(f"<li>{escape(n)}</li>" for n in v["notes"])
                checks = "".join(f"<li class='down'>{escape(f)}</li>" for f in v["flags"])
                out.append(f"<details open><summary><b>{escape(t)}</b>: assumptions and checks</summary><ul>{items}"
                           f"{checks}</ul></details>")
    if missing:
        out.append("<p class='note warn'><b>No DCF:</b> " + escape("; ".join(missing)) + "</p>")
    return out


def _hold(row) -> str:
    """Percentiles 25-75 de los dias que dura la operacion con esta regla en este valor."""
    p25, p75 = row.get("hold_p25"), row.get("hold_p75")
    return "" if p25 is None or pd.isna(p25) else f"{p25:.0f}-{p75:.0f} d"


def _tstat(v) -> str:
    return "" if v is None or pd.isna(v) else f"{v:+.1f}"


def _usd(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return "0" if abs(v) < 0.5 else f"{v:+,.0f}"


def _ticker(t: str, currency) -> str:
    """Ticker con su moneda si no cotiza en USD: los niveles van en esa moneda."""
    ccy = "" if currency in (None, "USD") or pd.isna(currency) else f" <span class='muted'>{escape(str(currency))}</span>"
    return f"<b>{escape(t)}</b>{ccy}"


def _expectations(r: AnalysisResult, tp: dict, plan: pd.DataFrame, p, h2) -> list[str]:
    out = [h2("What to expect: backtests")]
    cols = ["bt_trades", "bt_win_rate", "bt_exp_r", "bt_avg_win_r", "bt_avg_loss_r", "bt_profit_factor",
            "bt_avg_days", "exp_return_pct", "exp_pnl"]
    if all(c in plan.columns for c in cols):
        rows = []
        for t, row in plan.iterrows():
            pf = row["bt_profit_factor"]
            rows.append(
                f"<tr><td class='l'><b>{escape(t)}</b></td><td class='l'>{_badge(row['action'])}</td>"
                f"<td>{int(row['bt_trades']):,}</td><td>{_pct(row['bt_win_rate'], 0)}</td>"
                f"<td>{row['bt_exp_r']:+.2f}</td><td>{_tstat(row.get('bt_t_stat'))}</td>"
                f"<td>{row['bt_avg_win_r']:+.2f}</td><td>{row['bt_avg_loss_r']:+.2f}</td>"
                f"<td>{'&infin;' if pd.notna(pf) and pf == float('inf') else f'{pf:.2f}'}</td>"
                f"<td>{row['bt_avg_days']:.0f}</td><td>{row['exp_return_pct'] * 100:+.1f}%</td>"
                f"<td>{_usd(row['exp_pnl'])}</td></tr>")
        out.append("<h3 class='sub3'>Each trade: the same rule on this stock's past</h3>")
        out.append("<div class='wrap'><table><tr><th class='l'>Ticker</th><th class='l'>Plan</th><th>Trades</th>"
                   "<th>Win rate</th><th>E[R]</th><th>t</th><th>Avg win</th><th>Avg loss</th><th>Profit factor</th>"
                   "<th>Avg days</th><th>E[return]</th><th>E[P&amp;L] today</th></tr>" + "".join(rows)
                   + "</table></div>")
        out.append(f"<p class='muted'>The plan's levels (stop {p.stop_atr} ATR, TP1 {p.tp_r[0]}R, TP2 {p.tp_r[1]}R, "
                   f"half out at TP1 and stop to entry) opened on every day of the last {p.history_d // 252} years, "
                   f"closed after {p.horizon_d} sessions at most, net of transaction costs and, for shorts, borrow. "
                   "E[R] is the average result in multiples of the risk at the stop; t is E[R] over its "
                   "Newey-West standard error (daily entries overlap), so |t| below 2 is not distinguishable "
                   "from zero; E[return] is on the position; "
                   "E[P&amp;L] applies E[R] to today's size. Conservative fills: a day that touches stop and target "
                   "counts as stop, and a gap through the stop fills at the day's worst price.</p>")
        traded = tp["orders"].reindex(plan.index)["target"].fillna(0) != 0
        negative = plan[traded & (plan["bt_exp_r"] < 0)]
        if len(negative):
            out.append("<p class='note warn'><b>Negative historical expectancy on traded names:</b> "
                       + escape(", ".join(f"{t} ({v:+.2f}R)" for t, v in negative["bt_exp_r"].items()))
                       + ". This rule lost money on these stocks in the past: review before entering.</p>")
    pb = tp.get("plan_backtest")
    if pb:
        out.append("<h3 class='sub3'>The whole plan, opened together</h3>")
        kpis = [(_usd(pb["mean"]), f"mean P&amp;L ({pb['mean_pct'] * 100:+.2f}% of capital)"),
                (_usd(pb["median"]), "median P&amp;L"),
                (f"{_usd(pb['p05'])} / {_usd(pb['p95'])}", "5th / 95th percentile"),
                (_pct(pb["prob_loss"], 0), "probability of a loss"),
                (_usd(pb["worst"]), "worst past outcome")]
        out.append("<div class='kpis'>" + "".join(
            f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>" for v, l in kpis) + "</div>")
        out.append(f"<div class='charts' style='margin-top:14px'><figure><img alt='Plan P&amp;L distribution' "
                   f"src='data:image/png;base64,{_pnl_chart(pb['pnl'])}'></figure></div>")
        out.append(f"<p class='muted'>Today's positions, each managed with its own levels, opened together on each of "
                   f"{pb['n_dates']} past days ({pb['first']:%Y-%m-%d} to {pb['last']:%Y-%m-%d}). Using the same "
                   "dates for every name keeps the correlation between them: on a bad day for the market, all the "
                   "longs suffer together." + (" The beta hedge is not included." if tp.get("hedge") else "")
                   + "</p>")
    pf = tp.get("portfolio") or {}
    if pf.get("cagr") is not None:
        out.append("<h3 class='sub3'>The basket as a portfolio</h3>")
        kpis = [(_pct(pf.get("cagr")), f"CAGR &middot; {escape(str(r.settings.benchmark))} {_pct(pf.get('bench_cagr'))}"),
                (f"{pf.get('sharpe'):.2f}" if pf.get("sharpe") is not None else "n/a", "Sharpe"),
                (_pct(pf.get("max_drawdown")), "max drawdown"),
                (_pct(pf.get("mc_median_return")), "Monte Carlo 1y median"),
                (f"{_pct(pf.get('mc_p05_return'))} / {_pct(pf.get('mc_p95_return'))}", "Monte Carlo 5% / 95%"),
                (_pct(pf.get("mc_prob_loss"), 0), "Monte Carlo P(loss)")]
        out.append("<div class='kpis'>" + "".join(
            f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>" for v, l in kpis) + "</div>")
        equity = _equity_chart(r)
        if equity:
            out.append(f"<div class='charts' style='margin-top:14px'><figure><img alt='Basket cumulative return' "
                       f"src='data:image/png;base64,{equity}'></figure></div>")
        out.append(f"<p class='muted'>{escape(str(pf.get('method')))} portfolio with research's directions over the "
                   "evaluation window (Mode A): today's basket applied to the past, so the selection is in-sample. "
                   "The Monte Carlo resamples its daily returns with the historical mean removed (neutral drift): "
                   "it measures risk, not expected return.</p>")
    out.append("<p class='note warn'>Every number here is a historical base rate: it carries each stock's past "
               "trend and knows nothing about what comes next. Use it to compare trades and size risk, not as a "
               "promise.</p>")
    return out


def build(r: AnalysisResult) -> str:
    counter = iter(range(1, 20))

    def h2(title: str) -> str:
        return f"<h2>{next(counter)}. {escape(title)}</h2>"

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
    kind = "Long-only" if tp.get("long_only") else "Long/short"
    out.append(f"<h1>Trade Plan</h1><p class='sub'>{kind} research basket &middot; levels from the "
               f"{levels_date:%Y-%m-%d} close</p>")
    out.append("<div class='meta'>"
               f"<span>Research LONG: <b>{escape(' '.join(r.longs))}</b></span>"
               f"<span>Research SHORT: <b>{escape(' '.join(r.shorts))}</b></span>"
               f"<span>Capital: {_money(s.initial_capital)}</span>"
               f"<span>Construction: {escape(s.construction)}</span>"
               f"<span>Benchmark: {escape(s.benchmark)}</span>"
               f"<span>Data: Yahoo prices{' + WRDS (Compustat, IBES, short interest, Fama-French)' if wrds_ok else ''}</span>"
               "</div>")

    if r.excluded:
        why = {t: (f"only {int(r.quality.loc[t, 'sessions'])} sessions of price history, the analysis needs "
                   "a year" if r.quality.loc[t, "sessions"] else "no price data")
               for t in r.excluded if t in r.quality.index}
        out.append("<p class='note alert'><b>Not analyzed (data quality):</b> "
                   + escape("; ".join(f"{t}: {why.get(t, 'insufficient data')}" for t in r.excluded))
                   + ". The rest of the plan is built without it.</p>")
    if not tp.get("has_book", True):
        out.append("<p class='note warn'><b>No book to size.</b> After the corrections "
                   + ("no LONG is left to trade" if tp.get("long_only") else "one leg has nothing left to trade")
                   + f". Each name the data supports is sized on its own (its stop costs "
                   f"{tp['params'].max_risk_per_trade:.0%} of capital, less when reduced); the orders are those "
                   "sizes.</p>")
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

    # Senales: que hacer, por que y como.
    out.extend(_signal_cards(r, h2))

    # 1. Ordenes
    out.append(h2("Orders"))
    if not tp.get("has_positions"):
        out.append("<p class='muted'>No current positions were given: every order opens from zero.</p>")
    rows = []
    for t, o in orders.iterrows():
        qty = f"{int(o['qty']):,}" if o["qty"] else "&ndash;"
        is_hedge = o.get("verdict") == "HEDGE"
        rows.append(
            f"<tr><td class='l'>{_ticker(t, o.get('currency'))}</td><td class='l'>{_badge(o['signal'])}</td><td>{qty}</td>"
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
    out.append(h2("What the model changed, and why"))
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
    sv = getattr(r, "score_validation", None) or {}
    if sv.get("available"):
        from ..signals.score_validation import evidence  # noqa: PLC0415

        ev = evidence(sv, 2.0)
        text = {"predictive": "it ranked next-month returns well on this universe",
                "none": "no evidence that it predicts on this universe: it ranks, it does not forecast",
                "inverse": "it ranked returns backwards on this universe: treat its verdicts with caution"}[ev]
        out.append(f"<p class='note{' warn' if ev != 'predictive' else ''}'><b>Does the Quant Score predict?</b> "
                   f"Recomputed at {sv['months']} past month-ends with the data of each day, against the next "
                   f"month: IC {sv['ic_mean']:+.3f} (t {sv['ic_t']:+.1f}), {escape(text)}."
                   + ((f" {escape(', '.join(tp['unproven_flips']))}: the score says FLIP, but going against "
                       "research needs that evidence, so it stays NO TRADE." if p.unproven_flip == "no_trade" else
                       f" {escape(', '.join(tp['unproven_flips']))}: the score says FLIP; without that evidence "
                       f"the other side is taken at {p.reduce_size:.0%} size with LOW conviction.")
                      if tp.get("unproven_flips") else "")
                   + "</p>")
    out.append(f"<p class='muted'>Aligned score = Quant Score &times; (+1 if research says LONG, &minus;1 if SHORT). "
               f"&ge; {p.weak:+.2f} confirm; between {p.conflict:+.2f} and {p.weak:+.2f} reduce to "
               f"{p.reduce_size:.0%}; up to {p.flip:+.2f} no trade; below {p.flip:+.2f} flip. Drivers are "
               "the components with the largest weighted contribution for and against the research direction.</p>")

    # 3. Niveles y probabilidades
    be1, be2 = 1 / (1 + p.tp_r[0]), 1 / (1 + p.tp_r[1])
    out.append(h2("Levels, size and historical odds"))
    rows = []
    for t, row in plan.iterrows():
        rows.append(
            f"<tr><td class='l'>{_ticker(t, row.get('currency'))}</td><td class='l'>{_badge(row['action'])}</td>"
            f"<td>{_price(row['entry'])}</td><td>{_price(row['stop'])}</td><td>{_pct(row['stop_pct'])}</td>"
            f"<td>{_price(row['tp1'])}</td><td>{_price(row['tp2'])}</td><td>{int(row['shares']):,}</td>"
            f"<td>{_money(row['notional'])}</td><td>{row['weight'] * 100:+.1f}%</td>"
            f"<td>{_odds(row['p_tp1_before_stop'], be1)}</td><td>{_odds(row['p_tp2_before_stop'], be2)}</td>"
            f"<td>{_hold(row)}</td></tr>")
    out.append("<div class='wrap'><table><tr><th class='l'>Ticker</th><th class='l'>Plan</th><th>Entry</th>"
               "<th>Stop</th><th>Stop %</th><th>TP1</th><th>TP2</th><th>Shares</th><th>Notional</th>"
               "<th>Weight</th><th>P(TP1)</th><th>P(TP2)</th><th>Typical hold</th></tr>" + "".join(rows)
               + "</table></div>")
    capped = list(plan.index[plan["size_capped_by_risk"]])
    out.append(
        f"<p class='muted'>Stop = {p.stop_atr} &times; ATR(14) from entry. R = entry-to-stop distance; TP1 = "
        f"{p.tp_r[0]}R, TP2 = {p.tp_r[1]}R. Size = portfolio weight with the corrected directions, capped so a "
        f"stop costs at most {p.max_risk_per_trade:.0%} of capital"
        + (f" (cap binds on {escape(', '.join(capped))})" if capped else "") + ". "
        f"P(TP) = share of past entries in this stock over the last {p.history_d // 252} years that reached the "
        f"target before the stop within {p.horizon_d} sessions. Green if above break-even for that target "
        f"({be1:.0%} for TP1, {be2:.0%} for TP2), red if below. A historical base rate, not a forecast."
        + (" Prices are in each stock's trading currency (shown next to the ticker); notional, weight and "
           "risk are in USD." if "currency" in plan and plan["currency"].ne("USD").any() else "") + "</p>")

    # 4. Expectativas
    out.extend(_expectations(r, tp, plan, p, h2))

    out.extend(_valuation_section(r, h2))

    # 5. Cobertura
    out.append(h2("Market exposure"))
    if hedge:
        out.append(
            f"<p class='note warn'><b>Beta hedge: {escape(hedge['action'])} {hedge['shares']:,} "
            f"{escape(hedge['ticker'])}</b> at {_price(hedge['price'])} (notional {_money(hedge['notional'])}). "
            f"Book beta {hedge['beta_before']:+.2f} &rarr; {hedge['beta_after']:+.2f}. The corrections and risk caps "
            "changed the book the optimizer built; without the hedge it would carry market exposure nobody chose.</p>")
    elif tp.get("long_only"):
        out.append(f"<p class='note'>Long-only basket: market beta ({tp['beta']:+.2f}) is the intended "
                   "exposure, so there is no beta hedge.</p>" if tp.get("gross") else
                   "<p class='note'>Long-only basket: nothing to trade, so nothing to hedge.</p>")
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
    if tp.get("note") and tp.get("has_book", True):
        out.append(f"<p class='note warn'>{escape(tp['note'])}</p>")

    # 5. Graficos
    live = orders[(orders["target"] != 0) & orders["stop"].notna()] if "stop" in orders else orders.iloc[0:0]
    if len(live):
        out.append(h2("Price and levels, last six months") + "<div class='charts'>")
        for t, o in live.iterrows():
            out.append(f"<figure><img alt='{escape(t)} levels' src='data:image/png;base64,{_chart(r, t, o)}'>"
                       "</figure>")
        out.append("</div>")

    # 6. Limites
    out.append(h2("Read before executing") + "<ul>"
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
