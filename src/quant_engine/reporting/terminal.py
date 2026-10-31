"""Reporte en terminal con `rich`. Solo presenta: todo se calcula en `analysis.py`.

Regla de redaccion del motor: DESCRIBIR, nunca recomendar. "NVDA tiene el
mayor score de momentum del universo", nunca "compra NVDA". Las frases de
interpretacion de abajo estan escritas asi a proposito.

La excepcion son el plan de operacion (14) y la senal final: ahi si salen
BUY / SELL, pero como aplicacion mecanica de reglas declaradas en el YAML
(umbrales, ATR, tope de riesgo), no como opinion. El texto lo dice.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ..analysis import AnalysisResult

Fmt = Callable[[object], str]
LINE = "=" * 60

# `box.SIMPLE_HEAD` en ASCII. El original subraya la cabecera con U+2500, y todo
# lo que se imprime va en ASCII: la consola de Windows usa cp1252 al redirigir.
ASCII_SIMPLE_HEAD = box.Box(
    "    \n"
    "    \n"
    " -- \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n",
    ascii=True,
)


def pct(v: object, d: int = 1) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v) * 100:.{d}f}%"


def num(v: object, d: int = 2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):.{d}f}"


def signed(v: object, d: int = 2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):+.{d}f}"


def money(v: object) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):,.0f}"


def table(frame: pd.DataFrame, formats: dict[str, Fmt] | None = None, title: str | None = None,
          index_name: str = "") -> Table:
    formats = formats or {}
    t = Table(title=title, box=ASCII_SIMPLE_HEAD, show_edge=False, title_justify="left")
    t.add_column(index_name or (frame.index.name or ""), style="bold")
    for col in frame.columns:
        t.add_column(str(col), justify="right")
    for idx, row in frame.iterrows():
        cells = []
        for col in frame.columns:
            v = row[col]
            f = formats.get(col)
            if f is not None:
                cells.append(f(v))
            elif isinstance(v, float):
                cells.append(num(v))
            else:
                cells.append("" if v is None else escape(str(v)))
        t.add_row(str(idx), *cells)
    return t


def section(console: Console, number: int | str, title: str) -> None:
    console.print()
    console.rule(f"[bold][{number}] {title}", align="left", characters="-")


def header(console: Console, r: AnalysisResult) -> None:
    s = r.settings
    console.print(LINE)
    console.print("        QUANTITATIVE HEDGE FUND RESEARCH ENGINE")
    console.print(LINE)
    console.print()
    console.print("UNIVERSE")
    console.print("-" * 60)
    console.print(f"Long:       {' '.join(r.longs)}")
    console.print(f"Short:      {' '.join(r.shorts) if r.shorts else 'none (long-only basket)'}")
    console.print(f"Benchmark:  {s.benchmark}")
    console.print(f"Period:     {s.lookback} ({r.eval_start.date()} to {r.as_of.date()})")
    console.print(f"Construction: {s.construction} | Rebalance: {s.rebalance} | "
                  f"Capital: {money(s.initial_capital)} | TC: {pct(s.transaction_cost, 2)} | "
                  f"Borrow: {pct(s.borrow_cost, 2)}/yr | rf: {pct(s.risk_free_rate, 2)}")
    console.print(f"Config fingerprint: {s.fingerprint}")
    if r.reference_peers:
        console.print(f"Quant Score against {len(r.reference_peers)} reference peers (scoring.reference_peers): "
                      "they only score, they are never traded.")
    w = r.wrds
    if w is not None and w.available:
        console.print("Data: prices Yahoo | WRDS: Fama-French 5+UMD, GICS, Compustat fundamentals (point-in-time "
                      "by report date), IBES consensus and revisions, short interest")
    else:
        why = escape(w.reason) if w is not None else "not requested"
        console.print(f"Data: prices Yahoo | factors from ETF spreads | sectors from SEC SIC | WRDS unavailable: {why}")
    console.print("-" * 60)


def data_quality(console: Console, r: AnalysisResult) -> None:
    section(console, 1, "DATA QUALITY")
    q = r.quality.copy()
    q["start"] = q["start"].map(lambda d: d.date() if pd.notna(d) else "n/a")
    q["end"] = q["end"].map(lambda d: d.date() if pd.notna(d) else "n/a")
    console.print(table(q[["start", "end", "sessions", "missing_pct", "status", "notes"]],
                        {"missing_pct": lambda v: pct(v, 2), "sessions": lambda v: f"{int(v)}"},
                        index_name="Ticker"))
    console.print(f"Benchmark: {r.settings.benchmark}  |  Observations (evaluation window): "
                  f"{int((r.clean.returns.index >= r.eval_start).sum())}")
    if r.excluded:
        console.print(f"[bold red]EXCLUDED for data quality: {', '.join(r.excluded)}[/]")
    for t, src in r.sector_source.items():
        if r.sectors.get(t) == "Unknown":
            console.print(f"Sector unavailable for {t}: {escape(src)}")


def performance(console: Console, r: AnalysisResult) -> None:
    section(console, 2, "PERFORMANCE")
    tr = r.trailing[["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "3Y_ann", "5Y_ann"]]
    console.print(table(tr, {c: pct for c in tr.columns}, index_name="Ticker"))
    ra = r.risk[["cagr", "sharpe", "sortino", "calmar", "information_ratio", "treynor"]]
    console.print(table(ra, {"cagr": pct, "treynor": pct}, title="Risk-adjusted (evaluation window)",
                        index_name="Ticker"))


def risk(console: Console, r: AnalysisResult) -> None:
    section(console, 3, "RISK")
    cols = ["vol_annual", "downside_vol", "max_dd", "avg_dd", "max_dd_duration_d", "recovery_d", "skewness", "kurtosis"]
    console.print(table(r.risk[cols], {"vol_annual": pct, "downside_vol": pct, "max_dd": pct, "avg_dd": pct,
                                       "max_dd_duration_d": lambda v: num(v, 0), "recovery_d": lambda v: num(v, 0)},
                        index_name="Ticker"))
    v = r.volatility[["vol_21d", "vol_ewma", "vol_percentile", "vol_regime", "vol_of_vol", "atr_pct"]]
    console.print(table(v, {"vol_21d": pct, "vol_ewma": pct, "vol_percentile": lambda x: pct(x, 0),
                            "atr_pct": lambda x: pct(x, 2)}, title="Volatility regime", index_name="Ticker"))
    th = r.settings.get("volatility_regime", {}) or {}
    console.print(f"Regime = percentile of current 21d vol within its own history: "
                  f"LOW < {th.get('low', 0.2):.0%} <= NORMAL < {th.get('high', 0.8):.0%} <= HIGH < "
                  f"{th.get('extreme', 0.95):.0%} <= EXTREME")


def momentum(console: Console, r: AnalysisResult) -> None:
    section(console, 4, "MOMENTUM & MEAN REVERSION")
    m = r.momentum[["mom_1m", "mom_3m", "mom_6m", "mom_12_1", "price_to_sma_50", "price_to_sma_200",
                    "rsi_14", "adx_14", "trend_quality"]]
    console.print(table(m, {c: pct for c in ["mom_1m", "mom_3m", "mom_6m", "mom_12_1", "price_to_sma_50",
                                             "price_to_sma_200"]} | {"rsi_14": lambda v: num(v, 0),
                                                                     "adx_14": lambda v: num(v, 0)},
                        index_name="Ticker"))
    mr = r.mean_reversion[["zscore_20d", "bollinger_pct_b", "vol_adj_momentum", "autocorr_1",
                           "variance_ratio_5", "regime"]]
    console.print(table(mr, {"zscore_20d": signed, "vol_adj_momentum": signed, "autocorr_1": signed},
                        title="Mean reversion / persistence", index_name="Ticker"))
    rc = r.settings.get("regime", {}) or {}
    console.print(f"MEAN REVERSION: |20d z| >= {rc.get('reversion_threshold', 1.5)} and VR(5) < 1.  "
                  f"MOMENTUM: |12-1 return / vol| >= {rc.get('trend_threshold', 0.5)} and price on the "
                  "same side of SMA200.  Otherwise NEUTRAL.")
    top = r.scores["momentum"].idxmax()
    console.print(f"{top} has the highest cross-sectional momentum score in the current universe "
                  f"({signed(r.scores.loc[top, 'momentum'])}).")


def beta_alpha(console: Console, r: AnalysisResult) -> None:
    section(console, 5, "BETA / ALPHA")
    cols = ["beta", "beta_60d", "beta_120d", "beta_252d", "beta_252_range", "r2", "correlation",
            "alpha_annual", "alpha_t", "tracking_error", "information_ratio"]
    reg = r.regression[[c for c in cols if c in r.regression.columns]]
    console.print(table(reg, {"alpha_annual": pct, "tracking_error": pct, "r2": num, "alpha_t": signed},
                        index_name="Ticker"))
    console.print("Regression on excess returns: (R_i - rf) = alpha + beta (R_m - rf) + e, HAC standard errors. "
                  "|alpha t| < 2: alpha not distinguishable from zero. beta_252_range = p95 - p5 of rolling "
                  "252d beta; above ~0.5 beta is unstable.")


def correlation(console: Console, r: AnalysisResult) -> None:
    section(console, 6, "CORRELATION & COVARIANCE")
    c = r.correlation
    console.print(table(c["pearson"], {k: lambda v: num(v, 2) for k in c["pearson"].columns},
                        title="Pearson correlation (daily returns)", index_name=""))
    if r.shorts:
        console.print(f"Average pairwise: {num(c['avg_pairwise'])} | within LONG: {num(c['avg_within_long'])} | "
                      f"within SHORT: {num(c['avg_within_short'])} | LONG vs SHORT: {num(c['avg_long_short'])}")
    else:
        console.print(f"Average pairwise: {num(c['avg_pairwise'])}")
    clusters = [g for g in c["clusters"].values() if len(g) > 1]
    console.print("Clusters (rho >= 0.6, average linkage, Mantegna distance): "
                  + ("; ".join(", ".join(g) for g in clusters) if clusters else "none"))
    rows = {name: {"condition_number": v["condition"], "shrinkage": v["shrinkage"], "psd": v["psd"]}
            for name, v in r.covariances.items()}
    console.print(table(pd.DataFrame(rows).T, {"condition_number": lambda v: num(v, 1),
                                               "shrinkage": lambda v: num(v, 3)},
                        title="Covariance estimators", index_name="Estimator"))


def factor_exposure(console: Console, r: AnalysisResult) -> None:
    section(console, 7, "FACTOR EXPOSURE")
    f = r.factors
    exp = f.exposures.copy()
    exp["R2"] = f.r2
    exp["sector"] = pd.Series(r.sectors)
    console.print(table(exp, {c: signed for c in f.exposures.columns} | {"R2": num}, index_name="Ticker"))
    if f.source.startswith("Fama-French"):
        console.print("Factors: Fama-French 5 (market, size, value, profitability, investment) + momentum, daily, "
                      "from WRDS. Excess returns use the Fama-French daily risk-free rate.")
    else:
        console.print("Factors built from ETF spreads: size IWM-SPY, value IWD-IWF, momentum MTUM-SPY, "
                      "low_volatility USMV-SPY, quality QUAL-SPY. Not academic Fama-French factors.")
    if f.unavailable:
        for k, why in f.unavailable.items():
            console.print(f"[yellow]Factor {k} UNAVAILABLE: {escape(why)}[/]")
    if len(f.vif):
        console.print(f"Factor condition number: {num(f.condition_number, 1)}. VIF: "
                      + ", ".join(f"{k} {num(v, 1)}" for k, v in f.vif.items())
                      + ". VIF > 5: that factor's individual beta is unreliable.")


_EVIDENCE_TEXT = {
    "predictive": "the score ranked next-month returns well on this universe",
    "none": "NO EVIDENCE that the score predicts on this universe: it ranks, it does not forecast",
    "inverse": "the score ranked returns BACKWARDS on this universe: its verdicts are not reliable",
}


def validation_line(r: AnalysisResult) -> str:
    """Una linea con el veredicto de la validacion del Quant Score."""
    from ..signals.score_validation import evidence  # noqa: PLC0415

    sv = getattr(r, "score_validation", None) or {}
    if not sv:
        return "Quant Score validation disabled (validation.enabled: false)."
    if not sv.get("available"):
        return f"Quant Score validation unavailable: {sv.get('reason')}."
    return (f"Quant Score validation: IC {signed(sv['ic_mean'], 3)} (t {signed(sv['ic_t'], 1)}) over "
            f"{sv['months']} months, {sv['avg_names']:.0f} names: {_EVIDENCE_TEXT[evidence(sv, 2.0)]}.")


def score_validation_block(console: Console, r: AnalysisResult) -> None:
    """Predice el Quant Score? El mismo score en cada fin de mes pasado frente al mes siguiente."""
    sv = getattr(r, "score_validation", None) or {}
    console.print()
    console.print(escape(validation_line(r)))
    if not sv.get("available"):
        return
    console.print(f"Same score recomputed at each month-end from {sv['first']:%Y-%m} to {sv['last']:%Y-%m} with the "
                  f"data of that day (basket + reference peers), against the next {sv['horizon_d']} sessions. "
                  f"IC positive in {pct(sv['ic_hit'], 0)} of months; top-minus-bottom tercile "
                  f"{signed(sv['spread_mean'] * 100, 2)}%/month (t {signed(sv['spread_t'], 1)}).")
    ph = sv.get("plan_horizon") or {}
    if ph.get("available"):
        console.print(f"Over the plan's {ph['horizon_d']}-session horizon: IC {signed(ph['ic_mean'], 3)} "
                      f"(t {signed(ph['ic_t'], 1)}, Newey-West: windows overlap), positive in "
                      f"{pct(ph['ic_hit'], 0)} of {ph['months']} months. The FLIP rule uses the "
                      f"{sv['horizon_d']}-session test, which has no overlap.")
    comps = sv.get("components")
    if comps is not None and len(comps):
        show = comps[["ic_mean", "ic_t", "hit_rate", "weight"]].rename(
            columns={"ic_mean": "IC", "ic_t": "t", "hit_rate": "IC > 0", "weight": "weight"})
        console.print(table(show, {"IC": lambda v: signed(v, 3), "t": lambda v: signed(v, 1),
                                   "IC > 0": lambda v: pct(v, 0), "weight": lambda v: pct(v, 1)},
                            title="Each component, same test", index_name="Component"))
    if sv.get("not_validated"):
        console.print("Not validated (no data: WRDS unavailable or no match): " + ", ".join(sv["not_validated"])
                      + ". The validated score is the YAML score renormalized without them. Universe: today's "
                      "large caps, so survivorship makes this test optimistic, not pessimistic.")


def signals(console: Console, r: AnalysisResult) -> None:
    section(console, 8, "QUANTITATIVE SIGNALS")
    w = r.settings.get("signal_weights", {}) or {}
    console.print("Quant Score weights: " + ", ".join(f"{k} {v:.0%}" for k, v in w.items()))
    a = r.agreement.copy()
    comp_cols = [c for c in ["momentum", "value", "quality", "analyst", "risk_adjusted_return", "short_interest",
                             "volatility", "mean_reversion", "beta", "liquidity", "statistical"]
                 if c in r.scores.columns and r.scores[c].notna().any()]
    a = a.join(r.scores[comp_cols])
    console.print(table(a[["research", "quant_score", "rank", "agreement", *comp_cols]],
                        {c: signed for c in ["quant_score", *comp_cols]} | {"rank": lambda v: str(v)},
                        index_name="Ticker"))
    fund = r.fundamentals
    if fund is not None and not fund.empty:
        cols = [c for c in ["earnings_yield", "fcf_yield", "book_to_price", "roe", "gross_profitability",
                            "debt_to_equity", "accruals", "eps_revision_3m", "rec_change_3m", "meanrec",
                            "si_pct_float", "days_to_cover"] if c in fund.columns]
        show = fund[cols].apply(pd.to_numeric, errors="coerce")
        pcts = {c: pct for c in ["earnings_yield", "fcf_yield", "roe", "gross_profitability", "accruals",
                                 "eps_revision_3m", "si_pct_float"]}
        console.print(table(show, pcts | {"book_to_price": num, "debt_to_equity": num, "rec_change_3m": signed,
                                          "meanrec": num, "days_to_cover": lambda v: num(v, 1)},
                            title="Fundamentals, analysts and short interest (WRDS)", index_name="Ticker"))
        if "report_date" in fund:
            dates = pd.to_datetime(fund["report_date"]).dropna()
            if len(dates):
                console.print(f"Fundamentals: trailing 4 quarters published by {dates.max().date()} "
                              "(point-in-time by report date). meanrec: 1 = strong buy ... 5 = sell; "
                              "rec_change < 0 = upgrades. EPS revision: same fiscal year, 3 months, capped at +-100%.")
    th = r.settings.get("agreement", {}) or {}
    counts = a["agreement"].value_counts().to_dict()
    console.print("Agreement counts: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    console.print(f"Aligned score = score x (+1 LONG, -1 SHORT). STRONG >= {th.get('strong', 0.5)}, "
                  f"WEAK >= {th.get('weak', 0.15)}, CONFLICT <= {th.get('conflict', -0.15)}. "
                  "The quant score does not overwrite the research signal.")
    rs = r.rank_stability
    console.print(f"Rank stability under {rs['draws']} random weight draws: median Spearman "
                  f"{num(rs['median_spearman'])}, 5th pct {num(rs['p05_spearman'])}.")
    score_validation_block(console, r)
    if r.reference_peers:
        n = len(a) + len(r.reference_peers)
        console.print(f"Scored against the basket plus {len(r.reference_peers)} reference peers: z-scores rest on "
                      f"{n} names and rank is out of {n}. A large-cap reference: liquidity and size can weigh "
                      "against small caps.")
    else:
        console.print("With N = {} names, cross-sectional z-scores rest on {} points each; small score "
                      "differences are noise.".format(len(a), len(a)))
    fm = r.feature_matrix
    console.print(table(fm, {c: signed for c in fm.columns}, title="Cross-sectional feature matrix (z-scores)",
                        index_name="Ticker"))


def long_vs_short(console: Console, r: AnalysisResult) -> None:
    section(console, 9, "LONG VS SHORT")
    if not r.shorts:
        console.print("Long-only basket: there is no SHORT leg to compare against.")
        return
    lvs = r.long_vs_short[["long_mean", "short_mean", "difference", "welch_p", "mw_p", "perm_p", "ci_low", "ci_high"]]
    console.print(table(lvs, {c: lambda v: num(v, 3) for c in lvs.columns}, index_name="Metric"))
    n_l, n_s = len(r.longs), len(r.shorts)
    console.print(f"[bold]N = {n_l} LONG vs {n_s} SHORT.[/] Cross-sectional tests on this few names have very "
                  "low power: a p-value above 0.05 does not mean 'no difference', and one below 0.05 can be "
                  "chance. Mann-Whitney with 5 vs 5 cannot go below p = 0.008. Permutation p is exact "
                  f"(all {math.comb(n_l + n_s, n_l)} splits).")
    sp = r.spread
    if sp:
        console.print(f"Daily spread LONG basket - SHORT basket (equal weight): {pct(sp['mean_annual'])}/yr, "
                      f"Newey-West t = {num(sp['t_nw'])}, p = {num(sp['p_value'], 4)}, "
                      f"{sp['observations']} days, positive on {pct(sp['hit_rate'])} of days. "
                      "In-sample: the baskets were chosen today.")


def portfolio_construction(console: Console, r: AnalysisResult) -> None:
    section(console, 10, "PORTFOLIO CONSTRUCTION")
    console.print("[bold yellow]MODE A - HYPOTHETICAL PATH.[/] Today's basket applied to the past. "
                  "Weights are estimated walk-forward (only data before each rebalance), but the SELECTION of "
                  "names uses today's information. This is NOT a backtest of the research process.")
    rows = {}
    weights = {}
    for m, b in r.portfolios.items():
        sm = b.get("summary", {})
        ex = b.get("ex_ante", {})
        rows[m] = {"status": b.get("status"), "gross": ex.get("gross"), "net": ex.get("net"),
                   "long": ex.get("long"), "short": ex.get("short"), "beta": ex.get("beta"),
                   "vol_ex_ante": ex.get("vol_annual"), "sharpe": sm.get("sharpe"), "sortino": sm.get("sortino"),
                   "max_dd": sm.get("max_drawdown"), "var_95": ex.get("var_95_1d"), "cvar_95": ex.get("cvar_95_1d"),
                   "turnover": sm.get("turnover_annual")}
        if b.get("status") == "ok":
            weights[m] = b["weights"]
    t = pd.DataFrame(rows).T
    console.print(table(t, {"gross": num, "net": signed, "long": num, "short": num, "beta": signed,
                            "vol_ex_ante": pct, "sharpe": num, "sortino": num, "max_dd": pct,
                            "var_95": lambda v: pct(v, 2), "cvar_95": lambda v: pct(v, 2), "turnover": pct},
                        index_name="Strategy"))
    console.print("VaR/CVaR: 1-day, 95%, historical, current weights. Turnover: annual, one-way.")
    for m, b in r.portfolios.items():
        if b.get("note"):
            console.print(f"  {m}: {escape(b['note'])}")
    console.print(table(pd.DataFrame(weights), {k: signed for k in weights}, title="Current weights (fraction of capital)",
                        index_name="Ticker"))

    if r.beta_comparison.empty:
        console.print("Long-only basket: beta-neutral and sector-neutral books need a SHORT leg; not applicable.")
        return
    console.print()
    console.print("[bold]50/50 vs beta-neutral[/] (equal weight within legs)")
    console.print(table(r.beta_comparison[["long", "short", "gross", "net", "beta"]],
                        {"long": num, "short": num, "gross": num, "net": signed, "beta": signed}, index_name=""))
    bn = r.sector_neutral.get("beta_neutral_backtest", {})
    if bn.get("summary"):
        console.print(f"Beta-neutral backtest: Sharpe {num(bn['summary'].get('sharpe'))}, "
                      f"realized beta {signed(bn['summary'].get('beta_realized'))}, "
                      f"max DD {pct(bn['summary'].get('max_drawdown'))}.")

    sn = r.sector_neutral
    if sn.get("after") is not None:
        exp = pd.DataFrame({"50/50": sn["before"], "sector neutral": sn["after"]}).fillna(0.0)
        console.print(table(exp, {c: signed for c in exp.columns}, title="Net sector exposure", index_name="Sector"))
    if sn.get("note"):
        console.print(f"  {escape(sn['note'])}")


def risk_analysis(console: Console, r: AnalysisResult) -> None:
    section(console, 11, f"RISK ANALYSIS ({r.primary['method']})")
    sm = r.primary.get("summary", {})
    rows = {
        "CAGR": pct(sm.get("cagr")), "Cumulative return": pct(sm.get("cumulative_return")),
        "Volatility": pct(sm.get("volatility")), "Downside deviation": pct(sm.get("downside_deviation")),
        "Max drawdown": pct(sm.get("max_drawdown")), "Sharpe": num(sm.get("sharpe")),
        "Sortino": num(sm.get("sortino")), "Calmar": num(sm.get("calmar")),
        "Information ratio": num(sm.get("information_ratio")),
        "Realized beta": signed(sm.get("beta_realized")), "Alpha (annual)": pct(sm.get("alpha_annual")),
        "VaR 95% 1d": pct(sm.get("var_95_1d"), 2), "CVaR 95% 1d": pct(sm.get("cvar_95_1d"), 2),
        "VaR 99% 1d": pct(sm.get("var_99_1d"), 2), "CVaR 99% 1d": pct(sm.get("cvar_99_1d"), 2),
        "Turnover (annual)": pct(sm.get("turnover_annual")), "Cost drag (annual)": pct(sm.get("cost_drag_annual"), 2),
        "Trades": num(sm.get("n_trades"), 0), "Win rate (periods)": pct(sm.get("win_rate_periods")),
        "Win rate (positions)": pct(sm.get("win_rate_positions")), "Avg win": pct(sm.get("avg_win"), 2),
        "Avg loss": pct(sm.get("avg_loss"), 2), "Profit factor": num(sm.get("profit_factor")),
    }
    t = Table(box=ASCII_SIMPLE_HEAD, show_edge=False)
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    for k, v in rows.items():
        t.add_row(k, v)
    console.print(t)
    rc = r.primary["ex_ante"]["risk_contribution"]
    console.print("Risk contribution (sums to 100%; negative = hedging): "
                  + ", ".join(f"{k} {pct(v)}" for k, v in rc.sort_values(ascending=False).items()))
    if r.historical:
        hs = r.historical["summary"]
        console.print(f"[bold]MODE B - historical signals backtest:[/] CAGR {pct(hs.get('cagr'))}, "
                      f"Sharpe {num(hs.get('sharpe'))}, max DD {pct(hs.get('max_drawdown'))}, "
                      f"IR {num(hs.get('information_ratio'))}.")


def stress_test(console: Console, r: AnalysisResult) -> None:
    section(console, 12, "STRESS TEST & MONTE CARLO")
    h = r.stress["historical"]
    if not h.empty:
        console.print(table(h.set_index("scenario"),
                            {c: pct for c in ["portfolio", "benchmark", "long_leg_pnl", "short_leg_pnl"]},
                            title="Historical episodes: today's weights held through past prices",
                            index_name="Scenario"))
        console.print("portfolio = long_leg_pnl + short_leg_pnl, as a fraction of capital. Only episodes inside "
                      "the downloaded data are shown.")
    console.print(table(r.stress["covariance"].set_index("shock"),
                        {"vol_annual": pct, "var_95_1d_normal": lambda v: pct(v, 2)},
                        title="Volatility / correlation shocks (ex-ante)", index_name="Shock"))
    if r.shorts:
        console.print("In a long/short book higher correlation between the legs makes the hedge WORK, so "
                      "'correlation -> 1' can lower risk. The dangerous case is the hedge breaking down.")
    console.print(table(r.stress["market"].set_index("market_move"),
                        {"pnl_current_beta": pct, "pnl_adverse_beta": pct},
                        title="Market shock x portfolio beta", index_name="Market move"))
    console.print("Adverse beta: each LONG at the 95th percentile and each SHORT at the 5th percentile of its "
                  "rolling 252d beta. A joint worst case, not a likely scenario.")
    liq = r.stress["liquidity"]
    slowest = liq["days_to_exit"].idxmax()
    console.print(f"Liquidity: slowest exit {liq.loc[slowest, 'days_to_exit']:.4f} days ({slowest}, position "
                  f"{money(liq.loc[slowest, 'position_usd'])} vs ADV {money(liq.loc[slowest, 'adv_usd'])}) at "
                  f"{pct(r.settings.get('stress.liquidity_participation', 0.1), 0)} participation. Forced-exit cost "
                  f"(TC x {r.settings.get('stress.liquidity_spread_multiplier', 3)}): {pct(r.stress['exit_cost'], 2)} of capital.")
    fund = r.fundamentals
    if fund is not None and "si_pct_float" in fund.columns and r.shorts:
        shorts = [t for t in r.shorts if t in fund.index]
        si = fund.loc[shorts, ["si_pct_float", "days_to_cover"]].apply(pd.to_numeric, errors="coerce")
        crowded = si[(si["si_pct_float"] > 0.10) | (si["days_to_cover"] > 5)]
        console.print("Short interest on SHORT names: " + ", ".join(
            f"{t} {pct(row['si_pct_float'])} ({num(row['days_to_cover'], 1)}d to cover)" for t, row in si.iterrows()))
        if len(crowded):
            console.print(f"[bold]Crowded shorts (>10% of shares or >5 days to cover): {', '.join(crowded.index)}.[/] "
                          "Squeeze risk and expensive borrow; the flat borrow cost in the backtest understates it.")
    mc = r.montecarlo
    console.print(f"[bold yellow]SIMULATED[/] - 1 year, {int(mc['bootstrap_neutral']['paths'])} paths, "
                  f"seed {r.settings.get('montecarlo.seed')}")
    labels = {"bootstrap_neutral": "bootstrap (neutral drift)", "parametric_neutral": "normal (neutral drift)",
              "bootstrap_historical_drift": "bootstrap (hist. drift)"}
    t = pd.DataFrame({labels[k]: {m: v for m, v in mc[k].items() if m != "paths"} for k in labels})
    console.print(table(t, {c: lambda v: num(v, 3) for c in t.columns}, index_name="Metric"))
    console.print("Neutral drift: the historical mean is removed and returns are centered on the risk-free "
                  "rate, so the simulation measures RISK (dispersion, drawdowns), not expected return. The "
                  "historical-drift column inherits Mode A's in-sample selection: an upper bound, not a forecast.")


def robustness(console: Console, r: AnalysisResult) -> None:
    section(console, 13, "ROBUSTNESS")
    rob = r.robustness
    if not rob:
        console.print("Not run. Select [3] in the menu.")
        return
    t = rob["table"].copy()
    t.index = t["dimension"] + " | " + t["variant"].astype(str)
    cols = [c for c in ["sharpe", "cagr", "max_drawdown", "turnover_annual", "cost_drag_annual"] if c in t.columns]
    console.print(table(t[cols], {"sharpe": num, "cagr": pct, "max_drawdown": pct, "turnover_annual": pct,
                                  "cost_drag_annual": lambda v: pct(v, 2)}, index_name="Variant"))
    console.print(f"Sharpe range across variants: {num(rob['sharpe_min'])} to {num(rob['sharpe_max'])} "
                  f"(dispersion {num(rob['sharpe_dispersion'])}). Read the dispersion, not the best row: "
                  "picking the best variant is overfitting.")


def trade_plan(console: Console, r: AnalysisResult) -> None:
    section(console, 14, "TRADE PLAN")
    tp = r.trade_plan
    if not tp:
        return
    missing = tp.get("wrds_missing")
    if missing:
        console.print(f"[bold red]WARNING: WRDS unavailable. Value, quality, analyst and short-interest "
                      f"({missing['weight']:.0%} of the Quant Score) are missing; verdicts below use price data "
                      f"only and can differ from a run with WRDS.[/]")
        console.print(f"  Reason: {escape(missing['reason'])}")
        console.print()
    plan = tp["plan"]
    p = tp["params"]
    corr = tp["corrections"]
    console.print(escape(validation_line(r)))
    if len(corr):
        console.print("[bold]CORRECTIONS TO THE RESEARCH SIGNAL[/]")
        for t, row in corr.iterrows():
            if row["verdict"] == "NO TRADE" and row.get("score_verdict") == "FLIP" and t in tp.get("unproven_flips", []):
                msg = (f"research says {row['research']}, data points the other way (aligned score "
                       f"{signed(row['aligned_score'])}), but the score has not shown it predicts: NO TRADE, "
                       "not FLIP")
            elif row["verdict"] == "FLIP" and row.get("unproven_flip"):
                msg = (f"research says {row['research']}, data points the other way (aligned score "
                       f"{signed(row['aligned_score'])}): model suggests {row['action']} at {p.reduce_size:.0%} "
                       "size, LOW conviction: the score has not shown it predicts here")
            elif row["verdict"] == "FLIP":
                msg = (f"research says {row['research']}, data points the other way (aligned score "
                       f"{signed(row['aligned_score'])}): model suggests {row['action']}")
            elif row["verdict"] == "NO TRADE":
                msg = (f"research says {row['research']}, data contradicts it (aligned score "
                       f"{signed(row['aligned_score'])}): do not trade until the thesis is reviewed")
            else:
                msg = (f"research says {row['research']}, data is neutral (aligned score "
                       f"{signed(row['aligned_score'])}): trade at {p.reduce_size:.0%} size")
            console.print(f"  {t:{max(6, *(len(str(x)) for x in corr.index))}} {row['verdict']:9} {msg}")
    else:
        console.print("No corrections: the data supports every research signal.")
    console.print()
    order = plan.assign(_o=plan["action"].map({"BUY": 0, "SELL SHORT": 1, "NO TRADE": 2})).sort_values("_o")
    price = lambda v: f"{v:,.2f}"  # noqa: E731
    # La moneda solo si hay alguna distinta de USD: los niveles van en la de cotizacion.
    foreign = "currency" in order and order["currency"].ne("USD").any()
    level_cols = ["action", "verdict", "entry", "stop", "tp1", "tp2", "stop_pct"] + (["currency"] if foreign else [])
    console.print(table(order[level_cols],
                        {"entry": price, "stop": price, "tp1": price, "tp2": price, "stop_pct": pct},
                        title="Levels" + (" (in each stock's trading currency; size and risk in USD)" if foreign
                                          else ""), index_name="Ticker"))
    size = order[["shares", "notional", "weight", "risk_capital_pct", "p_tp1_before_stop", "p_tp2_before_stop"]]
    size = size.rename(columns={"risk_capital_pct": "risk_cap", "p_tp1_before_stop": "P(TP1)",
                                "p_tp2_before_stop": "P(TP2)"})
    console.print(table(size, {"shares": lambda v: f"{int(v):,}", "notional": money,
                               "weight": lambda v: signed(v * 100, 1) + "%", "risk_cap": lambda v: pct(v, 2),
                               "P(TP1)": lambda v: pct(v, 0), "P(TP2)": lambda v: pct(v, 0)},
                        title="Size and historical base rate", index_name="Ticker"))
    capped = list(plan.index[plan["size_capped_by_risk"]])
    console.print(f"Entry: last close. Stop: {p.stop_atr} x ATR(14). TP1 = {p.tp_r[0]}R, TP2 = {p.tp_r[1]}R "
                  f"(R = entry-to-stop distance); exit 50% at each, move stop to entry after TP1. "
                  f"Size: portfolio weight with the corrected directions ({r.primary['method']}), capped so a "
                  f"stop costs at most {p.max_risk_per_trade:.0%} of capital (risk_cap)"
                  + (f" (capped: {', '.join(capped)})" if capped else "") + ".")
    h = tp.get("hedge")
    if h:
        console.print(f"[bold]BETA HEDGE:[/] {h['action']} {h['shares']:,} {h['ticker']} at {h['price']:,.2f} "
                      f"(notional {money(h['notional'])}). Book beta {signed(h['beta_before'])} -> "
                      f"{signed(h['beta_after'])}. Corrections and risk caps changed the book the optimizer "
                      "built; without this line it carries market exposure nobody chose.")
    elif tp.get("long_only"):
        console.print(f"Long-only basket: market beta ({signed(tp.get('beta', float('nan')))}) is the intended "
                      "exposure, so there is no beta hedge." if tp.get("gross") else
                      "Long-only basket: nothing to trade, so nothing to hedge.")
    elif not p.hedge_beta:
        console.print(f"[bold]Beta hedge disabled[/] (trade_plan.hedge_beta: false): the book carries beta "
                      f"{signed(tp.get('beta', float('nan')))} as is.")
    console.print(f"Book after corrections: gross {num(tp['gross'])}, net {signed(tp['net'])}, "
                  f"beta {signed(tp.get('beta', float('nan')))} before hedge. "
                  f"P(TP): share of past entries in this stock ({p.horizon_d}-session horizon, "
                  "same ATR multiples) that hit the target before the stop. A historical base rate, not a "
                  "forecast. Review at the next rebalance or before earnings.")
    expectations(console, r)


def _usd(v: object) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return "0" if abs(float(v)) < 0.5 else f"{float(v):+,.0f}"


def expectations(console: Console, r: AnalysisResult) -> None:
    """Backtest de la regla por nombre, del plan completo y de la canasta."""
    tp = r.trade_plan
    plan, p = tp["plan"], tp["params"]
    cols = ["bt_trades", "bt_win_rate", "bt_exp_r", "bt_profit_factor", "bt_avg_days", "exp_return_pct", "exp_pnl"]
    if all(c in plan.columns for c in cols):
        if "bt_t_stat" in plan.columns:
            cols = cols[:3] + ["bt_t_stat"] + cols[3:]
        # Parentesis y no corchetes: `rich` lee "[ret]" como estilo y se lo come.
        bt = plan[cols].rename(columns={"bt_trades": "n", "bt_win_rate": "win", "bt_exp_r": "E(R)",
                                        "bt_t_stat": "t", "bt_profit_factor": "PF", "bt_avg_days": "days",
                                        "exp_return_pct": "E(ret)", "exp_pnl": "E($)"})
        console.print()
        console.print(table(bt, {"n": lambda v: f"{int(v):,}", "win": lambda v: pct(v, 0), "E(R)": signed,
                                 "t": lambda v: signed(v, 1), "PF": num, "days": lambda v: num(v, 0),
                                 "E(ret)": lambda v: signed(v * 100, 1) + "%", "E($)": _usd},
                            title=f"Rule backtest: these levels opened every day of the last {p.history_d // 252} "
                                  f"years, {p.horizon_d}-session max, costs in", index_name="Ticker"))
        console.print("E(R): average result per trade in multiples of the risk at the stop. t: E(R) over its "
                      "Newey-West standard error (entries overlap); |t| < 2 is not distinguishable from zero. "
                      "E(ret): on the position. E($): on today's size, 0 when nothing is traded. Same-day stop "
                      "and target counts as stop; a gap through the stop fills at the day's worst price.")
        traded = tp["orders"].reindex(plan.index)["target"].fillna(0) != 0
        negative = plan[traded & (plan["bt_exp_r"] < 0)]
        if len(negative):
            t_of = plan["bt_t_stat"] if "bt_t_stat" in plan else pd.Series(dtype=float)
            console.print("[bold]Negative historical expectancy on traded names:[/] "
                          + ", ".join(f"{t} ({signed(v)}R" + (f", t {signed(t_of[t], 1)}" if t in t_of else "") + ")"
                                      for t, v in negative["bt_exp_r"].items())
                          + ". This rule lost money on these stocks in the past: review before entering.")
    pb = tp.get("plan_backtest")
    if pb:
        console.print(f"[bold]Plan expectation:[/] the whole plan opened together on each of {pb['n_dates']} past "
                      f"days ({pb['first']:%Y-%m-%d} to {pb['last']:%Y-%m-%d}): mean {_usd(pb['mean'])} "
                      f"({signed(pb['mean_pct'] * 100)}% of capital), 90% of outcomes between {_usd(pb['p05'])} "
                      f"and {_usd(pb['p95'])}, P(loss) {pct(pb['prob_loss'], 0)}, worst {_usd(pb['worst'])}. "
                      "Same dates for every name, so the correlation between them is kept."
                      + (" The beta hedge is not included." if tp.get("hedge") else ""))
    pf = tp.get("portfolio") or {}
    if pf.get("cagr") is not None:
        console.print(f"[bold]Basket as a portfolio[/] ({pf.get('method')}, research directions, Mode A "
                      f"in-sample): CAGR {pct(pf['cagr'])} vs {pct(pf.get('bench_cagr'))} for "
                      f"{r.settings.benchmark}, Sharpe {num(pf.get('sharpe'))}, max DD {pct(pf.get('max_drawdown'))}. "
                      f"Monte Carlo 1 year (neutral drift): median {pct(pf.get('mc_median_return'))}, 5%-95% "
                      f"{pct(pf.get('mc_p05_return'))} to {pct(pf.get('mc_p95_return'))}, P(loss) "
                      f"{pct(pf.get('mc_prob_loss'), 0)}, median max DD {pct(pf.get('mc_median_max_dd'))}.")
    console.print("All of this is a historical base rate: it carries each stock's past trend and knows "
                  "nothing about what comes next.")


def summary(console: Console, r: AnalysisResult, show_warnings: bool = True) -> None:
    console.print()
    console.print(LINE)
    console.print("FINAL QUANTITATIVE SUMMARY")
    console.print(LINE)
    sm = r.primary.get("summary", {})
    ex = r.primary.get("ex_ante", {})
    counts = r.agreement["agreement"].value_counts().to_dict()
    lines = [
        f"- The {r.primary['method']} {'LONG/SHORT' if r.shorts else 'long-only'} portfolio has an ex-ante "
        f"beta of {signed(ex.get('beta'))}, "
        f"net exposure {signed(ex.get('net'))} and ex-ante annualized volatility of {pct(ex.get('vol_annual'))}.",
        f"- On the hypothetical path (Mode A) its Sharpe was {num(sm.get('sharpe'))} with max drawdown "
        f"{pct(sm.get('max_drawdown'))}; the basket selection is in-sample.",
        "- Research/quant agreement: " + ", ".join(f"{v} {k.lower()}" for k, v in counts.items()) + ".",
    ]
    if r.spread:
        lines.append(f"- LONG minus SHORT daily spread: {pct(r.spread['mean_annual'])}/yr, p = "
                     f"{num(r.spread['p_value'], 4)} (Newey-West), in-sample.")
    best = r.scores["quant_score"].idxmax()
    worst = r.scores["quant_score"].idxmin()
    lines.append(f"- {best} has the highest composite quant score in the universe; {worst} the lowest.")
    conflicts = list(r.agreement.index[r.agreement["agreement"] == "SIGNAL CONFLICT"])
    if conflicts:
        lines.append(f"- Quantitative data contradicts the research signal for: {', '.join(conflicts)}.")
    if r.robustness:
        lines.append(f"- Sharpe across robustness variants ranges {num(r.robustness['sharpe_min'])} to "
                     f"{num(r.robustness['sharpe_max'])}.")
    for line in lines:
        console.print(line)
    if r.warnings and not show_warnings:
        console.print(f"- {len(dict.fromkeys(r.warnings))} warning(s): see the full report.")
    elif r.warnings:
        console.print()
        console.print("[bold]Warnings[/]")
        for w in dict.fromkeys(r.warnings):
            console.print(f"  ! {escape(w)}")


_NOTES = {
    "REDUCE": "half size, quant data neutral",
    "NO TRADE": "not traded, quant data contradicts research",
    "FLIP": "direction flipped, quant data strongly contradicts research",
    "NOT ANALYZED": "not in the ticker list: no signal, position left as is and outside the beta hedge",
}


def final_signal(console: Console, r: AnalysisResult) -> None:
    tp = r.trade_plan
    if not tp:
        return
    orders = tp["orders"]
    console.print()
    console.print(LINE)
    if tp.get("wrds_missing"):
        console.print("[bold red]PRICE-ONLY SIGNAL: WRDS unavailable, see the warning in [14].[/]")
    console.print("FINAL SIGNAL" + ("  (orders from current positions to the plan)" if tp.get("has_positions")
                                    else "  (no current positions given: orders open the plan from zero)"))
    console.print(LINE)
    # Ancho del ticker segun el mas largo: "SHOP.TO" (7) se pegaba a la senal.
    w = max(7, *(len(str(t)) + 1 for t in orders.index))
    console.print(f"{'Ticker':{w}}{'Signal':15}{'Qty':>7}  {'Now':>7} -> {'Target':<7}{'Stop':>9}{'TP1':>9}{'TP2':>9}")
    fmt = lambda v: "" if v is None or pd.isna(v) else f"{v:,.2f}"  # noqa: E731
    for t, o in orders.iterrows():
        tail = ("   beta hedge" if o["verdict"] == "HEDGE"
                else "" if o["signal"] == "NOT ANALYZED"
                else f"{fmt(o.get('stop')):>9}{fmt(o.get('tp1')):>9}{fmt(o.get('tp2')):>9}")
        qty = f"{int(o['qty']):,}" if o["qty"] else "-"
        console.print(f"{t:{w}}{o['signal']:15}{qty:>7}  {int(o['current']):>7,} -> {int(o['target']):<7,}{tail}")
    notes = []
    if not tp.get("has_book", True):
        notes.append("No book to size: after the corrections " + ("no LONG is left to trade" if tp.get("long_only")
                     else "one leg has nothing left to trade") + ", so no orders from a book. Each traded name "
                     "gets a standalone, directional size in its card below.")
    for verdict, text in _NOTES.items():
        names = list(orders.index[(orders["verdict"] == verdict)
                                  & ((orders["target"] != 0) | (verdict not in ("CONFIRM", "REDUCE")))])
        if names:
            notes.append(f"{', '.join(names)}: {text}.")
    reason = orders.get("hold_reason", pd.Series("", index=orders.index)).fillna("")
    band = list(orders.index[reason == "within hold band"])
    if band:
        notes.append(f"{', '.join(band)}: HOLD, already within {tp['params'].hold_band:.1%} of capital of the target.")
    no_book = list(orders.index[reason == "no long/short book: kept as is"])
    if no_book:
        notes.append(f"{', '.join(no_book)}: HOLD, after the corrections only one leg is left, so there is "
                     "no long/short book to size against.")
    kept = list(orders.index[reason == "kept against research"])
    if kept:
        notes.append(f"{', '.join(kept)}: current position is against research and the data does not back "
                     "research either: kept, with levels for the side you hold.")
    for n in notes:
        console.print(f"  {n}")
    pb = tp.get("plan_backtest")
    if pb and tp.get("gross"):
        console.print(f"Expected P&L of the plan (rule backtest): {_usd(tp.get('expected_pnl'))} per trade cycle; "
                      f"90% range {_usd(pb['p05'])} to {_usd(pb['p95'])}, P(loss) {pct(pb['prob_loss'], 0)}.")
    console.print(f"Orders at market next session. Levels from the {r.clean.close.index[-1]:%Y-%m-%d} close; stop and TP apply to "
                  "the whole resulting position.")
    integrity(console, r)
    time_stop_evidence(console, r)
    signal_cards(console, r)
    console.print("Mechanical output of the rules in config/quant_engine.yaml (thresholds, ATR, risk cap). "
                  "Not investment advice: review before executing.")


def time_stop_evidence(console: Console, r: AnalysisResult) -> None:
    """El stop por tiempo fuera de muestra, sumado sobre los nombres del plan."""
    plan = (r.trade_plan or {}).get("plan")
    if plan is None or "oos_ts_improvement" not in plan:
        return
    imp = pd.to_numeric(plan["oos_ts_improvement"], errors="coerce").dropna()
    if imp.empty:
        return
    console.print(f"Time stop out of sample: review day picked on the first half of each name's history, tested on "
                  f"the second. It improved E(R) in {(imp > 0).sum()} of {len(imp)} names here (mean "
                  f"{imp.mean():+.2f}R). It is part of the rule for shorts only: on 50 large caps it improved "
                  "shorts and hurt longs.")


def integrity(console: Console, r: AnalysisResult) -> None:
    """Las identidades del plan: acciones, nocional, riesgo, ordenes, cobertura."""
    errors = (r.trade_plan or {}).get("integrity_errors")
    if errors is None:
        return
    if errors:
        console.print("[bold red]PLAN INTEGRITY CHECK FAILED: do not send these orders until fixed.[/]")
        for e in errors:
            console.print(f"[red]  - {escape(e)}[/]")
    else:
        console.print("Plan integrity: prices, sides, shares = notional / price, risk caps, orders and hedge "
                      "all reconcile.")


def signal_cards(console: Console, r: AnalysisResult) -> None:
    """Una ficha por nombre: senal, por que, fundamentales y como ejecutarla."""
    from ..playbook import (  # noqa: PLC0415
        build_cards,
        card_size_note,
        standalone_total,
    )

    cards = build_cards(r)
    if not cards:
        return
    p = r.trade_plan["params"]
    bench = r.settings.benchmark
    bench_close = r.clean.close[bench].dropna()
    total = standalone_total(cards, bench, float(bench_close.iloc[-1]) if len(bench_close) else float("nan"))
    console.print()
    console.print(LINE)
    console.print("SIGNALS: WHAT TO DO, WHY, AND HOW")
    console.print(LINE)
    for t, c in cards.items():
        tag = c["signal"] + (f" | conviction {c['conviction']}" if c.get("conviction") else "")
        console.print(f"[bold]{escape(t)}[/]  {escape(tag)}")
        console.print(f"  {escape(c['headline'])}")
        if c["why_for"]:
            console.print(f"  Backs research {c['research']}: " + escape("; ".join(c["why_for"])))
        if c["why_against"]:
            console.print(f"  Against research {c['research']}: " + escape("; ".join(c["why_against"])))
        if c["fundamentals"]:
            console.print("  Fundamentals: " + escape(" | ".join(c["fundamentals"])))
        for line in c.get("valuation", []):
            console.print("  " + escape(line))
        for line in c.get("technical", []):
            console.print("  Technical: " + escape(line))
        for line in c.get("timing", []):
            console.print("  [bold]Timing:[/] " + escape(line))
        if c.get("valuation_vs_signal"):
            console.print(f"  [bold]{escape(c['valuation_vs_signal'][0].upper() + c['valuation_vs_signal'][1:])}.[/]")
        if c["data_flags"]:
            console.print("[yellow]  Data to verify: " + escape("; ".join(c["data_flags"])) + "[/]")
        if c.get("steps"):
            console.print(f"  Size: {c['shares']:,} shares, ~{c['notional']:,.0f} USD, stop costs "
                          f"{c['risk_pct']:.2%} of capital ({escape(card_size_note(c, p.max_risk_per_trade))}).")
            console.print("  How to execute:")
            for i, step in enumerate(c["steps"], 1):
                console.print(f"    {i}. {escape(step)}")
        if c.get("time_window"):
            console.print(f"  [bold]Time window ({'short' if c['signal'] == 'SHORT' else 'long'}):[/]")
            for line in c["time_window"]:
                console.print(f"    - {escape(line)}")
        console.print()
    if total:
        console.print(f"[bold]All standalone positions together[/] ({', '.join(total['names'])}): long "
                      f"{total['long']:,.0f} USD, short {total['short']:,.0f} USD, net {total['net']:+,.0f} USD, "
                      f"beta-weighted {total['beta_usd']:+,.0f} USD of {bench}, {total['risk_pct']:.2%} of capital "
                      f"if every stop hits. One hedge for all of them instead of one per name: "
                      f"{total['hedge_action']} {total['hedge_shares']:,} {bench} at ~{total['bench_price']:,.2f}.")
        console.print()


def brief(r: AnalysisResult, console: Console) -> None:
    """Lo que se ve en pantalla por defecto: contexto, resumen y que hacer.

    Las trece secciones de analisis van al informe en disco; en pantalla
    enterraban el plan al final de ~600 lineas.
    """
    header(console, r)
    summary(console, r, show_warnings=False)
    trade_plan(console, r)
    final_signal(console, r)


def render(r: AnalysisResult, console: Console) -> None:
    header(console, r)
    for fn in (data_quality, performance, risk, momentum, beta_alpha, correlation, factor_exposure, signals,
               long_vs_short, portfolio_construction, risk_analysis, stress_test, robustness, trade_plan, summary,
               final_signal):
        fn(console, r)
