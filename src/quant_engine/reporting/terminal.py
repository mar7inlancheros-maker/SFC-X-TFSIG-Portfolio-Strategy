"""Reporte en terminal con `rich`. Solo presenta: todo se calcula en `analysis.py`.

Regla de redaccion del motor: DESCRIBIR, nunca recomendar. "NVDA tiene el
mayor score de momentum del universo", nunca "compra NVDA". Las frases de
interpretacion de abajo estan escritas asi a proposito.
"""

from __future__ import annotations

import math
from typing import Callable

import pandas as pd
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ..analysis import AnalysisResult

Fmt = Callable[[object], str]
LINE = "=" * 60


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
    t = Table(title=title, box=box.SIMPLE_HEAD, show_edge=False, title_justify="left")
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
    console.print(f"Short:      {' '.join(r.shorts)}")
    console.print(f"Benchmark:  {s.benchmark}")
    console.print(f"Period:     {s.lookback} ({r.eval_start.date()} to {r.as_of.date()})")
    console.print(f"Construction: {s.construction} | Rebalance: {s.rebalance} | "
                  f"Capital: {money(s.initial_capital)} | TC: {pct(s.transaction_cost, 2)} | "
                  f"Borrow: {pct(s.borrow_cost, 2)}/yr | rf: {pct(s.risk_free_rate, 2)}")
    console.print(f"Config fingerprint: {s.fingerprint}")
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
    console.print(f"Average pairwise: {num(c['avg_pairwise'])} | within LONG: {num(c['avg_within_long'])} | "
                  f"within SHORT: {num(c['avg_within_short'])} | LONG vs SHORT: {num(c['avg_long_short'])}")
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
    console.print("Factors built from ETF spreads: size IWM-SPY, value IWD-IWF, momentum MTUM-SPY, "
                  "low_volatility USMV-SPY, quality QUAL-SPY. Not academic Fama-French factors.")
    if f.unavailable:
        for k, why in f.unavailable.items():
            console.print(f"[yellow]Factor {k} UNAVAILABLE: {escape(why)}[/]")
    if len(f.vif):
        console.print(f"Factor condition number: {num(f.condition_number, 1)}. VIF: "
                      + ", ".join(f"{k} {num(v, 1)}" for k, v in f.vif.items())
                      + ". VIF > 5: that factor's individual beta is unreliable.")


def signals(console: Console, r: AnalysisResult) -> None:
    section(console, 8, "QUANTITATIVE SIGNALS")
    w = r.settings.get("signal_weights", {}) or {}
    console.print("Quant Score weights: " + ", ".join(f"{k} {v:.0%}" for k, v in w.items()))
    a = r.agreement.copy()
    comps = r.scores[["momentum", "risk_adjusted_return", "volatility", "mean_reversion", "liquidity",
                      "beta", "statistical"]]
    a = a.join(comps)
    console.print(table(a[["research", "quant_score", "rank", "agreement", "momentum", "risk_adjusted_return",
                           "volatility", "mean_reversion", "liquidity", "beta", "statistical"]],
                        {c: signed for c in ["quant_score", "momentum", "risk_adjusted_return", "volatility",
                                             "mean_reversion", "liquidity", "beta", "statistical"]}
                        | {"rank": lambda v: str(v)}, index_name="Ticker"))
    th = r.settings.get("agreement", {}) or {}
    counts = a["agreement"].value_counts().to_dict()
    console.print("Agreement counts: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    console.print(f"Aligned score = score x (+1 LONG, -1 SHORT). STRONG >= {th.get('strong', 0.5)}, "
                  f"WEAK >= {th.get('weak', 0.15)}, CONFLICT <= {th.get('conflict', -0.15)}. "
                  "The quant score does not overwrite the research signal.")
    rs = r.rank_stability
    console.print(f"Rank stability under {rs['draws']} random weight draws: median Spearman "
                  f"{num(rs['median_spearman'])}, 5th pct {num(rs['p05_spearman'])}.")
    console.print("With N = {} names, cross-sectional z-scores rest on {} points each; small score "
                  "differences are noise.".format(len(a), len(a)))
    fm = r.feature_matrix
    console.print(table(fm, {c: signed for c in fm.columns}, title="Cross-sectional feature matrix (z-scores)",
                        index_name="Ticker"))


def long_vs_short(console: Console, r: AnalysisResult) -> None:
    section(console, 9, "LONG VS SHORT")
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
    console.print("[bold yellow]MODE A - HYPOTHETICAL PATH.[/] Today's LONG/SHORT basket applied to the past. "
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
    t = Table(box=box.SIMPLE_HEAD, show_edge=False)
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


def summary(console: Console, r: AnalysisResult) -> None:
    console.print()
    console.print(LINE)
    console.print("FINAL QUANTITATIVE SUMMARY")
    console.print(LINE)
    sm = r.primary.get("summary", {})
    ex = r.primary.get("ex_ante", {})
    counts = r.agreement["agreement"].value_counts().to_dict()
    lines = [
        f"- The {r.primary['method']} LONG/SHORT portfolio has an ex-ante beta of {signed(ex.get('beta'))}, "
        f"net exposure {signed(ex.get('net'))} and ex-ante annualized volatility of {pct(ex.get('vol_annual'))}.",
        f"- On the hypothetical path (Mode A) its Sharpe was {num(sm.get('sharpe'))} with max drawdown "
        f"{pct(sm.get('max_drawdown'))}; the basket selection is in-sample.",
        f"- Research/quant agreement: " + ", ".join(f"{v} {k.lower()}" for k, v in counts.items()) + ".",
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
    if r.warnings:
        console.print()
        console.print("[bold]Warnings[/]")
        for w in dict.fromkeys(r.warnings):
            console.print(f"  ! {escape(w)}")
    console.print()
    console.print("This engine describes; it does not recommend. It is a research and risk-analysis tool, "
                  "not an investment adviser.")


def render(r: AnalysisResult, console: Console) -> None:
    header(console, r)
    for fn in (data_quality, performance, risk, momentum, beta_alpha, correlation, factor_exposure, signals,
               long_vs_short, portfolio_construction, risk_analysis, stress_test, robustness, summary):
        fn(console, r)
