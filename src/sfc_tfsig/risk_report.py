"""Reporte de riesgo en Markdown, para el comite. Y el resumen previo a operar.

Mismos principios que `report.py`:

1. **Todo numero lleva su contexto.** Un VaR sin su backtest, o un Monte Carlo
   sin el descuento que se le aplico, no informa.
2. **Porcentaje y dolares.** El comite decide en dolares: "VaR 99% = 2.7%" se
   discute distinto que "5.400 USD en uno de cada cien dias".
3. **El semaforo va primero.** Lo que exige una decision se lee antes que lo
   que la explica.

Salida en ASCII puro: el reporte se redirige a fichero y Windows cambia de
codificacion al hacerlo.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .metrics import TRADING_DAYS
from .paths import REPORT_DIR
from .risk.analysis import RiskAnalysis

_METHOD_LABELS = {
    "historico": "Historical",
    "normal": "Normal",
    "cornish_fisher": "Cornish-Fisher",
    "fhs": "FHS (today's vol)",
    "ewma_normal": "Normal with EWMA vol",
    "historico_cartera": "Historical, current portfolio (1 year)",
    "mc_normal": "Monte Carlo normal",
    "mc_t_student": "Monte Carlo Student t",
    "garch_fhs": "Filtered GARCH (Monte Carlo)",
}

_LIMITATIONS = """\
## Limitations of the risk analysis

1. **VaR is not the worst loss.** It is where the tail starts. ES says how
   heavy the tail is on average, and not even ES bounds what is not in the
   sample.
2. **The bootstrap does not invent crises.** It resamples real days of the
   backtest, which starts in 2012: it does not contain 2008. Stress tests and
   stressed VaR exist for that.
3. **The backtest inherits survivorship bias.** That is why the bootstrap runs
   with the annual haircut declared in `config/risk.toml`. Without it, the
   probabilities of losing would come out optimistic.
4. **One-year covariance.** Ex-ante risk assumes next month looks like the
   last year. In a regime change, correlations rise exactly when it matters
   most; the Student t and the stress tests cover part of that.
5. **Linear stress.** Hypothetical scenarios and the reverse test use beta and
   declared shocks: orders of magnitude, not forecasts. Names without prices
   in a historical episode are approximated by their beta.
6. **Limits monitor, they do not constrain.** A breached limit does not change
   the portfolio: it goes to the committee.
7. **GARCH with full-sample parameters.** Persistence and long-run variance
   are estimated once; if the regime changes, GARCH is slow to notice. The
   current portfolio's history approximates names that were not listed by
   their beta, with no specific risk: in those stretches it underestimates a
   little.
8. **Small factor model.** SIC sectors, four styles and size. It does not
   model currency, rates or commodities, which fall into specific risk or the
   sectors.
9. **Market impact by formula.** The square-root law is an empirical
   regularity, not a measurement of this book: capacity is an order of
   magnitude.
"""


# ---------------------------------------------------------------------------
#  Formato
# ---------------------------------------------------------------------------


def _pct(value, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def _num(value, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def _money(value) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:,.0f}"


def _limit_value(value: float, unit: str) -> str:
    if unit == "pct":
        return _pct(value)
    if unit == "days":
        return _num(value, 2)
    return _num(value, 2)


def _rows(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


# ---------------------------------------------------------------------------
#  Secciones
# ---------------------------------------------------------------------------


def limits_section(analysis: RiskAnalysis) -> str:
    rows = []
    for _, row in analysis.limits.iterrows():
        sign = "<=" if row["direction"] == "max" else ">="
        rows.append([
            row["label"],
            _limit_value(row["value"], row["unit"]),
            f"{sign} {_limit_value(row['threshold'], row['unit'])}",
            _pct(row["usage"], 0),
            row["status"],
            row["action"],
        ])
    lines = [f"## Limit traffic light: {analysis.status}", ""]
    lines += _rows(["Limit", "Value", "Policy", "Usage", "Status", "What to do"], rows)
    lines += [
        "",
        "WARNING fires at the warning threshold in `config/risk.toml`. A",
        "BREACHED limit does not change the portfolio: it is escalated to the",
        "committee, which decides whether to act or accept it in writing.",
        "",
    ]
    return "\n".join(lines) + "\n"


def ex_ante_section(analysis: RiskAnalysis, top: int = 10) -> str:
    ea, conc = analysis.ex_ante, analysis.concentration
    nav = analysis.nav
    rows = [
        ["Annual ex-ante volatility", _pct(ea.volatility)],
        [f"Volatility {analysis.benchmark}", _pct(ea.benchmark_volatility)],
        ["Beta", _num(ea.beta)],
        ["Ex-ante tracking error", _pct(ea.tracking_error)],
        ["Correlation with the benchmark", _num(ea.correlation)],
        ["Systematic risk (share of variance)", _pct(ea.systematic_share, 1)],
        ["Diversification ratio", _num(ea.diversification_ratio)],
        [f"Normal VaR {ea.confidence:.0%} 1 day", f"{_pct(ea.var_normal_1d)} ({_money(ea.var_normal_1d * nav)} USD)"],
        ["Ledoit-Wolf shrinkage", _pct(ea.shrinkage, 1)],
        ["Positions", str(conc.get("n_names", "n/a"))],
        ["Invested / cash", f"{_pct(conc.get('invested'), 1)} / {_pct(conc.get('cash'), 1)}"],
        ["Largest weight", f"{conc.get('max_weight_name', '')} {_pct(conc.get('max_weight'))}"],
        ["Five largest weights", _pct(conc.get("top5_weight"))],
        ["Effective number of names", _num(conc.get("effective_names"), 1)],
        ["Largest sector", f"{conc.get('max_sector', '')} {_pct(conc.get('max_sector_weight'))}"],
    ]
    lines = ["## Current portfolio: ex-ante risk", "",
             "Covariance of the last year of daily returns, shrunk toward",
             "constant correlation (Ledoit-Wolf). Weights drifted to today's prices,",
             "not the targets of the last rebalance.", ""]
    lines += _rows(["Measure", "Value"], rows)

    contrib = ea.contributions.head(top)
    lines += ["", f"### Risk contribution: top {top}", ""]
    lines += _rows(
        ["Ticker", "Sector", "Weight", "Volatility", "Beta", "Contribution", "Share of total risk"],
        [[t, r["sector"], _pct(r["weight"]), _pct(r["volatility"], 1), _num(r["beta"]),
          _pct(r["contribution"]), _pct(r["risk_share"], 1)] for t, r in contrib.iterrows()],
    )
    lines += ["", "### By sector", ""]
    lines += _rows(
        ["Sector", "Weight", "Share of total risk", "Names"],
        [[s, _pct(r["weight"]), _pct(r["risk_share"], 1), int(r["n_names"])]
         for s, r in ea.sectors.iterrows()],
    )
    lines += [
        "",
        "When a name's or sector's share of risk clearly exceeds its weight,",
        "the weight cap is not containing it: it adds volatility or correlation",
        "beyond what its size suggests.",
        "",
    ]

    fx = analysis.factor_exposure
    if fx is not None and not fx.empty:
        lines += ["### Factor exposure (weighted mean z-score)", ""]
        lines += _rows(
            ["Factor", "Portfolio", "Universe", "Active", "Weight with data"],
            [[f, _num(r["portfolio"]), _num(r["universe"]), _num(r["active"]),
              _pct(r["weight_covered"], 0)] for f, r in fx.iterrows()],
        )
        lines += ["",
                  "A high active momentum exposure is the risk that the momentum crash",
                  "scenario puts to the test.", ""]
    return "\n".join(lines) + "\n"


_METHOD_ORDER = {m: i for i, m in enumerate(_METHOD_LABELS)}


def _var_rows(table: pd.DataFrame, nav: float) -> list[list[str]]:
    rows = []
    ordered = table.assign(_order=table["method"].map(_METHOD_ORDER))
    for _, r in ordered.sort_values(["horizon_d", "confidence", "_order"]).iterrows():
        rows.append([
            _METHOD_LABELS.get(r["method"], r["method"]),
            f"{int(r['horizon_d'])}d",
            f"{r['confidence']:.0%}",
            _pct(r["var"]),
            _money(r["var"] * nav),
            _pct(r["es"]),
            _money(r["es"] * nav),
        ])
    return rows


def var_section(analysis: RiskAnalysis, top: int = 10) -> str:
    nav = analysis.nav
    headers = ["Method", "Horizon", "Confidence", "VaR", "VaR USD", "ES", "ES USD"]
    lines = [
        "## VaR and Expected Shortfall",
        "",
        "Positive loss as a fraction of NAV and in dollars at today's NAV",
        f"({_money(nav)} USD). 1-day 99% VaR = the loss exceeded one day in a",
        "hundred; ES = the average loss on those days.",
        "",
        "### Current portfolio",
        "",
    ]
    lines += _rows(headers, _var_rows(analysis.book_var, nav))
    lines += [
        "",
        "The Student t and the normal share the covariance: the difference",
        "between them is the fat-tail premium a normal VaR does not see.",
        "",
    ]
    if analysis.strategy_var is not None and not analysis.strategy_var.empty:
        lines += ["### Strategy: backtest history", ""]
        lines += _rows(headers, _var_rows(analysis.strategy_var, nav))
        lines += [
            "",
            "FHS rescales the historical tail to today's EWMA volatility: if it",
            "comes out above the historical one, the market is more nervous today",
            "than the sample average. It is the reference VaR.",
            "",
        ]

    comp = analysis.component_es
    if comp is not None and not comp.empty:
        lines += ["### Who explains the tail: 1-day 99% ES by component (Student t)", ""]
        lines += _rows(
            ["Ticker", "Sector", "Weight", "ES contributed", "USD", "Share of ES"],
            [[r["ticker"], r["sector"], _pct(r["weight"]), _pct(r["component_es"], 3),
              _money(r["component_es"] * nav), _pct(r["es_share"], 1)]
             for _, r in comp.head(top).iterrows()],
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def backtest_section(analysis: RiskAnalysis) -> str:
    table = analysis.var_backtest
    if table is None or table.empty:
        return ""
    recent = int(table["recent_obs"].iloc[0])
    lines = ["## VaR backtest", "",
             "Each day the VaR is re-estimated with the information of the previous",
             "close and compared with the actual loss.", ""]
    lines += _rows(
        ["Model", "Days", "Expected", "Exceptions", "Rate", "Kupiec p",
         "Christoffersen p", "Conditional coverage p", f"Last {recent} days", "Basel"],
        [[_METHOD_LABELS.get(r["model"], r["model"]), int(r["n_obs"]), _num(r["expected"], 1),
          int(r["exceptions"]), _pct(r["rate"]), _num(r["kupiec_p"], 3),
          _num(r["christoffersen_p"], 3), _num(r["conditional_p"], 3),
          int(r["recent_exceptions"]), r["zone"]] for _, r in table.iterrows()],
    )
    lines += [
        "",
        "- Kupiec: p < 0.05 rejects that the exception frequency is the promised",
        "  one. It fails both a VaR that is exceeded too often and one that is",
        "  never exceeded (idle capital).",
        "- Christoffersen: p < 0.05 means clustered exceptions. The model does",
        "  not react to volatility changes and losses arrive together.",
        "- Basel: green, yellow or red by the binomial probability of seeing that",
        "  many exceptions with a correct model.",
        "",
    ]
    return "\n".join(lines) + "\n"


def tail_section(analysis: RiskAnalysis) -> str:
    tail = analysis.tail
    if not tail:
        return ""
    rows = [
        ["Days", str(tail["n_days"])],
        ["Skewness", _num(tail["skew"])],
        ["Excess kurtosis", _num(tail["excess_kurtosis"])],
        ["Worst day", f"{_pct(tail['worst_day'])} ({pd.Timestamp(tail['worst_day_date']).date()})"],
        ["Best day", f"{_pct(tail['best_day'])} ({pd.Timestamp(tail['best_day_date']).date()})"],
        ["Days below -3 sigma", f"{_pct(tail['freq_below_3sigma'], 3)} "
                                        f"(normal: {_pct(tail['normal_freq_below_3sigma'], 3)})"],
        ["Tail ratio (p95 / |p5|)", _num(tail["tail_ratio"])],
    ]
    lines = ["## Tails of the daily distribution", ""]
    lines += _rows(["Measure", "Value"], rows)
    lines += ["", "Excess kurtosis above 3, or days below -3 sigma several times more",
              "frequent than under the normal, invalidate the normal VaR as an estimate.", ""]
    return "\n".join(lines) + "\n"


def bootstrap_section(analysis: RiskAnalysis) -> str:
    summary = analysis.bootstrap_summary
    if summary is None or summary.empty:
        return ""
    boot = analysis.bootstrap
    nav = analysis.nav
    horizons = list(summary.index)
    header = ["Measure"] + [f"{summary.loc[h, 'years']:.0f} years" for h in horizons]

    def line(label: str, column: str, fmt) -> list[str]:
        return [label] + [fmt(summary.loc[h, column]) if column in summary else "n/a" for h in horizons]

    rows = [
        line("Cumulative return p5", "p5", _pct),
        line("Cumulative return p25", "p25", _pct),
        line("Median cumulative return", "p50", _pct),
        line("Cumulative return p75", "p75", _pct),
        line("Cumulative return p95", "p95", _pct),
        line("Median annual return", "median_annualized", _pct),
        line(f"Median {analysis.benchmark}", "benchmark_p50", _pct),
        line("Probability of losing money", "prob_loss", lambda v: _pct(v, 1)),
        line(f"Probability of trailing {analysis.benchmark}", "prob_underperform",
             lambda v: _pct(v, 1)),
        # Un VaR negativo significa que incluso el percentil 5 gana dinero: se
        # dice con palabras, porque "VaR = -9.9%" se lee como una perdida.
        line("Horizon 95% VaR", "var_95", lambda v: _pct(v) if v > 0 else "no loss"),
        line("Horizon 95% VaR, USD", "var_95",
             lambda v: _money(v * nav) if v > 0 else "no loss"),
        line("Horizon 95% ES", "es_95", _pct),
        line("Median max drawdown", "median_max_drawdown", _pct),
        line("Max drawdown p95", "p95_max_drawdown", _pct),
        line("Longest stretch below the peak, median (sessions)", "median_max_underwater_d",
             lambda v: _num(v, 0)),
        line("Longest stretch below the peak, p95 (sessions)", "p95_max_underwater_d",
             lambda v: _num(v, 0)),
    ]
    for column in [c for c in summary.columns if c.startswith("prob_dd_")]:
        rows.append(line(f"Probability of drawdown >= {column.removeprefix('prob_dd_')}%",
                         column, lambda v: _pct(v, 1)))
    n_paths = boot.terminal[horizons[0]].shape[0] if boot else 0
    lines = [
        "## Monte Carlo: bootstrap of the history",
        "",
        f"{n_paths:,} paths by stationary block bootstrap over the real daily",
        "returns of the strategy and the benchmark, with the same indices for",
        "both. Haircut applied to the strategy before resampling:",
        f"{_pct(boot.haircut if boot else float('nan'))} a year (declared survivorship bias).",
        "",
    ]
    lines += _rows(header, rows)
    lines += [
        "",
        "The p95 max drawdown is the one you must be willing to sit through: one",
        "path in twenty exceeds it. If the committee would not sit through it, the",
        "strategy is too risky for the mandate, whatever its expected return.",
        "",
    ]
    return "\n".join(lines) + "\n"


def stress_section(analysis: RiskAnalysis) -> str:
    nav = analysis.nav
    lines = ["## Stress tests", ""]

    hist = analysis.historical
    if hist is not None and not hist.empty:
        lines += ["### Historical: today's portfolio in real episodes", ""]
        rows = []
        for _, r in hist.iterrows():
            if r.get("status") != "ok":
                rows.append([r["scenario"], r["start"], r["end"], r.get("status", "no data")]
                            + [""] * 6)
                continue
            rows.append([
                r["scenario"], r["start"], r["end"], _pct(r["portfolio"]), _pct(r["benchmark"]),
                _pct(r["relative"]), _money(r["portfolio"] * nav), _pct(r["weight_proxied"], 0),
                f"{r['worst_name']} ({_pct(r['worst_contribution'])})", _pct(r["strategy_then"]),
            ])
        lines += _rows(["Episode", "From", "To", "Portfolio", analysis.benchmark, "Relative",
                        "P&L USD", "Weight proxied by beta", "Worst name", "Strategy then"],
                       rows)
        lines += [
            "",
            "'Weight proxied by beta': share of the portfolio without prices in the",
            "episode (not listed yet), moved with its beta. Above 50%, the scenario",
            "measures the portfolio's beta, not its names. 'Strategy then': what the",
            "backtest lost with the portfolio it held that day.",
            "",
        ]

    hyp = analysis.hypothetical
    if hyp is not None and not hyp.empty:
        lines += ["### Hypothetical", ""]
        lines += _rows(
            ["Scenario", "Market", "Portfolio", "Relative", "P&L USD", "From beta",
             "From sector", "From factor", "Worst name"],
            [[r["scenario"], _pct(r["market"]), _pct(r["portfolio"]), _pct(r["relative"]),
              _money(r["portfolio"] * nav), _pct(r["from_beta"]), _pct(r["from_sectors"]),
              _pct(r["from_factors"]), f"{r['worst_name']} ({_pct(r['worst_contribution'])})"]
             for _, r in hyp.iterrows()],
        )
        lines += [
            "",
            "Shock per name = beta x market + its sector's extra + factor shock x",
            "its z-score. The extras are judgment declared in `config/risk.toml`,",
            "not a calibration.",
            "",
        ]

    rev = analysis.reverse
    if rev is not None and not rev.empty:
        lines += ["### Reverse: which market fall produces each loss", ""]
        lines += _rows(
            ["Portfolio loss", "USD", f"Required fall in {analysis.benchmark}"],
            [[_pct(r["loss"], 0), _money(r["loss"] * nav),
              _pct(r["market_move"], 1) if r["plausible"] else "not reachable through beta"]
             for _, r in rev.iterrows()],
        )
        lines += ["", f"Linear in the ex-ante beta ({_num(analysis.ex_ante.beta)}) and without specific",
                  "risk: an order of magnitude, not a forecast.", ""]

    sv = analysis.stressed
    if sv:
        lines += ["### Stressed VaR: today's portfolio in its worst year", ""]
        lines += _rows(["Measure", "Value"], [
            ["Hardest window", f"{pd.Timestamp(sv['window_start']).date()} to "
                                 f"{pd.Timestamp(sv['window_end']).date()}"],
            ["Stressed 1-day 99% VaR", f"{_pct(sv['stressed_var'])} ({_money(sv['stressed_var'] * nav)} USD)"],
            ["Stressed 1-day 99% ES", f"{_pct(sv['stressed_es'])} ({_money(sv['stressed_es'] * nav)} USD)"],
            ["1-day 99% VaR of the last year", _pct(sv["current_var"])],
            ["Stressed / current", _num(sv.get("ratio"))],
            ["Weight proxied by beta in the window", _pct(sv.get("weight_proxied_in_window"), 0)],
            ["History from", str(pd.Timestamp(sv["history_start"]).date())],
        ])
        lines += ["",
                  "Historical VaR of the current portfolio in each rolling window of its",
                  "rebuilt history; the worst one is taken (Basel 2.5). It answers 'how",
                  "much a bad day would cost if the worst year came back', not 'a bad day today'.",
                  ""]

    cov = analysis.covariance_stress
    if cov is not None and not cov.empty:
        lines += ["### Volatility and correlation shocks", ""]
        lines += _rows(
            ["Scenario", "Vol x", "Toward correlation 1", "Annual volatility", "Normal VaR 99% 1d",
             "USD", "Versus base"],
            [[r["scenario"], _num(r["vol_mult"], 1), _pct(r["corr_blend"], 0), _pct(r["vol_annual"]),
              _pct(r["var_1d"]), _money(r["var_1d"] * nav), f"x{_num(r['vs_base'])}"]
             for _, r in cov.iterrows()],
        )
        lines += ["",
                  "The crisis correlation alone measures how much diversification the",
                  "portfolio takes for granted: if it multiplies risk almost like doubling",
                  "volatility, the thirty names behave like a few.", ""]

    worst = analysis.worst_windows
    if worst is not None and not worst.empty:
        lines += ["### Worst windows of the strategy in the backtest", ""]
        lines += _rows(
            ["Sessions", "From", "To", "Strategy", analysis.benchmark],
            [[int(r["window_d"]), r["start"], r["end"], _pct(r["strategy"]),
              _pct(r.get("benchmark"))] for _, r in worst.iterrows()],
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def liquidity_section(analysis: RiskAnalysis, top: int = 5) -> str:
    s = analysis.liquidity_summary
    lines = ["## Liquidity and capacity", ""]
    lines += _rows(["Measure", "Value"], [
        ["Days to liquidate the least liquid position",
         f"{_num(s.get('max_days'))} ({s.get('max_days_name', '')})"],
        ["Share sellable in 1 session", _pct(s.get("liquid_1d"), 1)],
        ["Share sellable in 5 sessions", _pct(s.get("liquid_5d"), 1)],
        ["Capacity (max NAV at the days limit)",
         f"{_money(s.get('capacity_nav'))} USD ({s.get('capacity_name', '')})"],
        ["Weight without measurable volume", _pct(s.get("weight_without_adv"), 1)],
    ])
    table = analysis.liquidity.dropna(subset=["days_to_liquidate"]).head(top)
    if not table.empty:
        lines += ["", f"### {top} least liquid positions", ""]
        lines += _rows(
            ["Ticker", "Weight", "Value USD", "Daily volume USD", "Days"],
            [[t, _pct(r["weight"]), _money(r["value"]), _money(r["adv"]),
              _num(r["days_to_liquidate"])] for t, r in table.iterrows()],
        )
    lines += ["",
              "Capacity is the NAV at which the least liquid position would take the",
              "days limit to sell. Above it, the backtest's fixed costs stop being",
              "credible and market impact would have to be modeled.", ""]

    liq = analysis.liquidation
    if liq is not None and not liq.empty:
        lines += ["### Cost of liquidating the whole book", ""]
        lines += _rows(
            ["Conditions", "Cost", "USD"],
            [["Normal" if idx == "normal" else "Crisis (volume /2, vol x2, fixed cost x3)",
              _pct(r["cost"]), _money(r["cost_usd"])] for idx, r in liq.iterrows()],
        )
        lines += ["", "The backtest's fixed cost plus impact by the square-root law.",
                  "It is the loss added to the market's when you are forced to sell.", ""]

    cap = analysis.capacity
    if cap is not None and not cap.empty:
        lines += ["### Capacity: annual cost of rebalancing by capital", ""]
        lines += _rows(
            ["NAV", "Annual fixed cost", "Annual impact", "Annual total", "Largest share of daily volume"],
            [[_money(nav_level), _pct(r["fixed_annual"]), _pct(r["impact_annual"]),
              _pct(r["total_annual"]), _pct(r["max_participation"], 1)] for nav_level, r in cap.iterrows()],
        )
        lines += ["",
                  f"NAV at which the annual impact reaches the policy budget: "
                  f"{_money(analysis.capacity_nav)} USD. The backtest charges no impact: above",
                  "that figure its excess return is no longer reachable.", ""]
    return "\n".join(lines) + "\n"


def factor_section(analysis: RiskAnalysis) -> str:
    fr = analysis.factor_risk
    if fr is None:
        return ""
    lines = [
        "## Factor model: where the risk comes from",
        "",
        f"Daily cross-sectional regression of the universe ({fr.n_days} days, mean R2",
        f"{_pct(fr.mean_r2, 1)}) on sectors, the model's four styles and size.",
        "",
        f"Factor model volatility: {_pct(fr.total_vol)}, against",
        f"{_pct(analysis.ex_ante.volatility)} from Ledoit-Wolf. They do not measure the same thing:",
        "Ledoit-Wolf averages how these names moved over the last year; the factor",
        "model applies TODAY's exposures to the factors' volatility. If the factor",
        "figure is higher, the portfolio is more loaded on some style today than",
        "during the year, and the forward-looking figure is the factor one.",
        "",
    ]
    lines += _rows(
        ["Source", "Share of total risk (variance)", "Contribution to volatility"],
        [[group, _pct(r["variance_share"], 1), _pct(r["vol_contribution"])]
         for group, r in fr.decomposition.iterrows()],
    )
    styles = fr.factor_contributions[~fr.factor_contributions.index.str.startswith("sector_")]
    if not styles.empty:
        lines += ["", "### Exposure to the styles", ""]
        lines += _rows(
            ["Factor", "Exposure (z)", "Factor volatility", "Share of total risk"],
            [[f, _num(r["exposure"]), _pct(r["factor_vol"]), _pct(r["variance_share"], 1)]
             for f, r in styles.iterrows()],
        )
    lines += [
        "",
        "Low specific risk with thirty names means the portfolio is mostly a bet",
        "on the market, sectors and styles: name selection weighs less than it",
        "seems. A style with a large share of risk is the book's real bet,",
        "whether it was sought or not.",
        "",
    ]
    return "\n".join(lines) + "\n"


def conditional_section(analysis: RiskAnalysis) -> str:
    if not analysis.garch_fits:
        return ""
    nav = analysis.nav
    lines = ["## Conditional volatility: GARCH and filtered Monte Carlo", ""]
    lines += _rows(
        ["Series", "Alpha", "Beta", "Persistence", "Half-life (sessions)",
         "Volatility today", "Long-run volatility"],
        [[label, _num(f.alpha, 3), _num(f.beta, 3), _num(f.persistence, 3), _num(f.half_life_d, 0),
          _pct(f.current_vol), _pct(f.long_run_vol)] for label, f in analysis.garch_fits.items()],
    )
    terms = analysis.vol_terms
    if terms is not None and not terms.empty:
        lines += ["", "### Forecast annual volatility by horizon", ""]
        lines += _rows(
            ["Horizon (sessions)"] + list(terms.columns),
            [[int(h)] + [_pct(v) for v in row] for h, row in terms.iterrows()],
        )
    cond = analysis.conditional_var
    if cond is not None and not cond.empty:
        lines += ["", "### Conditional VaR and ES", ""]
        lines += _rows(
            ["Series", "Horizon", "Confidence", "VaR", "VaR USD", "ES", "ES USD"],
            [[r["series"], f"{int(r['horizon_d'])}d", f"{r['confidence']:.0%}", _pct(r["var"]),
              _money(r["var"] * nav), _pct(r["es"]), _money(r["es"] * nav)]
             for _, r in cond.sort_values(["series", "horizon_d", "confidence"]).iterrows()],
        )
    lines += [
        "",
        "Historical residuals fed back into the GARCH recursion from today's",
        "volatility: each path has volatility clusters and volatility reverts to",
        "its mean with the half-life in the table. It is the reference multi-day",
        "VaR; square-root-of-time overstates it after a spike and understates it",
        "in calm markets. 'portfolio' is today's portfolio over its rebuilt history.",
        "",
    ]
    return "\n".join(lines) + "\n"


def history_section(analysis: RiskAnalysis) -> str:
    summary = analysis.rolling_summary
    calib = analysis.calibration_summary
    if (summary is None or summary.empty) and not calib:
        return ""
    labels = {"vol": "Volatility", "beta": "Beta", "tracking_error": "Tracking error",
              "benchmark_vol": f"Volatility {analysis.benchmark}"}
    lines = ["## Risk history and model calibration", ""]
    if summary is not None and not summary.empty:
        fmt = lambda key, v: _num(v) if key == "beta" else _pct(v)  # noqa: E731
        lines += ["### Rolling risk: today against the strategy's history", ""]
        lines += _rows(
            ["Measure", "Today", "Median", "p10", "p90", "Today's percentile"],
            [[labels.get(k, k), fmt(k, r["current"]), fmt(k, r["median"]), fmt(k, r["p10"]),
              fmt(k, r["p90"]), _pct(r["percentile"], 0)] for k, r in summary.iterrows()],
        )
        lines.append("")
    if calib:
        lines += ["### Calibration: forecast versus realized volatility", ""]
        lines += _rows(["Measure", "Value"], [
            ["Periods between rebalances evaluated", str(calib["n_periods"])],
            ["Bias statistic (return deviation / forecast)", _num(calib["bias_statistic"])],
            ["95% interval if the model is right",
             f"{_num(calib['ci_low'])} to {_num(calib['ci_high'])}"],
            ["Average forecast volatility", _pct(calib["mean_predicted_vol"])],
            ["Average realized volatility", _pct(calib["mean_realized_vol"])],
            ["Periods beyond 2 deviations", _pct(calib["share_beyond_2sd"], 1)],
            ["Verdict", calib["verdict"]],
        ])
        lines += ["",
                  "At each rebalance, the volatility the model would have forecast with",
                  "the previous year is computed, without later data. If the bias comes",
                  "out above the interval, the model underestimates risk and every ex-ante",
                  "VaR in this report is optimistic in that proportion.", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
#  Reporte completo y resumen previo a operar
# ---------------------------------------------------------------------------


def build_risk_report(analysis: RiskAnalysis, cfg: Config, risk_cfg: Config) -> str:
    """Reporte de riesgo completo en Markdown."""
    header = [
        f"# Risk report -- {cfg.get('meta.name')}",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Portfolio as of: {analysis.as_of.date()} | NAV {_money(analysis.nav)} "
        f"{cfg.get('meta.base_currency')} | benchmark {analysis.benchmark}",
        f"- Strategy: `{cfg.source.name if cfg.source else 'in memory'}` "
        f"(fingerprint `{cfg.fingerprint}`)",
        f"- Risk policy: `{risk_cfg.source.name if risk_cfg.source else 'in memory'}` "
        f"(fingerprint `{risk_cfg.fingerprint}`)",
        f"- Overall status: **{analysis.status}**",
        "",
        "> Two risk reports are comparable if BOTH fingerprints match.",
        "> Changing the risk policy does not change the portfolio, but it",
        "> does change the numbers in this report.",
        "",
    ]
    sections = [
        "\n".join(header),
        limits_section(analysis),
        ex_ante_section(analysis),
        factor_section(analysis),
        var_section(analysis),
        conditional_section(analysis),
        backtest_section(analysis),
        tail_section(analysis),
        bootstrap_section(analysis),
        stress_section(analysis),
        history_section(analysis),
        liquidity_section(analysis),
    ]
    if analysis.warnings:
        unique = list(dict.fromkeys(analysis.warnings))
        sections.append("## Warnings\n\n" + "\n".join(f"- {w}" for w in unique) + "\n")
    sections.append(_LIMITATIONS)
    return "\n".join(s for s in sections if s)


def render_pre_trade(analysis: RiskAnalysis) -> str:
    """Resumen de riesgo de la cartera OBJETIVO, para la hoja de ordenes.

    Se mira antes de operar, no despues: si la cartera propuesta excede un
    limite, el comite lo sabe cuando todavia puede no aprobarla.
    """
    ea = analysis.ex_ante
    worst = (analysis.hypothetical.sort_values("portfolio").iloc[0]
             if not analysis.hypothetical.empty else None)
    lines = [
        f"## Target portfolio risk: {analysis.status}",
        "",
        f"- Risk policy: fingerprint `{analysis.risk_fingerprint}`",
        f"- Ex-ante volatility {_pct(ea.volatility)} | beta {_num(ea.beta)} | "
        f"tracking error {_pct(ea.tracking_error)}",
        f"- 1-day 99% VaR (Student t): {_pct(analysis.measures.get('var_99_1d'))} "
        f"({_money(analysis.measures.get('var_99_1d', float('nan')) * analysis.nav)} USD)",
        f"- Effective number of names: {_num(analysis.concentration.get('effective_names'), 1)}",
    ]
    if analysis.stressed:
        lines.append(f"- Stressed 1-day 99% VaR: {_pct(analysis.stressed['stressed_var'])} "
                     f"(worst window from {pd.Timestamp(analysis.stressed['window_start']).date()})")
    if analysis.factor_risk is not None:
        decomposition = analysis.factor_risk.decomposition["variance_share"]
        top = decomposition.drop("specific", errors="ignore").idxmax()
        lines.append(f"- Largest source of risk: {top} ({_pct(decomposition[top], 1)}); "
                     f"specific {_pct(decomposition.get('specific'), 1)}")
    if worst is not None:
        lines.append(f"- Worst hypothetical scenario: {worst['scenario']}, {_pct(worst['portfolio'])} "
                     f"({_money(worst['portfolio'] * analysis.nav)} USD)")
    lines.append("")
    flagged = analysis.limits[analysis.limits["status"].isin(["BREACHED", "WARNING"])]
    if flagged.empty:
        lines.append("Every limit with data is green. Drawdown is not evaluated before trading.")
    else:
        lines += _rows(
            ["Limit", "Value", "Policy", "Status", "What to do"],
            [[r["label"], _limit_value(r["value"], r["unit"]),
              f"{'<=' if r['direction'] == 'max' else '>='} {_limit_value(r['threshold'], r['unit'])}",
              r["status"], r["action"]] for _, r in flagged.iterrows()],
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
#  Ficheros
# ---------------------------------------------------------------------------


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def save_risk_report(text: str, name: str = "riesgo") -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{name}_{_stamp()}.md"
    path.write_text(text, encoding="utf-8")
    return path


def save_risk_artifacts(analysis: RiskAnalysis, name: str = "riesgo") -> dict[str, Path]:
    """Tablas del reporte en CSV, para recalcular o auditar cualquier cifra."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    frames = {
        "limites": analysis.limits,
        "contribuciones": analysis.ex_ante.contributions.reset_index(),
        "var_cartera": analysis.book_var,
        "var_estrategia": analysis.strategy_var,
        "backtest_var": analysis.var_forecasts,
        "montecarlo": analysis.bootstrap_summary,
        "estres_historico": analysis.historical_detail,
        "estres_hipotetico": analysis.hypothetical_detail,
        "liquidez": analysis.liquidity,
        "choques_covarianza": analysis.covariance_stress,
        "var_condicional": analysis.conditional_var,
        "calibracion": analysis.calibration,
        "capacidad": analysis.capacity,
        "factores": (analysis.factor_risk.factor_contributions
                     if analysis.factor_risk is not None else None),
    }
    paths: dict[str, Path] = {}
    for label, frame in frames.items():
        if frame is None or frame.empty:
            continue
        path = REPORT_DIR / f"{name}_{stamp}_{label}.csv"
        keep_index = label in ("backtest_var", "montecarlo", "liquidez", "capacidad", "factores")
        frame.to_csv(path, index=keep_index)
        paths[label] = path
    return paths


def save_risk_charts(analysis: RiskAnalysis, name: str = "riesgo") -> Path | None:
    """Abanico del Monte Carlo, backtest del VaR, volatilidad en el tiempo y riesgo por sector."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{name}_{_stamp()}_graficos.png"
    fig, axes = plt.subplots(4, 1, figsize=(11, 16))
    ax_fan, ax_var, ax_vol, ax_sector = axes

    boot = analysis.bootstrap
    if boot is not None and not boot.fan.empty:
        fan = boot.fan
        years = fan.index.to_numpy() / TRADING_DAYS
        ax_fan.fill_between(years, fan["p5"], fan["p95"], alpha=0.2, color="steelblue", label="p5-p95")
        ax_fan.fill_between(years, fan["p25"], fan["p75"], alpha=0.35, color="steelblue", label="p25-p75")
        ax_fan.plot(years, fan["p50"], color="navy", linewidth=1.4, label="median")
        ax_fan.axhline(1.0, color="grey", linewidth=0.8)
        ax_fan.set_xlabel("years")
        ax_fan.set_ylabel("wealth (1 = today)")
        ax_fan.set_title("Monte Carlo: bootstrap of the history")
        ax_fan.legend(loc="upper left")
        ax_fan.grid(alpha=0.3)

    forecasts = analysis.var_forecasts
    if forecasts is not None and not forecasts.empty:
        recent = forecasts.tail(750)
        ax_var.plot(recent.index, recent["return"] * 100, linewidth=0.6, color="grey", label="daily return")
        ax_var.plot(recent.index, -recent["fhs"] * 100, color="crimson", linewidth=1.0, label="-VaR FHS")
        hits = recent[recent["return"] < -recent["fhs"]]
        ax_var.scatter(hits.index, hits["return"] * 100, color="crimson", s=14, zorder=3, label="exceptions")
        ax_var.set_ylabel("%")
        ax_var.set_title("VaR backtest (last 3 years)")
        ax_var.legend(loc="lower left")
        ax_var.grid(alpha=0.3)

    rolling = analysis.rolling
    if rolling is not None and not rolling.empty:
        ax_vol.plot(rolling.index, rolling["vol"] * 100, linewidth=1.0, label="strategy, rolling")
        if "benchmark_vol" in rolling:
            ax_vol.plot(rolling.index, rolling["benchmark_vol"] * 100, linewidth=0.9, alpha=0.7,
                        label=f"{analysis.benchmark}, rolling")
        fit = analysis.garch_fits.get("strategy")
        if fit is not None:
            ax_vol.plot(fit.sigma.index, fit.sigma * np.sqrt(TRADING_DAYS) * 100, linewidth=0.8,
                        color="crimson", alpha=0.8, label="conditional GARCH")
            ax_vol.axhline(fit.long_run_vol * 100, color="grey", linestyle="--", linewidth=0.8,
                           label="GARCH long run")
        ax_vol.set_ylabel("% a year")
        ax_vol.set_title("Volatility over time")
        ax_vol.legend(loc="upper left")
        ax_vol.grid(alpha=0.3)

    sectors = analysis.ex_ante.sectors
    if not sectors.empty:
        x = np.arange(len(sectors))
        ax_sector.bar(x - 0.2, sectors["weight"] * 100, width=0.4, label="weight")
        ax_sector.bar(x + 0.2, sectors["risk_share"] * 100, width=0.4, label="risk")
        ax_sector.set_xticks(x)
        ax_sector.set_xticklabels(sectors.index, rotation=30, ha="right")
        ax_sector.set_ylabel("%")
        ax_sector.set_title("Weight versus contribution to risk, by sector")
        ax_sector.legend()
        ax_sector.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
