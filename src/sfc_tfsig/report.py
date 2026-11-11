"""Reporte del modelo en texto plano (Markdown), para el comite.

Tres principios:

1. **Todo numero lleva su contexto.** Un CAGR sin el del benchmark y sin el
   drawdown no informa: informa de que alguien quiere que el resultado parezca
   bueno.
2. **Las limitaciones van EN el reporte**, no en un anexo que nadie abre. Si el
   universo tiene sesgo de supervivencia, eso se lee antes que el Sharpe.
3. **El fingerprint de la configuracion va en la cabecera.** Un reporte sin el
   no se puede reproducir y no deberia discutirse en un comite.

Salida en ASCII puro y `enable_utf8_stdout()` en los puntos de entrada: los
reportes se redirigen a fichero constantemente y Windows cambia de codificacion
al hacerlo.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pandas as pd

from .backtest import BacktestResult
from .config import Config
from .metrics import Performance, yearly_returns
from .paths import REPORT_DIR

_LIMITATIONS = """\
## Declared limitations

These are not courtesy caveats: each one moves the result in a known
direction, and the committee should discount them before deciding anything.

1. **Survivorship bias.** The universe is built from the companies listed
   TODAY. Those that went bankrupt or were delisted are missing. This INFLATES
   the historical return; the typical size in US equities is 1 to 2 points of
   CAGR a year. It cannot be corrected without paid data.

2. **Partial Canadian coverage.** Fundamentals come from SEC XBRL, which covers
   Canadian issuers registered with the SEC (40-F / 20-F / 10-K) and not
   TSX-only ones. The Canadian part of the universe is therefore the large
   companies with a US listing.

3. **Sector classification by SIC.** SIC dates from 1987 and does not separate
   modern software or e-commerce well. Sector neutralization inherits that
   noise.

4. **No market impact.** Modeled costs are fixed commission, spread and
   slippage. At the simulated capital they are reasonable; at institutional
   scale they would not be.

5. **Fundamentals as first published.** Data is used as originally filed,
   without later restatements. That is the right way to avoid look-ahead, and
   it means some historical figures are the "wrong" ones -- the same ones the
   market had that day.
"""


def _fmt_pct(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def _fmt_num(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def _fmt_money(value: float) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.0f}"


def _table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Tabla Markdown sin dependencias externas."""
    if df is None or len(df) == 0:
        return "_(no data)_\n"
    frame = df.copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(
                lambda v: "" if pd.isna(v) else floatfmt.format(v)
            )
    headers = [str(frame.index.name or "")] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for index, row in frame.iterrows():
        lines.append("| " + " | ".join([str(index)] + [str(v) for v in row]) + " |")
    return "\n".join(lines) + "\n"


def performance_section(perf: Performance, cfg: Config) -> str:
    benchmark = cfg.get("backtest.benchmark")
    rows = [
        ("Period", f"{perf.start.date()} to {perf.end.date()} ({perf.years:.1f} years)"),
        ("Total return", _fmt_pct(perf.total_return)),
        ("CAGR", _fmt_pct(perf.cagr)),
        (f"CAGR {benchmark}", _fmt_pct(perf.benchmark_cagr)),
        ("Annual excess", _fmt_pct(perf.excess_cagr)),
        ("Volatility", _fmt_pct(perf.volatility)),
        ("Sharpe", _fmt_num(perf.sharpe)),
        ("Sortino", _fmt_num(perf.sortino)),
        ("Max drawdown", _fmt_pct(perf.max_drawdown)),
        (f"Max drawdown {benchmark}", _fmt_pct(perf.benchmark_max_drawdown)),
        ("Days underwater", str(perf.underwater_days) if perf.underwater_days else "n/a"),
        ("Calmar", _fmt_num(perf.calmar)),
        ("Positive months", _fmt_pct(perf.hit_rate_monthly)),
        ("Beta", _fmt_num(perf.beta)),
        ("Annual alpha", _fmt_pct(perf.alpha)),
        ("Tracking error", _fmt_pct(perf.tracking_error)),
        ("Information ratio", _fmt_num(perf.information_ratio)),
    ]
    lines = ["## Performance", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {label} | {value} |" for label, value in rows]
    return "\n".join(lines) + "\n"


def costs_section(result: BacktestResult, perf: Performance) -> str:
    if result.rebalances.empty:
        return "## Costs and turnover\n\n_(no rebalances)_\n"
    rebalances = result.rebalances
    total_cost = result.total_costs
    initial = float(rebalances["nav"].iloc[0])
    # Rebalanceos por ano REALES, no 12 fijo. Con el 12 cableado, una corrida
    # trimestral reportaba cuatro veces su rotacion anual y parecia la opcion
    # cara justo cuando es la barata.
    per_year = len(rebalances) / perf.years if perf.years else float("nan")
    annual_turnover = float(rebalances["turnover"].sum() / perf.years) if perf.years else float("nan")
    # Arrastre en puntos de NAV: suma de lo que cada rebalanceo le costo al NAV
    # de ESE momento. Dividir los dolares totales entre el capital inicial -- la
    # version anterior -- inflaba la cifra, porque los costes de 2025 se pagan
    # sobre un NAV varias veces mayor que el de 2012. Esta es la que se resta
    # directamente del CAGR.
    annual_cost = (
        float(rebalances["cost_bps_of_nav"].sum() / perf.years / 10_000.0)
        if perf.years else float("nan")
    )
    lines = [
        "## Costs and turnover",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Rebalances | {len(rebalances)} ({_fmt_num(per_year, 1)} per year) |",
        f"| Average turnover per rebalance | {_fmt_pct(result.average_turnover)} |",
        f"| Annualized turnover | {_fmt_pct(annual_turnover)} |",
        f"| Annual cost drag (NAV points) | {_fmt_pct(annual_cost)} |",
        f"| Total cost | {_fmt_money(total_cost)} |",
        f"| Cost over initial capital | {_fmt_pct(total_cost / initial if initial else float('nan'))} |",
        f"| Average cost per rebalance (bps of NAV) | {_fmt_num(rebalances['cost_bps_of_nav'].mean(), 1)} |",
        f"| Average positions | {_fmt_num(rebalances['n_positions'].mean(), 1)} |",
        "",
        "Cost is not an accounting detail: with the turnover above, a difference",
        "of 10 bps per trade changes the CAGR visibly. If the strategy only",
        "works with optimistic costs, it does not work.",
        "",
    ]
    return "\n".join(lines) + "\n"


def holdings_section(result: BacktestResult, top: int = 40) -> str:
    if result.holdings.empty:
        return "## Current portfolio\n\n_(no positions)_\n"
    last_date = result.holdings["date"].max()
    current = result.holdings[result.holdings["date"] == last_date].copy()
    current = current.sort_values("weight", ascending=False).head(top)
    current["weight"] = current["weight"].map(lambda v: _fmt_pct(v))
    current["price"] = current["price"].map(lambda v: f"{v:,.2f}")
    current["score"] = current["score"].map(lambda v: _fmt_num(v))
    columns = ["ticker", "sector", "weight", "price", "score"]
    lines = [f"## Portfolio as of {pd.Timestamp(last_date).date()}", "",
             "| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in current.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")

    sector_weights = (
        result.holdings[result.holdings["date"] == last_date]
        .groupby("sector")["weight"].sum().sort_values(ascending=False)
    )
    lines += ["", "### Sector exposure", "", "| Sector | Weight |", "|---|---|"]
    lines += [f"| {sector} | {_fmt_pct(weight)} |" for sector, weight in sector_weights.items()]
    return "\n".join(lines) + "\n"


def validation_section(validation: Mapping[str, object]) -> str:
    if not validation:
        return ""
    lines = ["## Statistical validation", ""]

    ic = validation.get("ic_summary") or {}
    if ic:
        lines += [
            "### Information coefficient (rank IC, monthly)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Mean IC | {_fmt_num(ic.get('mean'), 4)} |",
            f"| t (Newey-West) | {_fmt_num(ic.get('t_stat'))} |",
            f"| p-value | {_fmt_num(ic.get('p_value'), 4)} |",
            f"| Months with positive IC | {_fmt_pct(ic.get('hit_rate'))} |",
            f"| Months evaluated | {ic.get('n', 'n/a')} |",
            "",
            "Reference: a useful equity factor has a mean IC of 0.02 to 0.05.",
            "Above 0.10, the first thing to look for is a leak of future data,",
            "not an economic explanation.",
            "",
        ]

    means = validation.get("quantile_means")
    if means is not None and len(means):
        lines += ["### Average monthly return by quintile (Q1 = best score)", "",
                  "| Quintile | Average return |", "|---|---|"]
        for label, value in means.items():
            lines.append(f"| {label} | {_fmt_pct(value)} |")
        spread = validation.get("spread_test") or {}
        lines += [
            "",
            f"Spread Q1-Q5: {_fmt_pct(spread.get('mean'))} monthly, "
            f"t = {_fmt_num(spread.get('t_stat'))}, p = {_fmt_num(spread.get('p_value'), 4)}.",
            f"Monotonicity: {_fmt_num(validation.get('monotonicity'))} "
            "(-1.00 would be a perfect order from Q1 to Q5).",
            "",
        ]

    fm = validation.get("fama_macbeth")
    if isinstance(fm, pd.DataFrame) and not fm.empty:
        lines += ["### Fama-MacBeth by factor (Newey-West errors)", "",
                  "| Factor | Mean coef. | t | p |", "|---|---|---|---|"]
        for term, row in fm.iterrows():
            lines.append(
                f"| {term} | {_fmt_num(row['mean'], 5)} | {_fmt_num(row['t_stat'])} | "
                f"{_fmt_num(row['p_value'], 4)} |"
            )
        lines.append("")

    wf = validation.get("walk_forward")
    if isinstance(wf, pd.DataFrame) and not wf.empty:
        lines += ["### Walk-forward: in-sample and out-of-sample IC", "",
                  "| Window | Train IC | Test IC | Months |", "|---|---|---|---|"]
        for _, row in wf.iterrows():
            lines.append(
                f"| {row['window']} | {_fmt_num(row['train_ic'], 4)} | "
                f"{_fmt_num(row['test_ic'], 4)} | {int(row['test_months'])} |"
            )
        lines += [
            "",
            "The column that matters is the test one. If the train IC is high and",
            "the test IC is near zero, the model memorizes instead of predicting,",
            "and the full backtest is evidence of nothing.",
            "",
        ]
    return "\n".join(lines) + "\n"


def attribution_section(contributions: pd.DataFrame, yearly_excess: pd.DataFrame | None) -> str:
    """Amplitud del resultado: proceso o posiciones afortunadas.

    Va en el reporte y no en un analisis aparte porque es la primera pregunta
    que un LP hace ante un buen ano: cuantos nombres lo hicieron.
    """
    from . import attribution as attr

    if contributions is None or contributions.empty:
        return ""

    summary = attr.concentration_summary(contributions)
    lines = [
        "## Attribution",
        "",
        "### Breadth over the whole sample",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Names ever held | {summary['n_names_ever_held']} |",
        f"| Names with a positive contribution | {_fmt_pct(summary['pct_names_positive'])} |",
        f"| Share of the gross from the top 5 | {_fmt_pct(summary['top5_share'])} |",
        f"| Share of the gross from the top 10 | {_fmt_pct(summary['top10_share'])} |",
        f"| Names needed for half | {_fmt_num(summary['names_for_half'], 0)} |",
        f"| Gross without the top 5 | {_fmt_pct(summary['total_without_top5'])} "
        f"of {_fmt_pct(summary['total_gross_contribution'])} |",
        "",
        "If removing the five best names of the whole history makes the result",
        "disappear, the model has no process: it had five hits.",
        "",
    ]

    if yearly_excess is not None and not yearly_excess.empty:
        lines += [
            "### Annual excess and its breadth",
            "",
            "| Year | Strategy | Benchmark | Excess | Top-5 share of gross | Names for half |",
            "|---|---|---|---|---|---|",
        ]
        for year, row in yearly_excess.iterrows():
            lines.append(
                f"| {year} | {_fmt_pct(row['strategy'])} | {_fmt_pct(row['benchmark'])} | "
                f"{_fmt_pct(row['excess'])} | {_fmt_pct(row.get('top5_share'))} | "
                f"{_fmt_num(row.get('names_for_half'), 0)} |"
            )
        excess = yearly_excess["excess"].dropna()
        lines += [
            "",
            f"Years with positive excess: {int((excess > 0).sum())} of {len(excess)}. "
            f"Median excess: {_fmt_pct(excess.median())}.",
            "",
            "A positive mean with a negative median is the signature of a result",
            "that depends on a few years. A year of large excess with a high top-5",
            "share and few names for half cannot be promised to repeat.",
            "",
        ]

    best = attr.top_names(contributions, n=8)
    if not best.empty:
        lines += ["### Best and worst names, whole sample", "",
                  "| Ticker | Sector | Contribution | Periods | Average weight |",
                  "|---|---|---|---|---|"]
        for ticker, row in best.iterrows():
            lines.append(
                f"| {ticker} | {row['sector']} | {_fmt_pct(row['contribution'])} | "
                f"{int(row['periods_held'])} | {_fmt_pct(row['avg_weight'])} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def decay_section(by_lag: pd.DataFrame | None, by_horizon: pd.DataFrame | None) -> str:
    """Vida util de la senal: lo que decide la frecuencia de rebalanceo."""
    from .validation import rebalance_signal_cost

    if (by_lag is None or by_lag.empty) and (by_horizon is None or by_horizon.empty):
        return ""

    lines = ["## Signal decay", ""]
    if by_lag is not None and not by_lag.empty:
        lines += [
            "### IC with the signal lagged k months (next month's return)",
            "",
            "| Lag | Mean IC | t | p | Versus fresh |",
            "|---|---|---|---|---|",
        ]
        for lag, row in by_lag.iterrows():
            lines.append(
                f"| {lag} | {_fmt_num(row['ic_mean'], 4)} | {_fmt_num(row['t_stat'])} | "
                f"{_fmt_num(row['p_value'], 4)} | {_fmt_pct(row['vs_fresh'])} |"
            )
        lines += [
            "",
            "Signal kept by rebalance frequency:",
            "",
            f"- Monthly: {_fmt_pct(rebalance_signal_cost(by_lag, 1))}",
            f"- Every two months: {_fmt_pct(rebalance_signal_cost(by_lag, 2))}",
            f"- Quarterly: {_fmt_pct(rebalance_signal_cost(by_lag, 3))}",
            "",
            "Rebalancing less saves cost only if the signal is still alive. If",
            "quarterly keeps almost all the signal, monthly turnover is a toll",
            "with nothing in return.",
            "",
        ]
    if by_horizon is not None and not by_horizon.empty:
        lines += [
            "### IC by return horizon",
            "",
            "| Horizon (months) | Mean IC | t (Newey-West, lags >= h) | IC per month |",
            "|---|---|---|---|",
        ]
        for h, row in by_horizon.iterrows():
            lines.append(
                f"| {h} | {_fmt_num(row['ic_mean'], 4)} | {_fmt_num(row['t_stat'])} | "
                f"{_fmt_num(row['ic_per_month'], 4)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def build_report(
    result: BacktestResult,
    perf: Performance,
    cfg: Config,
    *,
    benchmark_nav: pd.Series | None = None,
    validation: Mapping[str, object] | None = None,
    universe_summary: Mapping[str, object] | None = None,
    contributions: pd.DataFrame | None = None,
    yearly_excess: pd.DataFrame | None = None,
    decay_lag: pd.DataFrame | None = None,
    decay_horizon: pd.DataFrame | None = None,
) -> str:
    """Reporte completo en Markdown."""
    title = cfg.get("meta.name")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    header = [
        f"# {title}",
        "",
        f"- Generated: {generated}",
        f"- Configuration: `{cfg.source.name if cfg.source else 'in memory'}` "
        f"(fingerprint `{cfg.fingerprint}`)",
        f"- Universe: {cfg.get('universe.exchanges')} | "
        f"rebalance {cfg.get('calendar.rebalance')} | "
        f"{cfg.get('portfolio.n_positions')} positions | "
        f"weighting `{cfg.get('portfolio.weighting')}`",
        f"- Factor weights: {cfg.factor_weights}",
        f"- Costs: {cfg.total_cost_bps:.0f} bps per side",
        "",
        "> Two results with the same fingerprint are comparable. With a",
        "> different fingerprint they are not: some investment decision",
        "> changed between one run and the other.",
        "",
    ]

    sections = ["\n".join(header)]

    if universe_summary:
        sections.append(
            "## Universe\n\n"
            f"- Names on the last date: {universe_summary.get('n_names', 'n/a')}\n"
            f"- Sectors represented: {universe_summary.get('n_sectors', 'n/a')}\n"
            f"- By country: {universe_summary.get('by_country', {})}\n"
        )

    sections.append(performance_section(perf, cfg))

    annual = yearly_returns(result.nav, benchmark_nav)
    if not annual.empty:
        sections.append("## Year by year return\n\n" + _table(annual, "{:.2%}"))

    sections.append(costs_section(result, perf))
    sections.append(holdings_section(result))

    if contributions is not None:
        sections.append(attribution_section(contributions, yearly_excess))

    if validation:
        sections.append(validation_section(validation))

    if decay_lag is not None or decay_horizon is not None:
        sections.append(decay_section(decay_lag, decay_horizon))

    if result.warnings:
        unique = list(dict.fromkeys(result.warnings))[:20]
        sections.append(
            "## Engine warnings\n\n"
            + "\n".join(f"- {w}" for w in unique)
            + ("\n\n_(showing the first 20)_\n" if len(result.warnings) > 20 else "\n")
        )

    sections.append(_LIMITATIONS)
    return "\n".join(sections)


def save_report(text: str, name: str = "backtest") -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = REPORT_DIR / f"{name}_{stamp}.md"
    path.write_text(text, encoding="utf-8")
    return path


def save_artifacts(result: BacktestResult, name: str = "backtest") -> dict[str, Path]:
    """Guarda NAV, operaciones, posiciones y rebalanceos para poder auditarlos.

    Un reporte sin los datos que lo sustentan obliga a creerselo. Con estos
    ficheros, cualquiera del equipo puede recalcular las metricas por su cuenta.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    paths: dict[str, Path] = {}

    nav_path = REPORT_DIR / f"{name}_{stamp}_nav.csv"
    result.nav.rename("nav").to_csv(nav_path, index_label="date")
    paths["nav"] = nav_path

    for label, frame in (("trades", result.trades), ("holdings", result.holdings),
                         ("rebalances", result.rebalances)):
        if frame is None or frame.empty:
            continue
        path = REPORT_DIR / f"{name}_{stamp}_{label}.csv"
        frame.to_csv(path, index=False)
        paths[label] = path

    return paths


def save_charts(result: BacktestResult, benchmark_nav: pd.Series | None = None,
                name: str = "backtest") -> Path | None:
    """Curva de NAV y drawdown. Opcional: requiere matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    from .metrics import drawdown_series

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = REPORT_DIR / f"{name}_{stamp}_curve.png"

    fig, (ax_nav, ax_dd) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax_nav.plot(result.nav.index, result.nav.to_numpy(), label="Strategy", linewidth=1.4)
    if benchmark_nav is not None and len(benchmark_nav):
        aligned = benchmark_nav.reindex(result.nav.index).ffill()
        ax_nav.plot(aligned.index, aligned.to_numpy(), label="Benchmark",
                    linewidth=1.1, alpha=0.75)
    ax_nav.set_yscale("log")
    ax_nav.set_ylabel("NAV (log scale)")
    ax_nav.legend(loc="upper left")
    ax_nav.grid(alpha=0.3)

    dd = drawdown_series(result.nav)
    ax_dd.fill_between(dd.index, dd.to_numpy() * 100, 0, alpha=0.4, color="crimson")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.grid(alpha=0.3)

    fig.suptitle("Equity curve and drawdown")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
