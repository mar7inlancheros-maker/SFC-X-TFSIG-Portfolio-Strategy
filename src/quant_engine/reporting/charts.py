"""Los diez graficos del reporte. Solo presentan datos ya calculados.

Estilo sobrio a proposito: un color por serie, sin rellenos decorativos, ejes
con unidades. Un grafico de comite tiene que leerse en diez segundos.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from ..analysis import AnalysisResult  # noqa: E402

LONG_COLOR, SHORT_COLOR, BENCH_COLOR, PORT_COLOR = "#2f6db5", "#c0392b", "#7f7f7f", "#1b1b1b"


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _nav(r: AnalysisResult):
    bt = r.primary["backtest"]
    nav = bt.nav / bt.nav.iloc[0]
    bench = (1.0 + r.clean.returns[r.settings.benchmark].reindex(nav.index).fillna(0.0)).cumprod()
    return nav, bench


def render_all(r: AnalysisResult, directory: Path, stamp: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    nav, bench = _nav(r)
    r_eval = r.clean.returns.loc[r.clean.returns.index >= r.eval_start]

    # 1. Retorno acumulado
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nav.index, nav - 1, color=PORT_COLOR, label=f"L/S {r.primary['method']}")
    ax.plot(bench.index, bench - 1, color=BENCH_COLOR, label=r.settings.benchmark)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Cumulative return (Mode A, hypothetical)")
    ax.legend()
    ax.grid(alpha=0.3)
    out.append(_save(fig, directory / f"01_cumulative_{stamp}.png"))

    # 2. Drawdown
    fig, ax = plt.subplots(figsize=(10, 4))
    for series, label, color in ((nav, "L/S", PORT_COLOR), (bench, r.settings.benchmark, BENCH_COLOR)):
        dd = series / series.cummax() - 1
        ax.plot(dd.index, dd, color=color, label=label)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Drawdown")
    ax.legend()
    ax.grid(alpha=0.3)
    out.append(_save(fig, directory / f"02_drawdown_{stamp}.png"))

    # 3. Volatilidad movil
    fig, ax = plt.subplots(figsize=(10, 4))
    port_ret = nav.pct_change()
    ax.plot((port_ret.rolling(63).std() * np.sqrt(252)).dropna(), color=PORT_COLOR, label="L/S")
    ax.plot((r_eval[r.settings.benchmark].rolling(63).std() * np.sqrt(252)).dropna(), color=BENCH_COLOR,
            label=r.settings.benchmark)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Rolling 63d annualized volatility")
    ax.legend()
    ax.grid(alpha=0.3)
    out.append(_save(fig, directory / f"03_rolling_vol_{stamp}.png"))

    # 4. Beta movil
    fig, ax = plt.subplots(figsize=(10, 4))
    rb = r.rolling_betas.loc[r.rolling_betas.index >= r.eval_start]
    for t in rb.columns:
        ax.plot(rb.index, rb[t], color=LONG_COLOR if t in r.longs else SHORT_COLOR, alpha=0.6, lw=1)
    ax.axhline(1.0, color=BENCH_COLOR, lw=0.8, ls="--")
    ax.set_title("Rolling 252d beta (blue = LONG, red = SHORT)")
    ax.grid(alpha=0.3)
    out.append(_save(fig, directory / f"04_rolling_beta_{stamp}.png"))

    # 5. Mapa de correlaciones
    corr = r.correlation["pearson"]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation (daily returns)")
    out.append(_save(fig, directory / f"05_correlation_{stamp}.png"))

    # 6. Pesos
    w = r.primary["weights"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(w.index, w.values, color=[LONG_COLOR if v > 0 else SHORT_COLOR for v in w.values])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"Portfolio weights ({r.primary['method']})")
    out.append(_save(fig, directory / f"06_weights_{stamp}.png"))

    # 7. Cesta larga frente a corta
    fig, ax = plt.subplots(figsize=(10, 5))
    lb = (1 + r_eval[r.longs].mean(axis=1).fillna(0)).cumprod() - 1
    sb = (1 + r_eval[r.shorts].mean(axis=1).fillna(0)).cumprod() - 1
    ax.plot(lb.index, lb, color=LONG_COLOR, label="LONG basket (EW)")
    ax.plot(sb.index, sb, color=SHORT_COLOR, label="SHORT basket (EW, as held long)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("LONG vs SHORT basket cumulative return")
    ax.legend()
    ax.grid(alpha=0.3)
    out.append(_save(fig, directory / f"07_long_vs_short_{stamp}.png"))

    # 8. Contribucion al riesgo
    rc = r.primary["ex_ante"]["risk_contribution"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(rc.index, rc.values, color=[LONG_COLOR if t in r.longs else SHORT_COLOR for t in rc.index])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Risk contribution (negative = hedging)")
    out.append(_save(fig, directory / f"08_risk_contribution_{stamp}.png"))

    # 9. Quant score
    qs = r.agreement["quant_score"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(qs.index, qs.values,
            color=[LONG_COLOR if r.agreement.loc[t, "research"] == "LONG" else SHORT_COLOR for t in qs.index])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Quant score (color = research signal)")
    out.append(_save(fig, directory / f"09_quant_score_{stamp}.png"))

    # 10. Exposiciones a factores
    exp = r.factors.exposures
    if not exp.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(exp.astype(float), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("Factor exposures (ETF-spread factors)")
        out.append(_save(fig, directory / f"10_factor_exposures_{stamp}.png"))
    return out
