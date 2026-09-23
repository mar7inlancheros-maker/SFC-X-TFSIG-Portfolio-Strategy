"""Exportacion: texto, JSON, CSV y HTML con los graficos enlazados.

El JSON lleva el fingerprint de la configuracion: dos reportes con el mismo
fingerprint salen de los mismos parametros. Sin eso, un reporte no se puede
reproducir y no deberia discutirse en un comite.
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from ..analysis import AnalysisResult


def _clean(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) or math.isinf(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Series):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        return {str(k): _clean(v) for k, v in value.to_dict(orient="index").items()}
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items() if not isinstance(v, (np.ndarray,))}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def to_json(r: AnalysisResult) -> dict:
    s = r.settings
    return _clean({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "fingerprint": s.fingerprint,
        "settings": {"longs": r.longs, "shorts": r.shorts, "benchmark": s.benchmark, "lookback": s.lookback,
                     "construction": s.construction, "rebalance": s.rebalance, "capital": s.initial_capital,
                     "transaction_cost": s.transaction_cost, "borrow_cost": s.borrow_cost,
                     "risk_free_rate": s.risk_free_rate},
        "evaluation_window": [r.eval_start, r.as_of],
        "data_quality": r.quality,
        "excluded": r.excluded,
        "sectors": r.sectors,
        "performance": r.trailing,
        "risk": r.risk,
        "regression": r.regression,
        "momentum": r.momentum,
        "mean_reversion": r.mean_reversion,
        "volatility": r.volatility,
        "factor_exposures": r.factors.exposures,
        "factor_unavailable": r.factors.unavailable,
        "quant_scores": r.scores,
        "agreement": r.agreement,
        "rank_stability": r.rank_stability,
        "long_vs_short": r.long_vs_short,
        "spread_test": r.spread,
        "portfolios": {m: {"status": b.get("status"), "note": b.get("note"), "weights": b.get("weights"),
                           "summary": b.get("summary"),
                           "ex_ante": {k: v for k, v in (b.get("ex_ante") or {}).items()}}
                       for m, b in r.portfolios.items()},
        "beta_comparison": r.beta_comparison,
        "stress": {k: v for k, v in r.stress.items()},
        "montecarlo": {k: v for k, v in r.montecarlo.items() if k != "bootstrap_terminal"},
        "robustness": {k: v for k, v in r.robustness.items()} if r.robustness else {},
        "warnings": r.warnings,
        "disclaimer": "Descriptive quantitative analysis. Not investment advice.",
    })


def export_all(r: AnalysisResult, console: Console, directory: Path, stamp: str,
               charts: list[Path] | None = None) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["txt"] = directory / f"report_{stamp}.txt"
    paths["txt"].write_text(console.export_text(clear=False), encoding="utf-8")

    paths["json"] = directory / f"report_{stamp}.json"
    paths["json"].write_text(json.dumps(to_json(r), indent=2, default=str), encoding="utf-8")

    w = pd.DataFrame({m: b["weights"] for m, b in r.portfolios.items() if b.get("status") == "ok"})
    w["research"] = [("LONG" if t in r.longs else "SHORT") for t in w.index]
    w["sector"] = [r.sectors.get(t, "Unknown") for t in w.index]
    paths["portfolio"] = directory / f"portfolio_{stamp}.csv"
    w.to_csv(paths["portfolio"], index_label="ticker")

    paths["signals"] = directory / f"signals_{stamp}.csv"
    r.agreement.join(r.scores.drop(columns=["quant_score", "rank"])).to_csv(paths["signals"], index_label="ticker")

    paths["risk"] = directory / f"risk_{stamp}.csv"
    r.risk.join(r.regression, rsuffix="_reg").join(r.volatility, rsuffix="_vol").to_csv(paths["risk"], index_label="ticker")

    paths["correlation"] = directory / f"correlation_{stamp}.csv"
    r.correlation["pearson"].to_csv(paths["correlation"])

    body = console.export_html(clear=False, inline_styles=True)
    if charts:
        imgs = "\n".join(f'<h3>{html.escape(p.stem)}</h3><img src="charts/{html.escape(p.name)}" '
                         'style="max-width:900px;width:100%">' for p in charts)
        body = body.replace("</body>", f"<hr><h2>Charts</h2>{imgs}</body>")
    paths["html"] = directory / f"report_{stamp}.html"
    paths["html"].write_text(body, encoding="utf-8")
    return paths
