"""El score del modelo multifactor (sfc_tfsig) como evidencia para el plan.

El Quant Score del motor long/short no ha demostrado que prediga (IC ~0 en su
validacion). El compuesto del modelo multifactor si: en su backtest de 2012 a
2026, sobre ~1.200 empresas al mes, IC de rangos 0,024 con t de Newey-West 3,0
y quintiles casi perfectamente ordenados. Este modulo lee su ultimo panel
puntuado (de disco, sin red) y dice en que percentil esta cada nombre.

La evidencia NO se escribe a mano: se lee del ultimo reporte de backtest
(`output/reports/backtest_*.md`). Sin reporte, o con un IC sin significancia,
el percentil se muestra pero no mueve la conviccion ni los FLIP.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

FACTOR_COLUMNS = ("score_value", "score_quality", "score_momentum", "score_lowvol")
STALE_DAYS = 45            # el panel es mensual: con mas de mes y medio, se avisa
MIN_EVIDENCE_T = 2.0       # t del IC del multifactor para que cuente como evidencia
EXTREME_QUINTILE = 0.20    # Q1 (>= 0,80 de percentil) o Q5 (<= 0,20)


def evidence_from_report(text: str, name: str = "") -> dict | None:
    """IC, su t y el spread de quintiles, leidos de un reporte de backtest."""
    def grab(pattern: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    ic = grab(r"\| IC medio \| (-?[\d.]+) \|")
    t = grab(r"\| t \(Newey-West\) \| (-?[\d.]+) \|")
    if ic is None or t is None:
        return None
    return {"ic": ic, "ic_t": t, "months": grab(r"\| Meses evaluados \| (\d+) \|"),
            "spread": grab(r"Spread Q1-Q5: (-?[\d.]+)% mensual"), "spread_t": grab(r"mensual, t = (-?[\d.]+)"),
            "report": name, "validated": t >= MIN_EVIDENCE_T}


def latest_evidence(report_dir: Path) -> dict | None:
    # Solo reportes oficiales con fecha (backtest_AAAAMMDD_HHMM.md): un
    # contrafactual o una copia con otro nombre no puede colarse como evidencia.
    reports = sorted(p for p in report_dir.glob("backtest_*.md")
                     if re.fullmatch(r"backtest_\d{8}_\d{4}\.md", p.name))
    if not reports:
        return None
    return evidence_from_report(reports[-1].read_text(encoding="utf-8"), reports[-1].name)


def _panel_ticker(ticker: str, available: set[str]) -> str | None:
    """Ticker del panel (estilo SEC) para un ticker de Yahoo. Toronto: el listado de EE.UU."""
    t = ticker.upper()
    if t in available:
        return t
    if "." in t:
        base = t[: t.rindex(".")]
        for cand in (base, base.replace(".", "-")):
            if cand in available:
                return cand
    return None


def scores_from_panel(panel: pd.DataFrame, tickers: list[str], as_of: pd.Timestamp) -> dict:
    """Percentil del compuesto del multifactor de cada ticker en la ultima fecha. Funcion pura."""
    if panel is None or panel.empty or "score_composite" not in panel:
        return {"available": False, "reason": "no scored multifactor panel: run python main.py panel"}
    date = pd.Timestamp(panel["date"].max())
    last = panel[panel["date"] == date].dropna(subset=["score_composite"])
    pct = last.set_index("ticker")["score_composite"].rank(pct=True)
    factors = last.set_index("ticker")[[c for c in FACTOR_COLUMNS if c in last]]
    rows, missing = {}, {}
    available = set(pct.index)
    for t in tickers:
        key = _panel_ticker(t, available)
        if key is None:
            missing[t] = ("not in the multifactor universe on that date (it filters by listing, liquidity, size "
                          "and requires value and quality data)")
            continue
        composite = float(last.set_index("ticker").loc[key, "score_composite"])
        rows[t] = {"panel_ticker": key, "percentile": float(pct[key]), "composite": composite,
                   **{c.replace("score_", ""): float(factors.loc[key, c]) for c in factors.columns
                      if pd.notna(factors.loc[key, c])}}
    return {"available": True, "date": date, "universe": len(last), "rows": rows, "missing": missing,
            "stale": (as_of - date).days > STALE_DAYS}


def load(tickers: list[str], as_of: pd.Timestamp) -> dict:
    """Panel del multifactor de disco y evidencia del ultimo reporte. Nunca lanza."""
    try:
        from sfc_tfsig import cli as mf_cli  # noqa: PLC0415
        from sfc_tfsig.config import load_config  # noqa: PLC0415
        from sfc_tfsig.data import cache  # noqa: PLC0415
        from sfc_tfsig.paths import REPORT_DIR  # noqa: PLC0415

        panel = cache.read_frame(mf_cli._panel_cache_name(load_config(None)))  # noqa: SLF001
        out = scores_from_panel(panel, tickers, as_of)
        out["evidence"] = latest_evidence(REPORT_DIR)
        return out
    except Exception as exc:  # noqa: BLE001 -- el plan sigue sin el multifactor
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def agreement(direction: int, percentile: float | None) -> int:
    """+1 si el multifactor respalda la direccion en su quintil extremo, -1 si la
    contradice ahi, 0 en medio o sin dato."""
    if percentile is None or not np.isfinite(percentile) or direction == 0:
        return 0
    if percentile >= 1 - EXTREME_QUINTILE:
        return 1 if direction > 0 else -1
    if percentile <= EXTREME_QUINTILE:
        return 1 if direction < 0 else -1
    return 0


def usable(mf: dict | None) -> bool:
    """El multifactor cuenta como evidencia: panel al dia y validado en su reporte."""
    return bool(mf and mf.get("available") and not mf.get("stale")
                and (mf.get("evidence") or {}).get("validated"))
