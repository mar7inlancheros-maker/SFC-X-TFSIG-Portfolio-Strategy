"""Reporte de calidad de datos y compuerta que excluye lo inservible.

Umbrales, declarados:

| Estado | Condicion |
|---|---|
| FAIL | sin datos; o menos de 126 sesiones (6 meses); o mas de 5% de huecos |
| WARN | 1-5% de huecos; retornos extremos; cierre identico 5+ sesiones seguidas; historia que empieza 30+ dias despues de lo pedido; mas de 1% de datos rellenados |
| OK   | nada de lo anterior |

Un FAIL se EXCLUYE del analisis con aviso visible. Un WARN se usa, con aviso.
Nunca se usa un dato malo sin decirlo.
"""

from __future__ import annotations

import pandas as pd

from .cleaning import CleanData
from .loader import MarketData

MIN_SESSIONS = 126
FAIL_MISSING = 0.05
WARN_MISSING = 0.01
WARN_FILLED = 0.01
LATE_START_DAYS = 30


def quality_report(
    clean: CleanData | None,
    market: MarketData,
    tickers: list[str] | tuple[str, ...],
    requested_start: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        if ticker in market.failures or clean is None or ticker not in clean.stats:
            rows.append({"ticker": ticker, "start": None, "end": None, "sessions": 0,
                         "missing_pct": 1.0, "status": "FAIL",
                         "notes": market.failures.get(ticker, "sin datos")})
            continue
        s = clean.stats[ticker]
        notes: list[str] = []
        status = "OK"

        if s["active_sessions"] < MIN_SESSIONS:
            status = "FAIL"
            notes.append(f"solo {int(s['active_sessions'])} sesiones (< {MIN_SESSIONS})")
        if s["missing_pct"] > FAIL_MISSING:
            status = "FAIL"
            notes.append(f"{s['missing_pct']:.1%} de huecos")
        if status != "FAIL":
            warnings: list[str] = []
            if s["missing_pct"] > WARN_MISSING:
                warnings.append(f"{s['missing_pct']:.1%} de huecos")
            if s["extreme_returns"] > 0:
                warnings.append(f"{int(s['extreme_returns'])} retornos |r|>50%")
            if s["stale_run"] >= 5:
                warnings.append(f"cierre identico {int(s['stale_run'])} sesiones")
            if s["start"] is not None and (s["start"] - requested_start).days > LATE_START_DAYS:
                warnings.append(f"historia desde {s['start'].date()}")
            if s["active_sessions"] and s["filled"] / s["active_sessions"] > WARN_FILLED:
                warnings.append(f"{int(s['filled'])} dias rellenados")
            if s["non_positive"] > 0:
                warnings.append(f"{int(s['non_positive'])} precios <= 0 anulados")
            if warnings:
                status = "WARN"
                notes.extend(warnings)
            if s["duplicates"] > 0:
                # Informativo: los duplicados se resuelven sin perder datos.
                notes.append(f"{int(s['duplicates'])} duplicados eliminados")

        rows.append({
            "ticker": ticker,
            "start": s["start"],
            "end": s["end"],
            "sessions": int(s["active_sessions"]),
            "missing_pct": s["missing_pct"],
            "status": status,
            "notes": "; ".join(notes),
        })
    return pd.DataFrame(rows).set_index("ticker")


def failed(report: pd.DataFrame) -> list[str]:
    return list(report.index[report["status"] == "FAIL"])
