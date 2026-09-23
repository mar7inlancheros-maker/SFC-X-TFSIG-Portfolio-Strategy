"""Acuerdo entre la senal de research y la evidencia cuantitativa.

Se mide con el Quant Score ALINEADO al research:

    alineado = score        si research dice LONG
    alineado = -score       si research dice SHORT

Asi, un valor alto siempre significa "los datos apoyan lo que dice research",
sea largo o corto. Categorias, con los umbrales del YAML (unidades de z):

    alineado >= strong            STRONG AGREEMENT
    weak <= alineado < strong     WEAK AGREEMENT
    conflict < alineado < weak    NEUTRAL
    alineado <= conflict          SIGNAL CONFLICT

Esto es descripcion, no recomendacion. Un CONFLICT no dice que research se
equivoca: dice que su tesis va contra lo que muestran precio, riesgo y liquidez
hoy, y que merece una conversacion en el comite antes de poner capital.
"""

from __future__ import annotations

import pandas as pd

LABELS = ("STRONG AGREEMENT", "WEAK AGREEMENT", "NEUTRAL", "SIGNAL CONFLICT")


def classify(aligned: float, strong: float, weak: float, conflict: float) -> str:
    if pd.isna(aligned):
        return "NEUTRAL"
    if aligned >= strong:
        return "STRONG AGREEMENT"
    if aligned >= weak:
        return "WEAK AGREEMENT"
    if aligned <= conflict:
        return "SIGNAL CONFLICT"
    return "NEUTRAL"


def agreement_table(scores: pd.DataFrame, signs: dict[str, int], thresholds: dict[str, float]) -> pd.DataFrame:
    strong = float(thresholds.get("strong", 0.5))
    weak = float(thresholds.get("weak", 0.15))
    conflict = float(thresholds.get("conflict", -0.15))
    rows = []
    for ticker, sign in signs.items():
        if ticker not in scores.index:
            continue
        score = float(scores.loc[ticker, "quant_score"])
        aligned = score * sign
        rows.append({
            "ticker": ticker,
            "research": "LONG" if sign > 0 else "SHORT",
            "quant_score": score,
            "rank": scores.loc[ticker, "rank"],
            "aligned_score": aligned,
            "agreement": classify(aligned, strong, weak, conflict),
        })
    return pd.DataFrame(rows).set_index("ticker").sort_values("quant_score", ascending=False)


def agreement_summary(table: pd.DataFrame) -> dict[str, int]:
    counts = table["agreement"].value_counts()
    return {label: int(counts.get(label, 0)) for label in LABELS}
