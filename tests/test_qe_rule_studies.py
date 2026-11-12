"""La tabla del estudio de los consejos de entrada (tools/qe_rule_studies.py timing)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import qe_rule_studies as studies  # noqa: E402

from quant_engine.factors import technical  # noqa: E402


def _entries(effect: dict[tuple[int, str], float], months=36, per_month=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2023-01-31", periods=months, freq="ME"):
        for _ in range(per_month):
            for side in (1, -1):
                f = {k: bool(rng.random() < 0.3) for k in ("extended", "against_trend", "with_trend")}
                r = rng.normal(0.05, 1.0) + sum(effect.get((side, k), 0.0) for k, v in f.items() if v)
                rows.append({"date": d, "side": side, "r": r, **f})
    return pd.DataFrame(rows)


def test_la_tabla_sostiene_contradice_o_no_ve_nada():
    df = _entries({(1, "extended"): -0.4, (-1, "with_trend"): -0.4})
    res = studies.timing_table(df).set_index(["side", "flag"])
    assert res.loc[(1, "extended"), "verdict"] == "SUPPORTED"
    assert res.loc[(-1, "with_trend"), "verdict"] == "CONTRADICTED"      # la ficha afirma lo contrario
    assert res.loc[(1, "against_trend"), "verdict"] == "NOT SUPPORTED"
    assert (res["p_holm"] >= res["p"] - 1e-12).all()


def test_meses_con_pocas_entradas_no_cuentan():
    df = _entries({}, per_month=8)                     # ~2-3 entradas con la condicion por mes
    res = studies.timing_table(df)
    assert (res["months"] == 0).all() and (res["verdict"] == "NOT SUPPORTED").all()


def test_la_ficha_cita_una_cifra_por_cada_afirmacion_del_estudio():
    claims = {(side, flag) for side, flag, _ in studies.TIMING_CLAIMS + studies.LEVEL_CLAIMS}
    assert set(technical.TIMING_EVIDENCE) == claims
    for (side, flag), e in technical.TIMING_EVIDENCE.items():
        assert e["verdict"] in ("SUPPORTED", "CONTRADICTED", "NOT SUPPORTED")
        note = technical.evidence_note(side, flag)
        assert note.isascii() and f"t {e['t']:+.1f}" in note
