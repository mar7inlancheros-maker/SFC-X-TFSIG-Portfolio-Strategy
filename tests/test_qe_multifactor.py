"""El score validado del modelo multifactor como evidencia del plan long/short."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine import trade_plan as tp
from quant_engine.data import multifactor as mf

REPORT = """
### Coeficiente de informacion (IC de rangos, mensual)
| Metrica | Valor |
|---|---|
| IC medio | 0.0241 |
| t (Newey-West) | 3.00 |
| valor p | 0.0027 |
| Meses evaluados | 177 |
Spread Q1-Q5: 0.48% mensual, t = 1.88, p = 0.0606.
"""


def test_la_evidencia_se_lee_del_reporte_y_no_se_escribe_a_mano():
    ev = mf.evidence_from_report(REPORT, "backtest_x.md")
    assert ev["ic"] == 0.0241 and ev["ic_t"] == 3.0 and ev["months"] == 177
    assert ev["spread"] == 0.48 and ev["spread_t"] == 1.88 and ev["validated"]
    weak = mf.evidence_from_report(REPORT.replace("| 3.00 |", "| 1.20 |"))
    assert not weak["validated"]
    assert mf.evidence_from_report("sin tabla") is None


def _panel():
    date = pd.Timestamp("2026-09-30")
    tickers = [f"T{i}" for i in range(95)] + ["ARES", "RY", "BRK-B", "AAPL", "LOW"]
    score = np.r_[np.linspace(-1, 1, 95), -2.0, 0.5, 0.0, 3.0, np.nan]
    old = pd.DataFrame({"date": pd.Timestamp("2026-08-31"), "ticker": ["ARES"], "score_composite": [5.0]})
    new = pd.DataFrame({"date": date, "ticker": tickers, "score_composite": score, "score_value": -0.9,
                        "score_quality": 0.1, "score_momentum": -0.9, "score_lowvol": -1.1})
    return pd.concat([old, new], ignore_index=True)


def test_percentil_en_la_ultima_fecha_y_tickers_de_yahoo():
    out = mf.scores_from_panel(_panel(), ["ARES", "AAPL", "RY.TO", "BRK-B", "ZZZ", "LOW"], pd.Timestamp("2026-10-30"))
    assert out["available"] and out["universe"] == 99 and not out["stale"]   # LOW sin compuesto no cuenta
    assert out["rows"]["ARES"]["percentile"] == pytest.approx(1 / 99)         # el peor de la ultima fecha
    assert out["rows"]["AAPL"]["percentile"] == pytest.approx(1.0)
    assert out["rows"]["RY.TO"]["panel_ticker"] == "RY"                       # Toronto -> listado de EE.UU.
    assert out["rows"]["ARES"]["momentum"] == pytest.approx(-0.9)
    assert set(out["missing"]) == {"ZZZ", "LOW"}
    late = mf.scores_from_panel(_panel(), ["ARES"], pd.Timestamp("2026-12-31"))
    assert late["stale"]


@pytest.mark.parametrize("direction, pct, expected", [
    (-1, 0.05, 1), (-1, 0.95, -1), (1, 0.95, 1), (1, 0.05, -1), (1, 0.5, 0), (0, 0.05, 0), (1, np.nan, 0),
])
def test_acuerdo_solo_en_los_quintiles_extremos(direction, pct, expected):
    assert mf.agreement(direction, pct) == expected


def test_solo_cuenta_si_esta_al_dia_y_validado():
    base = {"available": True, "stale": False, "evidence": {"validated": True}}
    assert mf.usable(base)
    assert not mf.usable({**base, "stale": True})
    assert not mf.usable({**base, "evidence": {"validated": False}})
    assert not mf.usable({**base, "evidence": None}) and not mf.usable(None)


def test_el_multifactor_mueve_la_conviccion_un_escalon():
    assert tp.conviction("CONFIRM", 0.3, "none", 0.0, False) == "MEDIUM"
    assert tp.conviction("CONFIRM", 0.3, "none", 0.0, False, mf_agree=1) == "HIGH"
    assert tp.conviction("CONFIRM", 0.3, "none", 0.0, False, mf_agree=-1) == "LOW"
    assert tp.conviction("FLIP", -0.9, "none", 0.0, True, mf_agree=-1) == "LOW"   # no baja de LOW


def test_solo_los_reportes_oficiales_con_fecha_cuentan(tmp_path):
    (tmp_path / "backtest_20261004_1705.md").write_text(REPORT, encoding="utf-8")
    (tmp_path / "backtest_counterfactual_v2.md").write_text(REPORT.replace("0.0241", "0.9"), encoding="utf-8")
    (tmp_path / "backtest_20261030_1029_trades.md").write_text("x", encoding="utf-8")
    ev = mf.latest_evidence(tmp_path)
    assert ev["report"] == "backtest_20261004_1705.md" and ev["ic"] == 0.0241
