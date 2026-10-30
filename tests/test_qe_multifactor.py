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


def _synthetic_panel(n=80, seed=3):
    """Ultima fecha de un panel con las columnas que usa el multifactor, ya con ratios."""
    from sfc_tfsig import financials

    rng = np.random.default_rng(seed)
    rev = rng.uniform(500, 5000, n) * 1e6
    df = pd.DataFrame({
        "date": pd.Timestamp("2026-09-30"), "ticker": [f"N{i}" for i in range(n)],
        "sector": np.where(np.arange(n) % 2, "Technology", "Industrials"),
        "revenue": rev, "cogs": rev * rng.uniform(0.3, 0.7, n), "operating_income": rev * rng.uniform(0.05, 0.3, n),
        "net_income": rev * rng.uniform(0.02, 0.2, n), "ocf": rev * rng.uniform(0.05, 0.25, n),
        "capex": rev * 0.04, "interest_expense": rev * 0.01, "equity": rev * rng.uniform(0.3, 1.5, n),
        "assets": rev * 2.0, "liabilities": rev * 1.0, "debt_long": rev * 0.3, "debt_short": rev * 0.05,
        "cash": rev * 0.1, "current_assets": rev * 0.6, "current_liabilities": rev * 0.4,
        "shares": rng.uniform(50, 500, n) * 1e6, "price": rng.uniform(10, 300, n),
        "momentum": rng.normal(0.1, 0.3, n), "volatility": rng.uniform(0.15, 0.6, n), "shares_growth": 0.0,
    })
    df["market_cap"] = df["price"] * df["shares"]
    return financials.compute_all(df)


def test_un_nombre_de_fuera_cae_donde_cae_uno_identico_de_dentro():
    from sfc_tfsig.config import load_config
    from sfc_tfsig.factors.composite import build_scores

    cfg = load_config(None)
    panel = build_scores(_synthetic_panel(), cfg)
    raw_cols = [c for c in panel.columns if not c.startswith("score_")]
    twin = panel.iloc[[7]][raw_cols].assign(ticker="TWIN")
    out = mf.score_outside(panel, twin, cfg)
    inside = panel.dropna(subset=["score_composite"])
    me = inside.loc[7, "score_composite"]
    expected = (inside["score_composite"] < me).mean() + 0.5 * (inside["score_composite"] == me).mean()
    # Una fila mas mueve un poco la media y la desviacion de la seccion
    # cruzada: casi igual, no identico.
    assert out.loc["TWIN", "score_composite"] == pytest.approx(me, abs=0.02)
    assert out.loc["TWIN", "percentile"] == pytest.approx(expected, abs=2 / len(inside))


def test_razones_de_exclusion_del_universo():
    from sfc_tfsig.config import load_config

    cfg = load_config(None)
    row = pd.Series({"price": 3.0, "dollar_volume": 1e9, "market_cap": 2e9, "history_months": 120})
    assert mf.exclusion_reasons(row, cfg) == ["price 3.00 USD below the 5.00 minimum"]
    small = pd.Series({"price": 20.0, "dollar_volume": 1e6, "market_cap": 1e8, "history_months": 6})
    assert len(mf.exclusion_reasons(small, cfg)) == 3


def test_valor_relativo_frente_al_dcf():
    from quant_engine import playbook as pb

    m = {"available": True, "rows": {"A": {"value": 1.2}}, "outside": {"B": {"value": -0.9}}}
    cheap, dear = mf.relative_value(m, "A"), mf.relative_value(m, "B")
    assert cheap["label"] == "CHEAP" and cheap["in_universe"]
    assert dear["label"] == "EXPENSIVE" and not dear["in_universe"]
    assert mf.relative_value(m, "C") is None
    assert "agrees with the DCF" in pb.relative_check("UNDERVALUED", cheap)
    assert "disagrees with the DCF" in pb.relative_check("UNDERVALUED", dear)
    assert "outside its universe" in pb.relative_check("FAIR VALUE", dear)
