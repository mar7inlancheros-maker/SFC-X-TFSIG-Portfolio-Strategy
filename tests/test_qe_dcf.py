"""DCF automatizado: identidades del modelo, supuestos, datos de la SEC y Compustat."""

from __future__ import annotations

import faulthandler
import gc
import sys

import numpy as np
import pandas as pd
import pytest

from quant_engine.valuation import assumptions as asm
from quant_engine.valuation import data as vdata
from quant_engine.valuation import dcf
from quant_engine.valuation.engine import value_company

CAP = dcf.Capital(price=100.0, shares=10.0, cash=50.0, debt=150.0, minority=10.0)


def _flat(years=10, margin=0.2, tax=0.25, wacc=0.09, g=0.0, ronic=None, s2c=np.inf):
    return dcf.Assumptions(growth=np.full(years, g), margin=np.full(years, margin), tax=np.full(years, tax),
                           da_pct=0.05, capex_pct=0.05, nwc_pct=0.0, terminal_growth=g, wacc=wacc,
                           ronic=wacc if ronic is None else ronic, sales_to_capital=s2c)


# ---------------------------------------------------------------------------
#  Identidades del motor
# ---------------------------------------------------------------------------


def test_sin_crecimiento_el_ev_es_una_perpetuidad_a_mitad_de_ano():
    a = _flat()
    out = dcf.value(1000.0, a, CAP)
    nopat = 1000.0 * 0.2 * 0.75
    assert out["ev"] == pytest.approx(nopat / 0.09 * 1.09 ** 0.5, rel=1e-9)
    assert out["equity"] == pytest.approx(out["ev"] + 50 - 150 - 10)
    assert out["per_share"] == pytest.approx(out["equity"] / 10)


def test_el_terminal_reinvierte_g_entre_ronic():
    a = _flat(g=0.03, wacc=0.09)
    out = dcf.value(1000.0, a, CAP)
    nopat_n = out["table"]["nopat"].iloc[-1]
    assert out["tv"] == pytest.approx(nopat_n * 1.03 * (1 - 0.03 / 0.09) / (0.09 - 0.03))
    # Con RONIC = WACC el crecimiento del terminal no crea valor: g no mueve el TV
    # respecto a crecer cero con el mismo NOPAT del ano N+1.
    no_value = nopat_n * 1.03 / 0.09
    assert out["tv"] == pytest.approx(no_value)


def test_crecer_exige_capital_con_el_suelo_de_ventas_sobre_capital():
    a = _flat(g=0.20, s2c=2.0)
    p = dcf.value(1000.0, a, CAP)["table"]
    # D&A = CapEx y circulante 0: toda la reinversion es el suelo, delta ventas / 2.
    assert p["growth_capital"].iloc[0] == pytest.approx(200.0 / 2)
    assert p["fcff"].iloc[0] == pytest.approx(p["nopat"].iloc[0] - 100.0)


def test_fade_va_del_inicio_al_terminal_en_linea_recta():
    f = dcf.fade(0.30, 0.03, 10)
    assert f[0] == pytest.approx(0.30) and f[-1] == pytest.approx(0.03)
    assert np.allclose(np.diff(f), (0.03 - 0.30) / 9)
    held = dcf.fade(0.30, 0.03, 10, hold=3)
    assert (held[:3] == 0.30).all() and held[-1] == pytest.approx(0.03)


def test_blume_y_rating_sintetico():
    assert dcf.blume(1.6) == pytest.approx(1.402)
    assert dcf.synthetic_rating(np.inf, 1e6)[0] == "AAA"
    assert dcf.synthetic_rating(3.2, 1e6)[0] == "A-"
    assert dcf.synthetic_rating(3.2, 1e3)[0] == "BB"          # una pequena necesita mas cobertura
    assert dcf.synthetic_rating(-1.0, 1e6)[0] == "D"


def test_wacc_a_pesos_de_mercado():
    c = dcf.cost_of_capital(rf=0.04, beta_raw=1.0, erp=0.05, marginal_tax=0.25, ebit=100, interest=13,
                            cap=CAP, market_cap_usd=1e6)
    we = 1000 / (1000 + 150)
    assert c["cost_of_equity"] == pytest.approx(0.04 + 1.0 * 0.05)
    assert c["rating"] == "AA" and c["cost_of_debt"] == pytest.approx(0.04 + 0.0070)
    assert c["wacc"] == pytest.approx(we * 0.09 + (1 - we) * 0.047 * 0.75)


def test_el_dcf_inverso_recupera_el_crecimiento_que_pone_el_precio():
    a = dcf.Assumptions(growth=dcf.fade(0.15, 0.025, 10), margin=np.full(10, 0.25), tax=np.full(10, 0.25),
                        da_pct=0.04, capex_pct=0.05, nwc_pct=0.1, terminal_growth=0.025, wacc=0.09, ronic=0.09)
    target = dcf.with_shift(a, growth_shift=0.10)
    price = dcf.value(1000.0, target, CAP)["per_share"]
    rev = dcf.reverse_dcf(1000.0, a, dcf.Capital(price=price, shares=10, cash=50, debt=150, minority=10))
    assert rev["solved"] and rev["growth_shift"] == pytest.approx(0.10, abs=1e-4)
    assert rev["implied_year1_growth"] == pytest.approx(0.25, abs=1e-4)


def test_monte_carlo_centrado_en_el_caso_base_y_sin_terminales_explosivos():
    a = _flat(g=0.02, wacc=0.08)
    base = dcf.value(1000.0, a, CAP)["per_share"]
    mc = dcf.monte_carlo(1000.0, a, CAP, growth_sd=0.0, margin_sd=0.0, wacc_sd=0.0, g_halfwidth=0.0, n=200)
    assert mc["p50"] == pytest.approx(base) and mc["p05"] == pytest.approx(base)
    wild = dcf.monte_carlo(1000.0, a, CAP, growth_sd=0.05, margin_sd=0.03, wacc_sd=0.03, n=2000, seed=1)
    assert np.isfinite(wild["values"]).all() and wild["p95"] < 50 * base


def test_sensibilidad_wacc_g_ordenada():
    a = _flat(g=0.02, wacc=0.09)
    grid = dcf.sensitivity(1000.0, a, CAP, {"wacc": [-0.01, 0.0, 0.01]}, {"g": [-0.005, 0.0, 0.005]})
    assert grid.loc[0.0, 0.0] == pytest.approx(dcf.value(1000.0, a, CAP)["per_share"])
    assert (grid.diff(axis=0).iloc[1:] < 0).all().all()      # mas WACC, menos valor


def test_comprobaciones_avisan_de_lo_que_un_comite_preguntaria():
    bad = dcf.value(1000.0, _flat(g=0.075, wacc=0.09), CAP)
    assert any("within 2 points" in f for f in bad["flags"])
    loss = dcf.value(1000.0, _flat(margin=-0.05), CAP)
    assert any("never earns money" in f for f in loss["flags"])


# ---------------------------------------------------------------------------
#  Supuestos automaticos
# ---------------------------------------------------------------------------


def _hist(revenues=(800, 900, 1000, 1100), margins=(0.15, 0.16, 0.18, 0.20)):
    idx = pd.date_range("2022-12-31", periods=len(revenues), freq="YE")
    rev = np.array(revenues, dtype=float)
    return pd.DataFrame({
        "revenue": rev, "ebit": rev * np.array(margins), "da": rev * 0.04, "capex": rev * 0.05, "sbc": rev * 0.02,
        "tax_expense": rev * np.array(margins) * 0.18, "pretax_income": rev * np.array(margins),
        "interest_expense": 5.0, "cash": 100.0, "debt": 200.0, "debt_current": 20.0, "minority": 0.0,
        "preferred": 0.0, "current_assets": rev * 0.4, "current_liabilities": rev * 0.25, "equity": 500.0,
        "shares_diluted": 50.0, "shares_basic": 48.0,
    }, index=idx)


def test_supuestos_salen_de_la_historia_y_lo_dicen():
    coc = {"rf": 0.04, "wacc": 0.09}
    built = asm.build(_hist(), coc, "USD")
    base = built["cases"]["base"]
    g_last, cagr = 0.10, (1100 / 800) ** (1 / 3) - 1
    assert base.growth[0] == pytest.approx((g_last + cagr) / 2)
    assert base.growth[-1] == pytest.approx(0.025)
    assert base.margin[-1] == pytest.approx(0.20)                 # persistencia del margen actual
    assert base.tax[0] == pytest.approx(0.18 + 0.07 / 10) and base.tax[-1] == pytest.approx(0.25)
    assert base.capex_pct == pytest.approx(0.05) and base.da_pct == pytest.approx(0.04)
    assert any("IBES" not in n and "3-year CAGR" in n for n in base.notes)
    bear, bull = built["cases"]["bear"], built["cases"]["bull"]
    assert bear.wacc > base.wacc > bull.wacc and bear.margin[-1] < base.margin[-1] < bull.margin[-1]


def test_consenso_manda_sobre_la_historia_y_el_override_sobre_todo():
    coc = {"rf": 0.04, "wacc": 0.09}
    with_consensus = asm.build(_hist(), coc, "USD", consensus=[0.30, 0.20])["cases"]["base"]
    assert with_consensus.growth[0] == pytest.approx(0.30) and with_consensus.growth[1] == pytest.approx(0.20)
    over = asm.build(_hist(), coc, "USD", overrides={"target_margin": 0.30, "terminal_growth": 0.02,
                                                      "growth": 0.12})["cases"]["base"]
    assert over.margin[-1] == pytest.approx(0.30) and over.growth[0] == pytest.approx(0.12)
    assert over.terminal_growth == pytest.approx(0.02)


def test_el_terminal_nunca_crece_mas_que_la_tasa_libre_de_riesgo():
    built = asm.build(_hist(), {"rf": 0.015, "wacc": 0.08}, "USD", {"terminal_growth": 0.03})
    assert built["cases"]["base"].terminal_growth == pytest.approx(0.015)


def test_empresa_con_perdidas_vuelve_a_su_mejor_margen_positivo():
    built = asm.build(_hist(margins=(0.10, 0.05, -0.02, -0.04)), {"rf": 0.04, "wacc": 0.09}, "USD")
    assert built["cases"]["base"].margin[-1] == pytest.approx(0.10)


def test_valor_de_empresa_sin_ventas_no_se_inventa():
    h = _hist()
    h.loc[h.index[-1], "revenue"] = 0.0
    out = value_company("X", h, "USD", 10.0, 1.0, 0.04)
    assert not out["available"] and "pre-revenue" in out["reason"]


def test_acciones_de_portada_solo_si_cuadran_con_las_diluidas():
    h = _hist()
    good = value_company("X", h, "USD", 30.0, 1.0, 0.04, shares_out=49.0)
    assert good["capital"].shares == pytest.approx(49.0 * 50 / 48)
    multi = value_company("X", h, "USD", 30.0, 1.0, 0.04, shares_out=20.0)   # una sola clase
    assert multi["capital"].shares == pytest.approx(50.0)


# ---------------------------------------------------------------------------
#  Datos: SEC XBRL, Compustat, consenso
# ---------------------------------------------------------------------------


def _fact(val, end, start=None, form="10-K", filed="2025-02-01"):
    e = {"val": val, "end": end, "form": form, "filed": filed, "fy": int(end[:4]), "fp": "FY"}
    if start:
        e["start"] = start
    return e


def _facts():
    flows = lambda vals: [_fact(v, f"{y}-12-31", f"{y}-01-01") for y, v in vals]  # noqa: E731
    points = lambda vals: [_fact(v, f"{y}-12-31") for y, v in vals]  # noqa: E731
    return {"facts": {
        "us-gaap": {
            # 2023 con la etiqueta vieja, 2024 con la nueva: se cosen por ejercicio.
            "Revenues": {"units": {"USD": flows([(2023, 1_000e6)])}},
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": flows([(2024, 1_200e6)])}},
            "OperatingIncomeLoss": {"units": {"USD": flows([(2023, 150e6), (2024, 200e6)])}},
            "DepreciationDepletionAndAmortization": {"units": {"USD": flows([(2023, 40e6), (2024, 45e6)])}},
            "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": flows([(2023, 50e6), (2024, 60e6)])}},
            "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": points([(2023, 90e6), (2024, 100e6)])}},
            "ShortTermInvestments": {"units": {"USD": points([(2024, 20e6)])}},
            "LongTermDebtNoncurrent": {"units": {"USD": points([(2023, 300e6), (2024, 280e6)])}},
            "LongTermDebtCurrent": {"units": {"USD": points([(2024, 30e6)])}},
            "CommercialPaper": {"units": {"USD": points([(2024, 10e6)])}},
            "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": flows([(2024, 55e6)])}},
            # Un trimestre no es un ejercicio: no entra.
            "OperatingIncomeLossQ": {"units": {"USD": [_fact(1, "2024-03-31", "2024-01-01", form="10-Q")]}},
        },
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            _fact(54e6, "2025-01-31", form="10-K", filed="2025-02-01")]}}},
    }}


def test_estados_anuales_de_la_sec_cosen_etiquetas_y_suman_la_deuda():
    df, ccy, notes = vdata.annual_from_sec(_facts())
    assert ccy == "USD" and list(df.index.year) == [2023, 2024]
    assert df.loc["2023-12-31", "revenue"] == pytest.approx(1000.0)
    assert df.loc["2024-12-31", "revenue"] == pytest.approx(1200.0)
    assert df.loc["2024-12-31", "cash"] == pytest.approx(120.0)                 # caja + inversiones a corto
    assert df.loc["2024-12-31", "debt"] == pytest.approx(280.0 + 30.0 + 10.0)   # largo + corriente + papel
    assert df.loc["2024-12-31", "shares_diluted"] == pytest.approx(55.0)
    assert any("interest expense" in n for n in notes)
    assert vdata.latest_shares_outstanding(_facts())[0] == pytest.approx(54.0)


def test_una_reexpresion_posterior_sustituye_a_la_cifra_original():
    facts = _facts()
    facts["facts"]["us-gaap"]["OperatingIncomeLoss"]["units"]["USD"].append(
        _fact(210e6, "2024-12-31", "2024-01-01", filed="2026-02-01"))
    df, _, _ = vdata.annual_from_sec(facts)
    assert df.loc["2024-12-31", "ebit"] == pytest.approx(210.0)


def test_compustat_anual():
    funda = pd.DataFrame({"gvkey": "1", "datadate": ["2023-12-31", "2024-12-31"], "curcd": "CAD",
                          "revt": [100.0, 120.0], "oiadp": [10.0, 15.0], "dp": [3.0, 4.0], "capx": [5.0, 6.0],
                          "dltt": [50.0, 40.0], "dlc": [5.0, None], "che": [20.0, 25.0], "cshfd": [10.0, 10.5]})
    df, ccy = vdata.annual_from_compustat(funda)
    assert ccy == "CAD" and df.loc["2024-12-31", "debt"] == pytest.approx(40.0)
    assert df.loc["2023-12-31", "debt"] == pytest.approx(55.0) and df["ebit"].iloc[-1] == 15.0


def test_consenso_solo_si_es_el_ejercicio_siguiente_y_en_la_misma_moneda():
    rows = pd.DataFrame({"ticker": "X", "statpers": ["2026-09-17", "2026-09-17"], "fpi": ["1", "2"],
                         "fpedats": ["2026-12-31", "2027-12-31"], "meanest": [1320.0, 1452.0], "curcode": "USD"})
    g = vdata.consensus_growth(rows, pd.Timestamp("2025-12-31"), 1200.0, "USD")
    assert g == pytest.approx([0.10, 0.10])
    assert vdata.consensus_growth(rows, pd.Timestamp("2024-12-31"), 1200.0, "USD") is None   # otro ejercicio
    assert vdata.consensus_growth(rows, pd.Timestamp("2025-12-31"), 1200.0, "CAD") is None   # otra moneda


def test_ventas_son_el_total_y_no_un_desglose():
    facts = _facts()
    # Un desglose (servicios) mayor en prioridad que el total no puede ganarle.
    facts["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"].append(
        _fact(50e6, "2023-12-31", "2023-01-01"))
    df, _, _ = vdata.annual_from_sec(facts)
    assert df.loc["2023-12-31", "revenue"] == pytest.approx(1000.0)


def test_una_accion_no_vale_menos_que_cero():
    poor = dcf.Capital(price=5.0, shares=10.0, cash=0.0, debt=1e6)
    out = dcf.value(1000.0, _flat(), poor)
    assert out["equity"] < 0 and out["per_share"] == 0.0
    assert any("option value" in f for f in out["flags"])


def test_un_aviso_critico_quita_la_opinion_del_dcf():
    from quant_engine.valuation.engine import critical_flags

    flags = ["terminal value is 85% of enterprise value: the price rests on years 11+",
             "thin operating margin (0.7%): the value swings"]
    assert critical_flags(flags) == [flags[1]]
    thin = _hist(margins=(0.006, 0.007, 0.006, 0.007))
    v = value_company("X", thin, "USD", 30.0, 1.0, 0.04)
    assert v["stance"] == "NOT DECISION-GRADE" and v["critical"]


def test_ventas_sobre_capital_de_la_propia_empresa_si_es_mayor_que_el_suelo():
    h = _hist()
    h["equity"] = 60.0                    # capital invertido 60 + 200 - 100 = 160 -> ~6,9x ventas
    built = asm.build(h, {"rf": 0.04, "wacc": 0.09}, "USD")
    s2c = built["cases"]["base"].sales_to_capital
    assert s2c == pytest.approx(np.median([900 / 160, 1000 / 160, 1100 / 160]))
    assert asm.build(_hist(), {"rf": 0.04, "wacc": 0.09}, "USD")["cases"]["base"].sales_to_capital == 5.0


def _excel():
    """Excel por COM si esta instalado; si no, el test se salta."""
    client = pytest.importorskip("win32com.client")
    try:
        app = client.DispatchEx("Excel.Application")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Excel no disponible por COM: {exc}")
    app.Visible = False
    app.DisplayAlerts = False
    return app


def test_el_excel_recalculado_da_lo_mismo_que_el_motor(tmp_path):
    # La hoja DCF repite con formulas lo que hace dcf.value_vector. Si una de
    # las dos cambia sin la otra, el analista ve en el Excel un valor distinto
    # del que puso el motor en el plan.
    from quant_engine.valuation import excel

    v = value_company("X", _hist(), "USD", 30.0, 1.1, 0.04, shares_out=49.0)
    path = excel.write(v, tmp_path / "x_dcf.xlsx")
    app = _excel()
    got = {}
    try:
        wb = app.Workbooks.Open(str(path.resolve()))
        try:
            for choice, case in enumerate(("bear", "base", "bull"), start=1):
                wb.Worksheets("Inputs").Range("B28").Value = choice
                app.CalculateFull()
                got[case] = (wb.Worksheets("DCF").Range("B30").Value, wb.Worksheets("DCF").Range("B37").Value)
        finally:
            wb.Close(SaveChanges=False)
            # Un proxy COM que se libera con Excel ya cerrado da RPC_E_DISCONNECTED.
            del wb
    finally:
        # Al cerrar, Excel corta el canal RPC y Windows lanza 0x80010108, que
        # pywin32 maneja; faulthandler lo imprime igual como "fatal". Es ruido.
        was_on = faulthandler.is_enabled()
        faulthandler.disable()
        try:
            app.Quit()
            del app
            gc.collect()
        finally:
            if was_on:
                faulthandler.enable(file=sys.__stderr__)
    for case, (ev, per_share) in got.items():
        res = v["scenarios"]["cases"][case]
        assert ev == pytest.approx(res["ev"], rel=1e-9), case
        assert per_share == pytest.approx(res["per_share"], rel=1e-9), case
