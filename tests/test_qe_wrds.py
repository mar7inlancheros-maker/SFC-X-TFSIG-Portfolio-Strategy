"""Datos de WRDS para el motor: calculos puros, sin red."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine.data import wrds_data as wd
from quant_engine.factors import factor_model


def _fundq(n=6, rdq_lag=30):
    ends = pd.date_range("2025-03-31", periods=n, freq="QE")
    fq = ((ends.month - 1) // 3 + 1)
    return pd.DataFrame({
        "gvkey": "001", "datadate": ends, "rdq": ends + pd.Timedelta(days=rdq_lag),
        "fyearq": ends.year, "fqtr": fq,
        "saleq": 100.0, "cogsq": 60.0, "niq": 10.0, "oiadpq": 15.0, "xintq": 1.0,
        # acumulados del ano fiscal: 20 por trimestre
        "oancfy": 20.0 * fq, "capxy": 5.0 * fq,
        "ceqq": 200.0, "atq": 500.0, "ltq": 300.0, "dlttq": 50.0, "dlcq": 10.0, "cheq": 30.0,
        "actq": 120.0, "lctq": 80.0, "cshoq": np.linspace(100, 105, n),
    })


def test_trimestre_desde_acumulado_del_ano_fiscal():
    f = _fundq()
    q = wd.quarterly_from_ytd(f, "oancfy")
    assert (q == 20.0).all()


def test_ttm_suma_cuatro_trimestres_y_la_caja_sale_de_los_acumulados():
    ttm = wd.ttm_fundamentals(_fundq(), pd.Timestamp("2026-12-31"))
    assert ttm["revenue"] == pytest.approx(400.0)
    assert ttm["ocf"] == pytest.approx(80.0)
    assert ttm["capex"] == pytest.approx(20.0)


def test_point_in_time_ignora_trimestres_aun_no_publicados():
    f = _fundq(rdq_lag=40)
    # El cierre de 2026-06-30 se publica el 2026-08-09: el 2026-08-01 no existe.
    ttm = wd.ttm_fundamentals(f, pd.Timestamp("2026-08-01"))
    assert ttm["fiscal_period_end"] == pd.Timestamp("2026-03-31")


def test_sin_cuatro_trimestres_publicados_no_hay_ttm():
    assert wd.ttm_fundamentals(_fundq(n=3), pd.Timestamp("2026-12-31")) is None


def test_valores_anulables_de_wrds_no_revientan():
    f = _fundq().astype({"xintq": "Float64"})
    f.loc[f.index[-1], "xintq"] = pd.NA
    assert wd.ttm_fundamentals(f, pd.Timestamp("2026-12-31")) is not None


def test_revision_de_eps_compara_el_mismo_ejercicio_y_se_acota():
    eps = pd.DataFrame({
        "ticker": "X", "fpedats": ["2026-12-31"] * 2,
        "statpers": ["2026-05-15", "2026-09-15"], "meanest": [0.05, 1.00],
        "stdev": [0.1, 0.1], "numest": [10, 10], "numup": [5, 5], "numdown": [1, 1],
    })
    recs = pd.DataFrame({"ticker": "X", "statpers": ["2026-05-15", "2026-09-15"],
                         "meanrec": [2.5, 2.0], "numrec": [10, 12]})
    out = wd.analyst_signals(recs, eps, pd.Timestamp("2026-09-20"))
    assert out["eps_revision_3m"] == pytest.approx(1.0)  # +1900% acotado a +100%
    assert out["rec_change_3m"] == pytest.approx(-0.5)   # mejora de recomendacion


def test_revision_no_mezcla_ejercicios_fiscales():
    eps = pd.DataFrame({
        "ticker": "X", "fpedats": ["2026-06-30", "2027-06-30"],
        "statpers": ["2026-05-15", "2026-09-15"], "meanest": [1.0, 2.0],
        "stdev": [0.1, 0.1], "numest": [10, 10], "numup": [0, 0], "numdown": [0, 0],
    })
    out = wd.analyst_signals(pd.DataFrame(columns=["ticker", "statpers", "meanrec", "numrec"]),
                             eps, pd.Timestamp("2026-09-20"))
    assert np.isnan(out["eps_revision_3m"])


def test_emision_principal_prefiere_activa_01_y_norteamerica():
    secs = pd.DataFrame({
        "tic": ["X", "X", "X"], "gvkey": ["1", "1", "1"], "iid": ["90", "02", "01"],
        "excntry": ["GBR", "USA", "USA"], "secstat": ["A", "A", "A"], "ibtic": ["a", "b", "c"],
    })
    assert wd.pick_primary_security(secs).loc["X", "iid"] == "01"


def test_interes_corto_en_porcentaje_y_dias_para_cubrir():
    si = pd.DataFrame({"shortint": [5e6]}, index=["X"])
    out = wd.short_interest_metrics(si, pd.Series({"X": 50.0}), pd.Series({"X": 1e6}))
    assert out.loc["X", "si_pct_float"] == pytest.approx(0.10)
    assert out.loc["X", "days_to_cover"] == pytest.approx(5.0)


def test_gics_se_traduce_al_esquema_de_sectores():
    d = wd.WrdsData(available=True, ids=pd.DataFrame({"gsector": ["45", "40"]}, index=["AAPL", "RY"]))
    assert d.sectors == {"AAPL": "Technology", "RY": "Financials"}


def test_fama_french_usa_su_propia_rf_para_el_exceso():
    idx = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(1)
    ff = pd.DataFrame(rng.normal(0, 0.01, (300, 6)), index=idx,
                      columns=["mktrf", "smb", "hml", "rmw", "cma", "umd"])
    ff["rf"] = 0.0002
    r = pd.DataFrame({"A": ff["rf"] + 1.2 * ff["mktrf"] + 0.5 * ff["hml"]})
    res = factor_model.fama_french_exposures(r, ["A"], ff)
    assert res.exposures.loc["A", "market"] == pytest.approx(1.2, abs=1e-6)
    assert res.exposures.loc["A", "value"] == pytest.approx(0.5, abs=1e-6)
    assert res.source.startswith("Fama-French")


def test_sin_wrds_el_motor_no_falla(monkeypatch):
    import sfc_tfsig.data.wrds as conn_mod

    def boom():
        raise RuntimeError("WRDS_USERNAME no esta definido")
    monkeypatch.setattr(conn_mod, "get_connection", boom)
    d = wd.load(["AAPL"], pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01"))
    assert not d.available
    assert "WRDS_USERNAME" in d.reason
