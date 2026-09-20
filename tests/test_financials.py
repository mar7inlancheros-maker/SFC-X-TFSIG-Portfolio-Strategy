"""Ratios financieras: los errores de denominador y de signo que cuestan dinero."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig import financials as fin


def _frame(**columns):
    n = max(len(v) if isinstance(v, (list, tuple)) else 1 for v in columns.values())
    data = {k: (list(v) if isinstance(v, (list, tuple)) else [v] * n) for k, v in columns.items()}
    return pd.DataFrame(data)


def test_roe_no_sale_positivo_con_patrimonio_negativo():
    """Perdidas y patrimonio negativo dan ROE positivo si no se controla.

    Sin esta proteccion, una empresa tecnicamente quebrada aparece entre las mas
    rentables del universo y el factor de calidad la compra.
    """
    df = _frame(net_income=[-50.0], equity=[-100.0])
    assert pd.isna(fin.quality_metrics(df)["roe"].iloc[0])


def test_division_por_cero_devuelve_nan_no_infinito():
    df = _frame(net_income=[100.0], equity=[0.0])
    assert pd.isna(fin.quality_metrics(df)["roe"].iloc[0])


def test_rendimiento_negativo_ordena_bien():
    """Una empresa en perdidas queda ABAJO, no fuera ni arriba."""
    df = _frame(net_income=[-100.0, 50.0], market_cap=[1000.0, 1000.0])
    yields = fin.value_metrics(df)["earnings_yield"]
    assert yields.iloc[0] < 0 < yields.iloc[1]


def test_capex_se_resta_aunque_venga_con_signo_positivo():
    """El XBRL publica el capex como pago positivo. Sumarlo invierte el ranking."""
    df = _frame(ocf=[100.0], capex=[30.0])
    assert fin.free_cash_flow(df).iloc[0] == pytest.approx(70.0)


def test_capex_negativo_tambien_se_resta():
    df = _frame(ocf=[100.0], capex=[-30.0])
    assert fin.free_cash_flow(df).iloc[0] == pytest.approx(70.0)


def test_sin_deuda_declarada_la_deuda_es_cero_no_faltante():
    """Una empresa sin deuda no reporta la etiqueta; no puede quedar fuera."""
    df = _frame(debt_long=[np.nan], debt_short=[100.0])
    assert fin.total_debt(df).iloc[0] == pytest.approx(100.0)


def test_sin_ninguna_pata_de_deuda_si_es_faltante():
    df = _frame(debt_long=[np.nan], debt_short=[np.nan])
    assert pd.isna(fin.total_debt(df).iloc[0])


def test_margen_bruto_se_deriva_si_no_viene_reportado():
    df = _frame(gross_profit=[np.nan], revenue=[1000.0], cogs=[600.0])
    assert fin.gross_profit(df).iloc[0] == pytest.approx(400.0)


def test_tipo_impositivo_absurdo_cae_al_de_referencia():
    # Impuesto negativo sobre beneficio positivo: credito fiscal puntual.
    df = _frame(tax_expense=[-50.0], pretax_income=[100.0])
    assert fin.effective_tax_rate(df).iloc[0] == pytest.approx(fin.DEFAULT_TAX_RATE)


def test_tipo_impositivo_razonable_se_respeta():
    df = _frame(tax_expense=[21.0], pretax_income=[100.0])
    assert fin.effective_tax_rate(df).iloc[0] == pytest.approx(0.21)


def test_sin_gasto_financiero_la_cobertura_es_virtud_no_hueco():
    df = _frame(operating_income=[100.0], interest_expense=[np.nan])
    cobertura = fin.quality_metrics(df)["interest_coverage"].iloc[0]
    assert cobertura == pytest.approx(50.0)
    assert not pd.isna(cobertura)


def test_capital_invertido_negativo_se_anula():
    df = _frame(equity=[-500.0], debt_long=[100.0], cash=[50.0])
    assert pd.isna(fin.invested_capital(df).iloc[0])


def test_roic_usa_nopat_y_no_beneficio_neto():
    df = _frame(operating_income=[100.0], tax_expense=[20.0], pretax_income=[80.0],
                equity=[400.0], debt_long=[100.0], cash=[0.0], net_income=[60.0])
    # NOPAT = 100 * (1 - 0.25) = 75; capital = 400 + 100 - 0 = 500 -> 15%
    assert fin.quality_metrics(df)["roic"].iloc[0] == pytest.approx(0.15)


def test_todas_las_metricas_tienen_direccion_declarada():
    """Sin direccion no se sabe si mas es mejor: es el error de signo caro."""
    df = _frame(revenue=[1000.0], net_income=[100.0], equity=[500.0], assets=[2000.0],
                market_cap=[5000.0], ocf=[150.0], capex=[50.0], cogs=[600.0],
                operating_income=[200.0], cash=[100.0], debt_long=[300.0],
                current_assets=[800.0], current_liabilities=[400.0],
                interest_expense=[20.0], tax_expense=[40.0], pretax_income=[180.0],
                shares_growth=[0.02])
    metricas = set(fin.value_metrics(df).columns) | set(fin.quality_metrics(df).columns)
    sin_direccion = metricas - set(fin.METRIC_DIRECTION)
    assert not sin_direccion, f"metricas sin direccion: {sin_direccion}"


def test_compute_all_no_pierde_columnas_originales():
    df = _frame(ticker=["AAA"], revenue=[1000.0], net_income=[100.0], market_cap=[5000.0])
    out = fin.compute_all(df)
    assert "ticker" in out.columns
    assert "earnings_yield" in out.columns
    assert out["earnings_yield"].iloc[0] == pytest.approx(0.02)
