"""Ratios derivadas de los fundamentales point-in-time. Funciones puras.

Aqui no hay red, ni fechas, ni decisiones de inversion: entra un marco con los
conceptos crudos de la SEC y las variables de mercado de esa fecha, y salen las
ratios que alimentan los factores. Se prueba con datos sinteticos.

**Denominadores.** Media ratio financiera es una division por algo que puede ser
cero o negativo, y ahi es donde un modelo empieza a comprar basura sin que nadie
lo note:

- Un patrimonio negativo con beneficio negativo da un ROE POSITIVO. Una empresa
  quebrada aparece entre las mas rentables del universo. Se anula: ROE solo se
  calcula con patrimonio > 0.
- Un P/E negativo no ordena. Por eso el modelo usa rendimientos (beneficio sobre
  capitalizacion) y no multiplos: un rendimiento negativo ordena bien, y la
  empresa en perdidas queda abajo, que es donde debe estar.
- Dividir por cero en pandas da inf, no error. Un inf se cuela hasta el z-score
  y arrastra toda la seccion cruzada. Todo se pasa por `_safe_divide`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Tipo impositivo de referencia cuando la efectiva no se puede calcular o sale
# absurda (perdidas, creditos fiscales, cambios de ley). No se calibra: es un
# supuesto declarado, igual para todas las empresas y todas las fechas, para que
# no se convierta en un grado de libertad que se pueda ajustar a posteriori.
DEFAULT_TAX_RATE = 0.25


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    require_positive: bool = False,
) -> pd.Series:
    """Division que devuelve NaN en vez de inf o de un signo enganoso.

    `require_positive=True` exige denominador estrictamente positivo: es lo que
    impide el ROE positivo por patrimonio negativo.
    """
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    if require_positive:
        den = den.where(den > 0)
    else:
        den = den.where(den != 0)
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Columna o serie de NaN: un concepto ausente no puede tumbar el panel."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


# ---------------------------------------------------------------------------
#  Agregados intermedios
# ---------------------------------------------------------------------------


def total_debt(df: pd.DataFrame) -> pd.Series:
    """Deuda financiera total. Ausente = 0, no NaN.

    Una empresa sin deuda no reporta la etiqueta de deuda; tratar esa ausencia
    como dato faltante expulsaria del factor de calidad justo a las empresas sin
    apalancamiento, que es lo contrario de lo que se busca. Si faltan las dos
    patas, si es NaN: ahi si es falta de informacion.
    """
    long = _col(df, "debt_long")
    short = _col(df, "debt_short")
    both_missing = long.isna() & short.isna()
    combined = long.fillna(0.0) + short.fillna(0.0)
    return combined.where(~both_missing)


def gross_profit(df: pd.DataFrame) -> pd.Series:
    """Margen bruto reportado, o ingresos menos coste de ventas."""
    reported = _col(df, "gross_profit")
    derived = _col(df, "revenue") - _col(df, "cogs")
    return reported.fillna(derived)


def free_cash_flow(df: pd.DataFrame) -> pd.Series:
    """Caja operativa menos inversion en activo fijo.

    El capex en el XBRL viene como salida de caja con signo positivo (es un
    `Payments...`), asi que se RESTA. Sumarlo -- el error tipico -- convierte a
    las empresas mas intensivas en capital en las de mejor flujo libre.
    """
    return _col(df, "ocf") - _col(df, "capex").abs()


def effective_tax_rate(df: pd.DataFrame) -> pd.Series:
    """Tipo efectivo, acotado a [0, 50%]. Fuera de rango: tipo de referencia."""
    rate = _safe_divide(_col(df, "tax_expense"), _col(df, "pretax_income"))
    rate = rate.where((rate >= 0.0) & (rate <= 0.5))
    return rate.fillna(DEFAULT_TAX_RATE)


def invested_capital(df: pd.DataFrame) -> pd.Series:
    """Patrimonio + deuda - caja. El capital que el negocio tiene que rentabilizar."""
    capital = _col(df, "equity") + total_debt(df).fillna(0.0) - _col(df, "cash").fillna(0.0)
    return capital.where(capital > 0)


def enterprise_value(df: pd.DataFrame) -> pd.Series:
    """Capitalizacion + deuda - caja."""
    ev = _col(df, "market_cap") + total_debt(df).fillna(0.0) - _col(df, "cash").fillna(0.0)
    return ev.where(ev > 0)


# ---------------------------------------------------------------------------
#  Metricas de valor: todas como RENDIMIENTO (alto = barato)
# ---------------------------------------------------------------------------


def value_metrics(df: pd.DataFrame) -> pd.DataFrame:
    market_cap = _col(df, "market_cap").where(lambda s: s > 0)
    return pd.DataFrame(
        {
            "earnings_yield": _safe_divide(_col(df, "net_income"), market_cap),
            "fcf_yield": _safe_divide(free_cash_flow(df), market_cap),
            "book_to_price": _safe_divide(_col(df, "equity"), market_cap),
            "sales_to_price": _safe_divide(_col(df, "revenue"), market_cap),
            "ebit_to_ev": _safe_divide(_col(df, "operating_income"), enterprise_value(df)),
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
#  Metricas de calidad
#
#  Rentabilidad, solidez y prudencia contable -- las tres patas de Quality Minus
#  Junk (Asness, Frazzini y Pedersen, 2019), adaptadas a lo que el XBRL publico
#  permite calcular de forma fiable.
# ---------------------------------------------------------------------------


def quality_metrics(df: pd.DataFrame) -> pd.DataFrame:
    equity = _col(df, "equity")
    assets = _col(df, "assets").where(lambda s: s > 0)
    revenue = _col(df, "revenue").where(lambda s: s > 0)

    nopat = _col(df, "operating_income") * (1.0 - effective_tax_rate(df))

    # Cobertura de intereses: sin gasto financiero la cobertura es "infinita",
    # y eso es una virtud, no un dato faltante. Se acota a un valor alto finito
    # para que no distorsione la estandarizacion.
    interest = _col(df, "interest_expense").abs()
    coverage = _safe_divide(_col(df, "operating_income"), interest, require_positive=True)
    no_debt_cost = interest.isna() | (interest <= 0)
    coverage = coverage.where(~no_debt_cost, 50.0).clip(upper=50.0)

    return pd.DataFrame(
        {
            "roic": _safe_divide(nopat, invested_capital(df)),
            "roe": _safe_divide(_col(df, "net_income"), equity, require_positive=True),
            "gross_profitability": _safe_divide(gross_profit(df), assets),
            "fcf_margin": _safe_divide(free_cash_flow(df), revenue),
            "interest_coverage": coverage,
            "debt_to_equity": _safe_divide(total_debt(df), equity, require_positive=True),
            "current_ratio": _safe_divide(_col(df, "current_assets"), _col(df, "current_liabilities")),
            # Devengo: beneficio que no ha pasado por caja. Cuanto mas alto,
            # peor calidad contable (Sloan, 1996).
            "accruals": _safe_divide(_col(df, "net_income") - _col(df, "ocf"), assets),
            # Dilucion: crecimiento de acciones en circulacion a 12 meses. Lo
            # calcula `panel.py`, que es quien tiene la serie temporal.
            "shares_growth": _col(df, "shares_growth"),
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
#  Direccion de cada metrica: +1 = mas es mejor, -1 = menos es mejor
#
#  Esta tabla es la que impide el error de signo mas caro del modelo: ordenar
#  por deuda/patrimonio de mayor a menor y comprar las mas endeudadas.
# ---------------------------------------------------------------------------

METRIC_DIRECTION: dict[str, int] = {
    # valor: rendimiento alto = barato = mejor
    "earnings_yield": +1,
    "fcf_yield": +1,
    "book_to_price": +1,
    "sales_to_price": +1,
    "ebit_to_ev": +1,
    # calidad
    "roic": +1,
    "roe": +1,
    "gross_profitability": +1,
    "fcf_margin": +1,
    "interest_coverage": +1,
    "current_ratio": +1,
    "debt_to_equity": -1,     # mas apalancamiento, peor
    "accruals": -1,           # mas devengo, peor
    "shares_growth": -1,      # mas dilucion, peor
    # precio
    "momentum": +1,
    "volatility": -1,         # mas volatilidad, peor
}


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Panel crudo -> panel con todas las ratios agregadas."""
    out = df.copy()
    for block in (value_metrics(df), quality_metrics(df)):
        for column in block.columns:
            out[column] = block[column]
    out["total_debt"] = total_debt(df)
    out["free_cash_flow"] = free_cash_flow(df)
    out["enterprise_value"] = enterprise_value(df)
    return out
