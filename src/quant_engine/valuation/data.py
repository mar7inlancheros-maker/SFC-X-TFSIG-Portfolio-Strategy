"""Estados financieros anuales para el DCF: Compustat (WRDS) o SEC XBRL.

Salida comun: una fila por ejercicio fiscal (indice = fecha de cierre), del
mas antiguo al mas reciente, en MILLONES de la moneda de reporte:

    revenue, ebit, da, capex, sbc, tax_expense, pretax_income, interest_expense,
    cash, debt, debt_current, minority, preferred, current_assets,
    current_liabilities, equity, shares_diluted, shares_basic

Orden de preferencia: Compustat si WRDS esta disponible (normalizado, cubre
Canada); si no, SEC XBRL (gratis, EE.UU. y extranjeros que presentan 20-F o
40-F). Para valorar se usan las cifras MAS RECIENTES de cada ejercicio
(re-expresiones incluidas): aqui no hay backtest y lo correcto es lo ultimo
publicado.

Decisiones que cambian el numero:

- **Deuda = financiera + arrendamientos financieros.** Los arrendamientos
  operativos de US GAAP no: su coste ya esta dentro del EBIT, y restar el
  pasivo seria contarlo dos veces. En IFRS 16 todos los arrendamientos son
  financieros y entran.
- **Caja = efectivo + inversiones a corto.**
- **EBIT = resultado operativo.** Si la empresa no lo publica (gestoras,
  brokers), beneficio antes de impuestos + gasto financiero, y se avisa.
- **Ventas = la MAYOR de las etiquetas de ventas de cada ejercicio**, no la
  primera por prioridad. Gold.com publica "Revenues" = 25.513 M (el total) y
  "RevenueFromContractWithCustomer..." = 618 M (un desglose): con prioridad,
  sus ventas caian de 9.699 a 177 de un ano a otro y el margen salia en 27%.
  Un desglose nunca supera al total.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A", "10-KT"}
FIELDS = ("revenue", "ebit", "da", "capex", "sbc", "tax_expense", "pretax_income", "interest_expense",
          "cash", "debt", "debt_current", "minority", "preferred", "current_assets", "current_liabilities",
          "equity", "shares_diluted", "shares_basic")

# Etiquetas por concepto, de la mas especifica a la mas general. Se elige la
# primera disponible EN CADA EJERCICIO, como en `sfc_tfsig.data.sec`.
FLOW_TAGS: dict[str, list[tuple[str, str]]] = {
    "revenue": [("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
                ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
                ("us-gaap", "Revenues"), ("us-gaap", "SalesRevenueNet"), ("us-gaap", "SalesRevenueGoodsNet"),
                ("ifrs-full", "Revenue"), ("ifrs-full", "RevenueFromContractsWithCustomers")],
    "ebit": [("us-gaap", "OperatingIncomeLoss"), ("ifrs-full", "ProfitLossFromOperatingActivities")],
    "da": [("us-gaap", "DepreciationDepletionAndAmortization"), ("us-gaap", "DepreciationAndAmortization"),
           ("us-gaap", "DepreciationAmortizationAndAccretionNet"), ("us-gaap", "Depreciation"),
           ("ifrs-full", "DepreciationAndAmortisationExpense"),
           ("ifrs-full", "DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss")],
    "capex": [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
              ("us-gaap", "PaymentsToAcquireProductiveAssets"),
              ("ifrs-full", "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities")],
    "sbc": [("us-gaap", "ShareBasedCompensation"), ("us-gaap", "AllocatedShareBasedCompensationExpense")],
    "tax_expense": [("us-gaap", "IncomeTaxExpenseBenefit"), ("ifrs-full", "IncomeTaxExpenseContinuingOperations")],
    "pretax_income": [
        ("us-gaap", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"),
        ("us-gaap", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
        ("ifrs-full", "ProfitLossBeforeTax")],
    "interest_expense": [("us-gaap", "InterestExpense"), ("us-gaap", "InterestExpenseNonoperating"),
                         ("us-gaap", "InterestExpenseDebt"), ("us-gaap", "InterestAndDebtExpense"),
                         ("ifrs-full", "FinanceCosts"), ("ifrs-full", "InterestExpense")],
    "shares_diluted": [("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
                       ("ifrs-full", "AdjustedWeightedAverageShares")],
    "shares_basic": [("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
                     ("ifrs-full", "WeightedAverageShares")],
}
POINT_TAGS: dict[str, list[tuple[str, str]]] = {
    "cash_only": [("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
                  ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
                  ("ifrs-full", "CashAndCashEquivalents")],
    "st_investments": [("us-gaap", "ShortTermInvestments"), ("us-gaap", "MarketableSecuritiesCurrent"),
                       ("us-gaap", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"),
                       ("ifrs-full", "CurrentInvestments")],
    "debt_noncurrent": [("us-gaap", "LongTermDebtNoncurrent"),
                        ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
                        ("us-gaap", "LongTermLineOfCredit"),
                        ("ifrs-full", "NoncurrentPortionOfNoncurrentBorrowings"),
                        ("ifrs-full", "LongtermBorrowings")],
    "debt_total_lt": [("us-gaap", "LongTermDebt")],
    "debt_current_all": [("us-gaap", "DebtCurrent"),
                         ("ifrs-full", "CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings")],
    "ltd_current": [("us-gaap", "LongTermDebtCurrent"), ("us-gaap", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
                    ("ifrs-full", "CurrentPortionOfLongtermBorrowings")],
    "st_borrowings": [("us-gaap", "ShortTermBorrowings"), ("us-gaap", "CommercialPaper"),
                      ("us-gaap", "LineOfCredit"), ("ifrs-full", "ShorttermBorrowings")],
    "finance_leases": [("us-gaap", "FinanceLeaseLiability"), ("ifrs-full", "LeaseLiabilities")],
    "finance_leases_parts": [("us-gaap", "FinanceLeaseLiabilityNoncurrent"),
                             ("us-gaap", "FinanceLeaseLiabilityCurrent")],
    "minority": [("us-gaap", "MinorityInterest"), ("ifrs-full", "NoncontrollingInterests")],
    "preferred": [("us-gaap", "PreferredStockValue")],
    "current_assets": [("us-gaap", "AssetsCurrent"), ("ifrs-full", "CurrentAssets")],
    "current_liabilities": [("us-gaap", "LiabilitiesCurrent"), ("ifrs-full", "CurrentLiabilities")],
    "equity": [("us-gaap", "StockholdersEquity"), ("ifrs-full", "EquityAttributableToOwnersOfParent")],
}


def _entries(facts: Mapping, taxonomy: str, tag: str) -> list[tuple[str, dict]]:
    node = facts.get("facts", {}).get(taxonomy, {}).get(tag)
    if not node:
        return []
    return [(unit, e) for unit, rows in node.get("units", {}).items() for e in rows]


def reporting_currency(facts: Mapping) -> str | None:
    """Moneda de las ventas en los formularios anuales."""
    for taxonomy, tag in FLOW_TAGS["revenue"]:
        units = [u for u, e in _entries(facts, taxonomy, tag) if e.get("form") in ANNUAL_FORMS]
        if units:
            return max(set(units), key=units.count)
    return None


def _annual_flow(facts: Mapping, tags: list[tuple[str, str]], unit: str) -> pd.Series:
    """Valor anual por fecha de cierre: primera etiqueta disponible en cada ejercicio."""
    by_end: dict[pd.Timestamp, tuple[int, str, float]] = {}
    for rank, (taxonomy, tag) in enumerate(tags):
        for u, e in _entries(facts, taxonomy, tag):
            if u != unit or e.get("form") not in ANNUAL_FORMS or "start" not in e:
                continue
            start, end = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
            if not 330 <= (end - start).days <= 400:
                continue
            key = (rank, e.get("filed", ""))
            old = by_end.get(end)
            # Mejor etiqueta; con la misma, la presentacion mas reciente (re-expresiones).
            if old is None or (rank, ) < (old[0], ) or (rank == old[0] and key[1] > old[1]):
                by_end[end] = (rank, key[1], float(e["val"]))
    return pd.Series({k: v[2] for k, v in by_end.items()}, dtype=float).sort_index()


def _point(facts: Mapping, tags: list[tuple[str, str]], unit: str, dates) -> pd.Series:
    """Saldo en cada fecha de cierre: primera etiqueta con dato en esa fecha."""
    out = {}
    for d in dates:
        best = None
        for rank, (taxonomy, tag) in enumerate(tags):
            for u, e in _entries(facts, taxonomy, tag):
                if u != unit or "start" in e or pd.Timestamp(e["end"]) != d:
                    continue
                if e.get("form") not in ANNUAL_FORMS and not str(e.get("form", "")).startswith("10-Q"):
                    continue
                cand = (rank, e.get("filed", ""), float(e["val"]))
                if best is None or cand[0] < best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                    best = cand
            if best is not None and best[0] == rank:
                break
        out[d] = best[2] if best else np.nan
    return pd.Series(out, dtype=float)


def _sum_parts(facts, tags, unit, dates) -> pd.Series:
    """Suma de etiquetas que son partes (no alternativas) de un mismo saldo."""
    parts = [_point(facts, [t], unit, dates) for t in tags]
    frame = pd.concat(parts, axis=1)
    return frame.sum(axis=1, min_count=1)


def annual_from_sec(facts: Mapping, max_years: int = 8) -> tuple[pd.DataFrame, str | None, list[str]]:
    """Estados anuales desde el JSON `companyfacts` de EDGAR. Funcion pura."""
    currency = reporting_currency(facts)
    notes: list[str] = []
    if currency is None:
        return pd.DataFrame(columns=list(FIELDS)), None, ["no annual revenue in SEC XBRL"]
    flows = {k: _annual_flow(facts, tags, "shares" if k.startswith("shares") else currency)
             for k, tags in FLOW_TAGS.items() if k != "revenue"}
    per_tag = [x for x in (_annual_flow(facts, [tag], currency) for tag in FLOW_TAGS["revenue"]) if len(x)]
    revenue = pd.concat(per_tag, axis=1).max(axis=1) if per_tag else pd.Series(dtype=float)
    revenue.index = pd.DatetimeIndex(revenue.index)
    flows["revenue"] = revenue.sort_index()
    dates = flows["revenue"].index[-max_years:]
    df = pd.DataFrame({k: v.reindex(dates) for k, v in flows.items()}, index=dates)
    if df["ebit"].isna().all() and df["pretax_income"].notna().any():
        df["ebit"] = df["pretax_income"] + df["interest_expense"].fillna(0.0)
        notes.append("no operating income reported: EBIT = pre-tax income + interest expense")
    pt = {k: _point(facts, tags, currency, dates) for k, tags in POINT_TAGS.items()
          if k != "finance_leases_parts"}
    pt["finance_leases_parts"] = _sum_parts(facts, POINT_TAGS["finance_leases_parts"], currency, dates)
    df["cash"] = pt["cash_only"].add(pt["st_investments"].fillna(0.0))
    current = pt["debt_current_all"].combine_first(pt["ltd_current"].add(pt["st_borrowings"], fill_value=0.0))
    long_term = pt["debt_noncurrent"].combine_first(pt["debt_total_lt"] - pt["ltd_current"].fillna(0.0))
    leases = pt["finance_leases"].combine_first(pt["finance_leases_parts"])
    df["debt"] = long_term.fillna(0.0) + current.fillna(0.0) + leases.fillna(0.0)
    df["debt_current"] = current
    for k in ("minority", "preferred", "current_assets", "current_liabilities", "equity"):
        df[k] = pt[k]
    for k in ("da", "capex", "sbc", "interest_expense", "minority", "preferred"):
        if df[k].isna().all():
            notes.append(f"{k.replace('_', ' ')} not found in XBRL: taken as zero")
    for k in FIELDS:
        if k not in df:
            df[k] = np.nan
    money = [k for k in FIELDS if not k.startswith("shares")]
    df[money] = df[money] / 1e6
    df[["shares_diluted", "shares_basic"]] = df[["shares_diluted", "shares_basic"]] / 1e6
    df = df[list(FIELDS)].dropna(subset=["revenue"])
    df.index.name = "fiscal_end"
    return df, currency, notes


def latest_shares_outstanding(facts: Mapping) -> tuple[float, pd.Timestamp | None]:
    """Acciones en circulacion de la portada mas reciente (millones)."""
    rows = [e for u, e in _entries(facts, "dei", "EntityCommonStockSharesOutstanding") if u == "shares"]
    if not rows:
        return np.nan, None
    last = max(rows, key=lambda e: (e["end"], e.get("filed", "")))
    return float(last["val"]) / 1e6, pd.Timestamp(last["end"])


# Compustat anual (comp.funda), ya en millones.
FUNDA_MAP = {"revenue": "revt", "ebit": "oiadp", "da": "dp", "capex": "capx", "sbc": "stkco",
             "tax_expense": "txt", "pretax_income": "pi", "interest_expense": "xint", "cash": "che",
             "minority": "mib", "preferred": "pstk", "current_assets": "act", "current_liabilities": "lct",
             "equity": "ceq", "shares_diluted": "cshfd", "shares_basic": "csho"}


def annual_from_compustat(funda: pd.DataFrame, max_years: int = 8) -> tuple[pd.DataFrame, str | None]:
    """Estados anuales desde `comp.funda` de una empresa. Funcion pura."""
    if funda is None or funda.empty:
        return pd.DataFrame(columns=list(FIELDS)), None
    f = funda.copy()
    f["datadate"] = pd.to_datetime(f["datadate"])
    f = f.sort_values("datadate").drop_duplicates("datadate", keep="last").iloc[-max_years:]
    num = lambda c: pd.to_numeric(f[c], errors="coerce") if c in f else pd.Series(np.nan, index=f.index)  # noqa: E731
    df = pd.DataFrame({k: num(c).to_numpy() for k, c in FUNDA_MAP.items()}, index=pd.DatetimeIndex(f["datadate"]))
    dlc, dltt = num("dlc").to_numpy(), num("dltt").to_numpy()
    df["debt"] = np.nan_to_num(dltt) + np.nan_to_num(dlc)
    df["debt_current"] = dlc
    df.index.name = "fiscal_end"
    currency = str(f["curcd"].iloc[-1]) if "curcd" in f and pd.notna(f["curcd"].iloc[-1]) else "USD"
    return df[list(FIELDS)].dropna(subset=["revenue"]), currency


def consensus_growth(rows: pd.DataFrame, last_fiscal_end: pd.Timestamp, last_revenue: float,
                     currency: str) -> list[float] | None:
    """Crecimiento de ventas que espera el consenso de IBES para FY1 (y FY2).

    Solo si FY1 cierra unos doce meses despues del ultimo ejercicio publicado,
    esta en la misma moneda y el crecimiento es creible (< +150%): un consenso
    de otro ejercicio o en otra moneda daria un crecimiento inventado.
    """
    if rows is None or rows.empty or not last_revenue or last_revenue <= 0:
        return None
    r = rows.copy()
    r["fpedats"] = pd.to_datetime(r["fpedats"])
    r["statpers"] = pd.to_datetime(r["statpers"])
    r = r.sort_values("statpers").drop_duplicates("fpi", keep="last").set_index("fpi")
    fy1 = r.loc["1"] if "1" in r.index else None
    if fy1 is None or str(fy1.get("curcode", currency)) != currency:
        return None
    gap = (fy1["fpedats"] - last_fiscal_end).days
    if not 300 <= gap <= 430:
        return None
    g1 = float(fy1["meanest"]) / last_revenue - 1
    out = [g1] if -0.5 < g1 < 1.5 else None
    if out and "2" in r.index and float(r.loc["2", "meanest"]) > 0:
        g2 = float(r.loc["2", "meanest"]) / float(fy1["meanest"]) - 1
        if -0.5 < g2 < 1.5:
            out.append(g2)
    return out
