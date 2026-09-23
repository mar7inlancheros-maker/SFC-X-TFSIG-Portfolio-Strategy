"""Fundamentales point-in-time desde el XBRL de la SEC.

Este es el modulo del que depende que el backtest sea evidencia y no ficcion.

**El problema que resuelve.** Un balance con fecha de cierre 31-dic-2023 no fue
publico el 31-dic-2023: se presento en febrero o marzo de 2024. Un modelo que
compre en enero de 2024 usando ese dato esta usando informacion que nadie tenia,
y el backtest sale precioso por una razon que no se puede repetir con dinero
real. Aqui cada dato viaja con su fecha de presentacion (`filed`) y el panel solo
puede ver lo que ya estaba presentado (`filed <= fecha de rebalanceo`).

**Como primera publicacion, no reexpresado.** Cuando un periodo aparece varias
veces (la original y las reexpresiones posteriores), se conserva la de `filed`
mas temprano: es el numero que el mercado tuvo delante ese dia. Usar la
reexpresion es la otra mitad del mismo sesgo.

**TTM reconstruido.** Los flujos (ingresos, beneficio, caja operativa) se
publican unas veces por trimestre y otras acumulados en el año fiscal. Se
normalizan a trimestres -- diferenciando los acumulados que comparten fecha de
inicio -- y se suman en ventanas de cuatro. Si no hay cuatro trimestres limpios,
se cae al anual. Si tampoco, el dato falta: falta, no se inventa.

**Canada.** Los emisores canadienses inscritos en la SEC (40-F, 20-F, o 10-K si
son domesticos a efectos de la ley) presentan bajo la taxonomia IFRS. Por eso
cada concepto tiene candidatos en `us-gaap` Y en `ifrs-full`. Los exclusivos de
TSX no estan en EDGAR en absoluto: ver README, "Limitaciones declaradas".
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import requests

from ..paths import ROOT
from . import cache

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# La SEC publica un limite de 10 peticiones/segundo. Se pide por debajo a
# proposito: el coste de ir un 20% mas lento es nada, el de que bloqueen la IP
# del club es que nadie del equipo puede trabajar.
_MIN_INTERVAL_S = 0.125
_last_request_ts = 0.0

_TIMEOUT_S = 30
_MAX_RETRIES = 3


class SECError(RuntimeError):
    """Fallo de red, de credenciales o de formato al hablar con EDGAR."""


# ---------------------------------------------------------------------------
#  Identificacion (la SEC la exige)
# ---------------------------------------------------------------------------


def _read_env_file(path: Path) -> dict[str, str]:
    """Lector minimo de `.env`. utf-8-sig porque PowerShell mete BOM."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def user_agent() -> str:
    """User-Agent con nombre y correo reales, como exige la SEC."""
    agent = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if not agent:
        agent = _read_env_file(ROOT / ".env").get("SEC_EDGAR_USER_AGENT", "").strip()
    if not agent or "@" not in agent:
        raise SECError(
            "falta SEC_EDGAR_USER_AGENT con un correo real.\n"
            "  1. copia .env.example a .env\n"
            "  2. pon: SEC_EDGAR_USER_AGENT=Tu Nombre tu@correo.com\n"
            "La SEC responde 403 a quien no se identifica, y bloquea por IP a "
            "quien insiste con un agente generico."
        )
    return agent


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent(),
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


def _get_json(session: requests.Session, url: str) -> Any:
    """GET con limitacion de ritmo y reintentos con espera creciente."""
    global _last_request_ts

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        elapsed = time.monotonic() - _last_request_ts
        if elapsed < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - elapsed)
        _last_request_ts = time.monotonic()
        try:
            response = session.get(url, timeout=_TIMEOUT_S)
            if response.status_code == 404:
                return None  # la empresa no tiene XBRL; no es un error de red
            if response.status_code == 403:
                raise SECError(
                    f"403 de la SEC para {url}. Revisa SEC_EDGAR_USER_AGENT; "
                    "si es correcto, la IP puede estar limitada temporalmente."
                )
            if response.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                last_error = SECError("429: demasiadas peticiones")
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(1.0 * (attempt + 1))
    raise SECError(f"no se pudo descargar {url}: {last_error}")


# ---------------------------------------------------------------------------
#  Conceptos: un nombre economico -> muchas etiquetas XBRL posibles
#
#  El orden importa: para CADA PERIODO se toma la etiqueta disponible que este
#  mas arriba en la lista. Las de arriba son las mas especificas y modernas; las
#  de abajo, las historicas o las generales. Elegir periodo a periodo -- y no una
#  etiqueta unica para toda la empresa -- es lo que cose la historia a traves de
#  la transicion contable de ASC 606 en 2018.
# ---------------------------------------------------------------------------

_USD = ("USD",)
_SHARES = ("shares",)

CONCEPTS: dict[str, dict[str, Any]] = {
    # -- flujos (cuenta de resultados y flujo de caja) --------------------
    "revenue": {
        "kind": "flow",
        "units": _USD,
        "tags": [
            ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
            ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
            ("us-gaap", "Revenues"),
            ("us-gaap", "SalesRevenueNet"),
            ("us-gaap", "SalesRevenueGoodsNet"),
            ("ifrs-full", "Revenue"),
            ("ifrs-full", "RevenueFromContractsWithCustomers"),
        ],
    },
    "cogs": {
        "kind": "flow",
        "units": _USD,
        "tags": [
            ("us-gaap", "CostOfGoodsAndServicesSold"),
            ("us-gaap", "CostOfRevenue"),
            ("us-gaap", "CostOfGoodsSold"),
            ("ifrs-full", "CostOfSales"),
        ],
    },
    "gross_profit": {
        "kind": "flow",
        "units": _USD,
        "tags": [("us-gaap", "GrossProfit"), ("ifrs-full", "GrossProfit")],
    },
    "operating_income": {
        "kind": "flow",
        "units": _USD,
        "tags": [
            ("us-gaap", "OperatingIncomeLoss"),
            ("ifrs-full", "ProfitLossFromOperatingActivities"),
        ],
    },
    "net_income": {
        "kind": "flow",
        "units": _USD,
        "tags": [
            ("us-gaap", "NetIncomeLoss"),
            ("us-gaap", "ProfitLoss"),
            ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic"),
            ("ifrs-full", "ProfitLoss"),
            ("ifrs-full", "ProfitLossAttributableToOwnersOfParent"),
        ],
    },
    "interest_expense": {
        "kind": "flow",
        "units": _USD,
        "tags": [
            ("us-gaap", "InterestExpense"),
            ("us-gaap", "InterestExpenseDebt"),
            ("us-gaap", "InterestIncomeExpenseNet"),
            ("ifrs-full", "FinanceCosts"),
        ],
    },
    "tax_expense": {
        "kind": "flow",
        "units": _USD,
        "tags": [
            ("us-gaap", "IncomeTaxExpenseBenefit"),
            ("ifrs-full", "IncomeTaxExpenseContinuingOperations"),
        ],
    },
    "pretax_income": {
        "kind": "flow",
        "units": _USD,
        "tags": [
            ("us-gaap", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"),
            ("us-gaap", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
            ("ifrs-full", "ProfitLossBeforeTax"),
        ],
    },
    "ocf": {
        "kind": "flow",
        "units": _USD,
        "tags": [
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
            ("ifrs-full", "CashFlowsFromUsedInOperatingActivities"),
        ],
    },
    "capex": {
        "kind": "flow",
        "units": _USD,
        "tags": [
            ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
            ("us-gaap", "PaymentsToAcquireProductiveAssets"),
            ("ifrs-full", "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"),
        ],
    },
    "dividends_paid": {
        "kind": "flow",
        "units": _USD,
        "tags": [
            ("us-gaap", "PaymentsOfDividendsCommonStock"),
            ("us-gaap", "PaymentsOfDividends"),
            ("ifrs-full", "DividendsPaidClassifiedAsFinancingActivities"),
        ],
    },
    # -- saldos (balance) -------------------------------------------------
    "assets": {
        "kind": "point",
        "units": _USD,
        "tags": [("us-gaap", "Assets"), ("ifrs-full", "Assets")],
    },
    "current_assets": {
        "kind": "point",
        "units": _USD,
        "tags": [("us-gaap", "AssetsCurrent"), ("ifrs-full", "CurrentAssets")],
    },
    "liabilities": {
        "kind": "point",
        "units": _USD,
        "tags": [("us-gaap", "Liabilities"), ("ifrs-full", "Liabilities")],
    },
    "current_liabilities": {
        "kind": "point",
        "units": _USD,
        "tags": [("us-gaap", "LiabilitiesCurrent"), ("ifrs-full", "CurrentLiabilities")],
    },
    "equity": {
        "kind": "point",
        "units": _USD,
        "tags": [
            ("us-gaap", "StockholdersEquity"),
            ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
            ("ifrs-full", "EquityAttributableToOwnersOfParent"),
            ("ifrs-full", "Equity"),
        ],
    },
    "cash": {
        "kind": "point",
        "units": _USD,
        "tags": [
            ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
            ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
            ("ifrs-full", "CashAndCashEquivalents"),
        ],
    },
    "debt_long": {
        "kind": "point",
        "units": _USD,
        "tags": [
            ("us-gaap", "LongTermDebtNoncurrent"),
            ("us-gaap", "LongTermDebt"),
            ("ifrs-full", "NoncurrentPortionOfNoncurrentBorrowings"),
            ("ifrs-full", "Borrowings"),
        ],
    },
    "debt_short": {
        "kind": "point",
        "units": _USD,
        "tags": [
            ("us-gaap", "DebtCurrent"),
            ("us-gaap", "ShortTermBorrowings"),
            ("us-gaap", "LongTermDebtCurrent"),
            ("ifrs-full", "ShorttermBorrowings"),
            ("ifrs-full", "CurrentPortionOfLongtermBorrowings"),
        ],
    },
    "inventory": {
        "kind": "point",
        "units": _USD,
        "tags": [("us-gaap", "InventoryNet"), ("ifrs-full", "Inventories")],
    },
    "shares": {
        "kind": "point",
        "units": _SHARES,
        "tags": [
            ("dei", "EntityCommonStockSharesOutstanding"),
            ("us-gaap", "CommonStockSharesOutstanding"),
            ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
            ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
            ("ifrs-full", "NumberOfSharesOutstanding"),
        ],
    },
}

FLOW_CONCEPTS = tuple(k for k, v in CONCEPTS.items() if v["kind"] == "flow")
POINT_CONCEPTS = tuple(k for k, v in CONCEPTS.items() if v["kind"] == "point")

# Formularios que cuentan como presentacion periodica. Se excluye el resto (S-1,
# 8-K, etc.): sus cifras son extractos, no estados completos.
_ACCEPTED_FORMS = {"10-K", "10-Q", "20-F", "40-F", "10-K/A", "10-Q/A", "20-F/A", "40-F/A"}


# ---------------------------------------------------------------------------
#  Extraccion (funciones puras: se prueban con JSON sintetico, sin red)
# ---------------------------------------------------------------------------


def extract_raw_facts(facts_json: Mapping[str, Any]) -> pd.DataFrame:
    """JSON de companyfacts -> filas tidy de TODAS las etiquetas candidatas.

    Columnas: concept, tag, priority, start, end, filed, value, form, kind.

    No elige entre etiquetas: eso lo hace `select_by_priority`, periodo a
    periodo. Guardar todas las candidatas es lo que permite cambiar el orden de
    prioridad mas adelante sin volver a descargar 5.000 ficheros de EDGAR.
    """
    facts = facts_json.get("facts", {}) if facts_json else {}
    rows: list[dict[str, Any]] = []

    for concept, spec in CONCEPTS.items():
        for priority, (namespace, tag) in enumerate(spec["tags"]):
            block = facts.get(namespace, {}).get(tag)
            if not block:
                continue
            units = block.get("units", {})
            for unit_name in spec["units"]:
                entries = units.get(unit_name)
                if not entries:
                    continue
                for entry in entries:
                    form = entry.get("form")
                    if form not in _ACCEPTED_FORMS:
                        continue
                    end = entry.get("end")
                    filed = entry.get("filed")
                    value = entry.get("val")
                    if end is None or filed is None or value is None:
                        continue
                    rows.append(
                        {
                            "concept": concept,
                            "tag": f"{namespace}:{tag}",
                            "priority": priority,
                            "start": entry.get("start"),
                            "end": end,
                            "filed": filed,
                            "value": float(value),
                            "form": form,
                            "kind": spec["kind"],
                        }
                    )

    if not rows:
        return _empty_raw()

    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    return df.dropna(subset=["end", "filed"])


def select_by_priority(raw: pd.DataFrame) -> pd.DataFrame:
    """Una etiqueta por periodo: la de mayor prioridad que tenga dato.

    **Por periodo, no por empresa.** La version anterior elegia una sola
    etiqueta para toda la historia de la empresa: la primera de la lista que
    tuviera algun dato. Con la transicion contable de ASC 606 (2018) eso salia
    carisimo -- Apple publica `RevenueFromContractWithCustomerExcludingAssessedTax`
    desde 2017 y `Revenues` antes, asi que el modelo se quedaba con la moderna,
    cortaba y perdia toda la historia de ingresos anterior a 2017. La cobertura
    de ingresos del panel era del 52%, y `sales_to_price` y `fcf_margin` caian
    solas por debajo del umbral.

    Elegir periodo a periodo cose las dos eras. El riesgo conocido es que
    ambas etiquetas no midan exactamente lo mismo en el punto de union y
    aparezca un salto que parezca crecimiento; es un riesgo acotado a un
    trimestre y mucho menor que perder seis anos de historia.
    """
    if raw.empty:
        return raw
    return (
        raw.sort_values(["priority", "filed"])
        .groupby(["concept", "start", "end"], as_index=False, dropna=False)
        .first()
    )


def _empty_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concept": pd.Series(dtype="object"),
            "tag": pd.Series(dtype="object"),
            "priority": pd.Series(dtype="int64"),
            "start": pd.Series(dtype="datetime64[ns]"),
            "end": pd.Series(dtype="datetime64[ns]"),
            "filed": pd.Series(dtype="datetime64[ns]"),
            "value": pd.Series(dtype="float64"),
            "form": pd.Series(dtype="object"),
            "kind": pd.Series(dtype="object"),
        }
    )


def _first_reported(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Primera publicacion de cada periodo: descarta reexpresiones."""
    if df.empty:
        return df
    return (
        df.sort_values("filed")
        .groupby(list(keys), as_index=False, dropna=False)
        .first()
    )


def quarterly_segments(flows: pd.DataFrame) -> pd.DataFrame:
    """Normaliza flujos a trimestres.

    Dos casos:
    - Ya viene trimestral (duracion ~90 dias): se toma tal cual.
    - Viene acumulado del año fiscal (mismo `start`, `end` creciente): el
      trimestre es la diferencia con el acumulado anterior. La fecha de
      presentacion del trimestre derivado es la del acumulado mas reciente de
      los dos, que es cuando el numero se pudo calcular.
    """
    if flows.empty:
        return flows.assign(duration=pd.Series(dtype="float64"))

    df = flows.dropna(subset=["start"]).copy()
    if df.empty:
        return _empty_raw().assign(duration=pd.Series(dtype="float64"))

    df["duration"] = (df["end"] - df["start"]).dt.days
    df = df[(df["duration"] > 20) & (df["duration"] < 400)]
    df = _first_reported(df, ["concept", "start", "end"])

    native = df[df["duration"].between(75, 115)].copy()

    # Acumulados: mismo concepto y mismo inicio de año fiscal, varios cierres.
    #
    # Se agrupa sobre TODAS las filas que comparten `start`, no solo sobre las
    # de duracion larga. El primer trimestre del año fiscal se publica como
    # "3 meses desde el inicio", y es a la vez trimestre nativo y el acumulado
    # contra el que se resta el semestre. Excluirlo dejaba Q2 sin calcular.
    pieces: list[pd.DataFrame] = [native]
    for (concept, start), group in df.groupby(["concept", "start"], dropna=False):
        group = group.sort_values("end")
        if len(group) < 2 or group["duration"].max() <= 115:
            continue
        prior_end = group["end"].shift(1)
        prior_value = group["value"].shift(1)
        prior_filed = group["filed"].shift(1)
        derived = pd.DataFrame(
            {
                "concept": concept,
                "tag": group["tag"].to_numpy(),
                "start": prior_end,
                "end": group["end"].to_numpy(),
                # Se pudo conocer cuando estaban presentados AMBOS acumulados.
                "filed": pd.concat([group["filed"], prior_filed], axis=1).max(axis=1),
                "value": group["value"].to_numpy() - prior_value,
                "form": group["form"].to_numpy(),
                "kind": "flow",
            }
        ).dropna(subset=["start", "value"])
        derived["duration"] = (derived["end"] - derived["start"]).dt.days
        pieces.append(derived[derived["duration"].between(75, 115)])

    out = pd.concat(pieces, ignore_index=True)
    if out.empty:
        return out
    # Ante duplicados, la version nativa manda sobre la derivada: menos
    # aritmetica, menos margen de error de redondeo.
    out["_native"] = out["duration"].between(85, 100).astype(int)
    out = (
        out.sort_values(["_native", "filed"], ascending=[False, True])
        .groupby(["concept", "end"], as_index=False)
        .first()
        .drop(columns="_native")
    )
    return out.sort_values(["concept", "end"]).reset_index(drop=True)


def trailing_twelve_months(flows: pd.DataFrame) -> pd.DataFrame:
    """Suma movil de cuatro trimestres, con respaldo anual.

    Devuelve columnas: concept, period_end, filed, value.
    """
    quarters = quarterly_segments(flows)
    results: list[dict[str, Any]] = []

    for concept, group in quarters.groupby("concept", dropna=False):
        group = group.sort_values("end").reset_index(drop=True)
        for i in range(3, len(group)):
            window = group.iloc[i - 3 : i + 1]
            span_days = (window["end"].iloc[-1] - window["start"].iloc[0]).days
            if not 330 <= span_days <= 400:
                continue  # hueco o solape: no es un año limpio
            results.append(
                {
                    "concept": concept,
                    "period_end": window["end"].iloc[-1],
                    "filed": window["filed"].max(),
                    "value": float(window["value"].sum()),
                }
            )

    ttm = pd.DataFrame(results)

    # Respaldo: cifras anuales de la memoria, para los periodos que la suma
    # trimestral no pudo cubrir (tipico en emisores extranjeros que solo
    # presentan semestres).
    annual = flows.dropna(subset=["start"]).copy()
    if not annual.empty:
        annual["duration"] = (annual["end"] - annual["start"]).dt.days
        annual = annual[annual["duration"].between(350, 385)]
        annual = _first_reported(annual, ["concept", "end"])
        annual = annual.rename(columns={"end": "period_end"})[
            ["concept", "period_end", "filed", "value"]
        ]
        if ttm.empty:
            ttm = annual
        else:
            missing = annual.merge(
                ttm[["concept", "period_end"]], on=["concept", "period_end"], how="left", indicator=True
            )
            missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
            ttm = pd.concat([ttm, missing], ignore_index=True)

    if ttm.empty:
        return pd.DataFrame(
            {
                "concept": pd.Series(dtype="object"),
                "period_end": pd.Series(dtype="datetime64[ns]"),
                "filed": pd.Series(dtype="datetime64[ns]"),
                "value": pd.Series(dtype="float64"),
            }
        )
    return ttm.sort_values(["concept", "period_end"]).reset_index(drop=True)


def point_in_time_balances(points: pd.DataFrame) -> pd.DataFrame:
    """Saldos de balance, primera publicacion de cada cierre."""
    if points.empty:
        return pd.DataFrame(
            {
                "concept": pd.Series(dtype="object"),
                "period_end": pd.Series(dtype="datetime64[ns]"),
                "filed": pd.Series(dtype="datetime64[ns]"),
                "value": pd.Series(dtype="float64"),
            }
        )
    df = _first_reported(points, ["concept", "end"])
    return (
        df.rename(columns={"end": "period_end"})[["concept", "period_end", "filed", "value"]]
        .sort_values(["concept", "period_end"])
        .reset_index(drop=True)
    )


def facts_to_observations(facts_json: Mapping[str, Any]) -> pd.DataFrame:
    """companyfacts -> observaciones listas para el panel.

    Columnas: concept, period_end, filed, value. Los flujos vienen en TTM, los
    saldos en su valor de cierre.
    """
    return observations_from_raw(extract_raw_facts(facts_json))


def observations_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Filas crudas cacheadas -> observaciones listas para el panel.

    Se separa de `facts_to_observations` para poder recalcular desde la cache
    sin volver a pedir nada a EDGAR.
    """
    if raw.empty:
        return point_in_time_balances(raw)
    raw = select_by_priority(raw)
    flows = raw[raw["kind"] == "flow"]
    points = raw[raw["kind"] == "point"]
    out = pd.concat(
        [trailing_twelve_months(flows), point_in_time_balances(points)],
        ignore_index=True,
    )
    return out.sort_values(["concept", "period_end"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
#  Descarga
# ---------------------------------------------------------------------------


def listed_companies(max_age_days: float = 7) -> pd.DataFrame:
    """Todas las cotizadas con ticker y mercado, segun la propia SEC.

    Columnas: cik, ticker, name, exchange.
    """
    cached = cache.read_frame("sec_tickers", max_age_days=max_age_days)
    if cached is not None:
        return cached

    session = _session()
    payload = _get_json(session, SEC_TICKERS_URL)
    if not payload:
        raise SECError("la SEC devolvio una lista de tickers vacia")

    fields = [f.lower() for f in payload["fields"]]
    df = pd.DataFrame(payload["data"], columns=fields)
    df = df.rename(columns={"cik": "cik", "exchange": "exchange"})
    df["cik"] = df["cik"].astype("int64")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["exchange"] = df["exchange"].astype(str)
    df = df[["cik", "ticker", "name", "exchange"]].dropna(subset=["ticker"])
    cache.write_frame("sec_tickers", df)
    return df


# Version de la logica que convierte filas crudas en observaciones: seleccion de
# etiqueta por periodo, primera publicacion y reconstruccion del TTM. Forma parte
# del nombre de la cache de observaciones, asi que SUBIRLA invalida todas las
# observaciones calculadas con la logica anterior y las recalcula desde las
# crudas, sin tocar EDGAR.
#
# Regla: cualquier cambio en `select_by_priority`, `quarterly_segments`,
# `trailing_twelve_months` o `point_in_time_balances` sube esta version. Si no
# se sube, el panel mezcla observaciones de la logica vieja con codigo nuevo y
# nada falla ni avisa.
OBSERVATIONS_VERSION = "v2"


def _raw_name(cik: int) -> str:
    return f"facts_raw/cik_{cik:010d}"


def _observations_name(cik: int) -> str:
    return f"facts_obs_{OBSERVATIONS_VERSION}/cik_{cik:010d}"


def _derive_and_cache(cik: int, raw: pd.DataFrame) -> pd.DataFrame:
    observations = observations_from_raw(raw)
    observations.insert(0, "cik", cik)
    cache.write_frame(_observations_name(cik), observations)
    return observations


def company_observations(cik: int, *, session: requests.Session | None = None,
                         refresh: bool = False) -> pd.DataFrame:
    """Observaciones de una empresa, con dos niveles de cache por CIK.

    1. **Observaciones** (`facts_obs_<version>/`): el resultado final. Leerlo es
       inmediato.
    2. **Filas crudas** (`facts_raw/`): todas las etiquetas candidatas tal como
       las publica EDGAR. Si la version de la logica cambia, las observaciones
       se recalculan desde aqui sin red.

    Antes solo existia el nivel 2, y reconstruir el panel recalculaba el TTM de
    3.800 empresas en cada corrida: unos veinte minutos por iteracion, que es
    justo lo que desanima a probar una hipotesis mas. Con el nivel 1 esa parte
    baja a segundos.

    Ninguna cache caduca: un 10-K de 2015 no cambia. Para traer presentaciones
    nuevas, `refresh=True` vuelve a EDGAR y rehace ambos niveles.
    """
    if not refresh:
        observations = cache.read_frame(_observations_name(cik))
        if observations is not None:
            return observations
        raw = cache.read_frame(_raw_name(cik))
        if raw is not None:
            return _derive_and_cache(cik, raw)

    own_session = session is None
    session = session or _session()
    try:
        payload = _get_json(session, SEC_FACTS_URL.format(cik=cik))
    finally:
        if own_session:
            session.close()

    # Se cachean las filas CRUDAS, con todas las etiquetas candidatas, ademas de
    # las observaciones. Cuesta algo mas de disco y ahorra la leccion que costo
    # el fallo de ASC 606: cambiar el orden de prioridad de las etiquetas, o la
    # reconstruccion del TTM, obligaba a volver a descargar 5.000 ficheros de
    # EDGAR. Con las crudas en disco, ese cambio se recalcula sin red.
    raw = extract_raw_facts(payload or {})
    cache.write_frame(_raw_name(cik), raw)
    return _derive_and_cache(cik, raw)


def download_fundamentals(
    ciks: Iterable[int],
    *,
    refresh: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Descarga (o lee de cache) los fundamentales de varias empresas.

    Reanudable: cada CIK se cachea por separado, asi que una descarga cortada a
    mitad no obliga a empezar de cero.
    """
    ciks = list(dict.fromkeys(int(c) for c in ciks))
    frames: list[pd.DataFrame] = []
    session = _session()
    failures = 0
    try:
        for i, cik in enumerate(ciks, start=1):
            try:
                frames.append(company_observations(cik, session=session, refresh=refresh))
            except SECError as exc:
                failures += 1
                if failures > max(10, len(ciks) // 10):
                    raise SECError(
                        f"demasiados fallos consecutivos contra EDGAR ({failures}). "
                        f"Ultimo: {exc}"
                    ) from exc
            if progress and (i % 25 == 0 or i == len(ciks)):
                print(f"  fundamentales: {i}/{len(ciks)} empresas", flush=True)
    finally:
        session.close()

    if not frames:
        return pd.DataFrame(
            columns=["cik", "concept", "period_end", "filed", "value"]
        )
    out = pd.concat(frames, ignore_index=True)
    return out[out["value"].notna()].reset_index(drop=True)


def as_of(observations: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Corte point-in-time: lo ultimo presentado en o antes de `date`.

    Esta funcion es la barrera contra el look-ahead. Si alguien la rodea y
    filtra por `period_end`, el backtest deja de ser valido.
    """
    if observations.empty:
        return observations
    visible = observations[observations["filed"] <= pd.Timestamp(date)]
    if visible.empty:
        return visible
    latest = (
        visible.sort_values(["period_end", "filed"])
        .groupby(["cik", "concept"], as_index=False)
        .last()
    )
    wide = latest.pivot(index="cik", columns="concept", values="value")
    ages = latest.pivot(index="cik", columns="concept", values="period_end")
    wide["last_period_end"] = ages.max(axis=1)
    return wide.reset_index()
