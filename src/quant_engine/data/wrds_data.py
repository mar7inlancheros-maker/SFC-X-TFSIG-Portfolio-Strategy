"""Datos de WRDS para el motor: factores, GICS, fundamentales, analistas y cortos.

Que aporta cada fuente y por que:

- **Fama-French 5 + momentum** (`ff.fivefactors_daily`): los factores academicos
  de verdad, en vez de spreads de ETF como aproximacion.
- **GICS** (`comp.company`): el esquema sectorial que usa la industria, en vez
  del SIC de 1987 (que ponia a PayPal en "Industrials").
- **Fundamentales point-in-time** (`comp.fundq`): valor y calidad con los
  numeros que el mercado TENIA: solo trimestres con fecha de publicacion
  `rdq` anterior al dia del analisis.
- **Analistas** (`ibes`): consenso de recomendaciones y, sobre todo, la
  REVISION de estimaciones de beneficio a 3 meses, una de las senales mas
  documentadas (Chan, Jegadeesh y Lakonishok 1996).
- **Interes corto** (`comp.sec_shortint`): en un libro long/short importa por
  partida doble: mucho interes corto predice peores retornos (Boehmer, Jones y
  Zhang 2008), y en un nombre que se vende en corto significa riesgo de
  squeeze y coste de prestamo alto.

Si WRDS no esta configurado o falla, `load` devuelve `available=False` con el
motivo, y el motor sigue con Yahoo + ETFs. Nunca se inventa un dato.

Las consultas llevan la fecha del analisis en el SQL: la cache (que lleva hash
del SQL) se renueva sola cada dia y los datos recientes no se quedan viejos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GICS_SECTORS = {
    "10": "Energy", "15": "Materials", "20": "Industrials", "25": "ConsumerDiscretionary",
    "30": "ConsumerStaples", "35": "HealthCare", "40": "Financials", "45": "Technology",
    "50": "CommunicationServices", "55": "Utilities", "60": "RealEstate",
}
FF_COLUMNS = ("mktrf", "smb", "hml", "rmw", "cma", "umd")


def _numeric(frame: pd.DataFrame, exclude: tuple[str, ...]) -> pd.DataFrame:
    """WRDS devuelve columnas anulables con `pd.NA`, que revienta `float()`.
    Todo lo numerico pasa a float64 con NaN."""
    out = frame.copy()
    for col in out.columns:
        if col not in exclude:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


@dataclass
class WrdsData:
    available: bool
    reason: str = ""
    factors: pd.DataFrame = field(default_factory=pd.DataFrame)       # fecha x factor (+ rf)
    ids: pd.DataFrame = field(default_factory=pd.DataFrame)           # ticker -> gvkey, ibtic, gics
    fundamentals: pd.DataFrame = field(default_factory=pd.DataFrame)  # ticker -> conceptos TTM
    analysts: pd.DataFrame = field(default_factory=pd.DataFrame)      # ticker -> consenso y revisiones
    short_interest: pd.DataFrame = field(default_factory=pd.DataFrame)  # ticker -> shortint, fecha
    unmatched: list[str] = field(default_factory=list)

    @property
    def sectors(self) -> dict[str, str]:
        if self.ids.empty or "gsector" not in self.ids:
            return {}
        return {t: GICS_SECTORS.get(str(s), "Unknown") for t, s in self.ids["gsector"].items() if pd.notna(s)}


# ---------------------------------------------------------------------------
#  Calculos puros (se prueban sin red)
# ---------------------------------------------------------------------------


def pick_primary_security(securities: pd.DataFrame) -> pd.DataFrame:
    """Una fila por ticker: la emision principal.

    Un mismo `tic` puede tener varias filas en Compustat (clases, mercados).
    Preferencia: activa, emision '01', cotizada en EE.UU. o Canada.
    """
    if securities.empty:
        return securities
    s = securities.copy()
    s["_active"] = s.get("secstat", pd.Series("A", index=s.index)).fillna("A").eq("A")
    s["_main"] = s["iid"].astype(str).eq("01")
    s["_na"] = s.get("excntry", pd.Series("USA", index=s.index)).isin(["USA", "CAN"])
    s = s.sort_values(["tic", "_active", "_na", "_main"], ascending=[True, False, False, False])
    return s.drop_duplicates("tic").set_index("tic").drop(columns=["_active", "_main", "_na"])


def quarterly_from_ytd(frame: pd.DataFrame, column: str) -> pd.Series:
    """Compustat da la caja operativa y el capex ACUMULADOS en el ano fiscal
    (`oancfy`, `capxy`). El trimestre es la diferencia con el acumulado previo
    del mismo ano fiscal; el primer trimestre del ano es el propio acumulado."""
    f = frame.sort_values(["fyearq", "fqtr"])
    prev = f.groupby("fyearq")[column].shift(1)
    q = f[column] - prev
    q = q.where(f["fqtr"] > 1, f[column])
    return q.reindex(frame.index)


def ttm_fundamentals(fundq: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series | None:
    """Conceptos de los ultimos 4 trimestres PUBLICADOS antes de `as_of`.

    Point-in-time por `rdq` (fecha de publicacion). Sin `rdq`, se supone
    publicado 90 dias despues del cierre: conservador, nunca antes.
    Devuelve None si no hay 4 trimestres consecutivos: un TTM con huecos mezcla
    periodos y no significa nada.
    """
    f = _numeric(fundq, ("gvkey", "datadate", "rdq"))
    f["datadate"] = pd.to_datetime(f["datadate"])
    f["rdq"] = pd.to_datetime(f["rdq"]).fillna(f["datadate"] + pd.Timedelta(days=90))
    f = f[f["rdq"] <= as_of].sort_values("datadate").drop_duplicates("datadate", keep="last")
    if len(f) < 4:
        return None
    for col in ("oancfy", "capxy"):
        if col in f:
            f[col.replace("y", "q_derived")] = quarterly_from_ytd(f, col)
    last4 = f.iloc[-4:]
    span = (last4["datadate"].iloc[-1] - last4["datadate"].iloc[0]).days
    if not 250 <= span <= 300:  # tres trimestres entre el primero y el ultimo
        return None
    latest = f.iloc[-1]

    def ttm(col: str) -> float:
        return float(last4[col].sum(min_count=4)) if col in last4 else np.nan

    shares_prev = f.iloc[-5]["cshoq"] if len(f) >= 5 else np.nan
    return pd.Series({
        "revenue": ttm("saleq"),
        "cogs": ttm("cogsq"),
        "net_income": ttm("niq"),
        "operating_income": ttm("oiadpq"),
        "interest_expense": ttm("xintq"),
        "ocf": ttm("oancfq_derived"),
        "capex": ttm("capxq_derived"),
        "equity": latest.get("ceqq"),
        "assets": latest.get("atq"),
        "liabilities": latest.get("ltq"),
        "debt_long": latest.get("dlttq"),
        "debt_short": latest.get("dlcq"),
        "cash": latest.get("cheq"),
        "current_assets": latest.get("actq"),
        "current_liabilities": latest.get("lctq"),
        "shares_mm": latest.get("cshoq"),
        "shares_growth": (latest.get("cshoq") / shares_prev - 1.0) if shares_prev and shares_prev > 0 else np.nan,
        "fiscal_period_end": latest["datadate"],
        "report_date": latest["rdq"],
    })


def fundamental_metrics(concepts: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """Ratios de valor y calidad con las MISMAS formulas del modelo multifactor.

    Reutiliza `sfc_tfsig.financials`: un ROE o un rendimiento por beneficio
    significan lo mismo en los dos motores del repo. Compustat va en millones;
    la capitalizacion se calcula con el precio de hoy y las acciones del
    ultimo trimestre publicado.
    """
    from sfc_tfsig import financials  # noqa: PLC0415

    if concepts.empty:
        return pd.DataFrame()
    df = concepts.copy()
    df["market_cap"] = prices.reindex(df.index) * df["shares_mm"]
    out = pd.concat([financials.value_metrics(df), financials.quality_metrics(df)], axis=1)
    out["market_cap_mm"] = df["market_cap"]
    out["report_date"] = df["report_date"]
    return out


def analyst_signals(recs: pd.DataFrame, eps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series | None:
    """Consenso, cambio de recomendacion y revision de estimaciones a 3 meses.

    - `meanrec`: 1 = compra fuerte ... 5 = venta. MAS BAJO es mas favorable.
    - `rec_change_3m`: meanrec hoy menos hace 3 meses. Negativo = mejora.
    - `eps_revision_3m`: cambio de la estimacion media del ejercicio EN CURSO
      (fpi=1, mismo `fpedats`) en 3 meses, sobre su valor absoluto. Se compara
      el MISMO periodo fiscal: comparar estimaciones de ejercicios distintos
      mediria crecimiento esperado, no revision.
    """
    out: dict[str, float] = {}
    recs = _numeric(recs, ("ticker", "statpers"))
    eps = _numeric(eps, ("ticker", "statpers", "fpedats"))
    r = recs[pd.to_datetime(recs["statpers"]) <= as_of].sort_values("statpers")
    if len(r):
        now = r.iloc[-1]
        out["meanrec"] = float(now["meanrec"])
        out["numrec"] = float(now["numrec"])
        past = r[pd.to_datetime(r["statpers"]) <= as_of - pd.DateOffset(months=3)]
        out["rec_change_3m"] = float(now["meanrec"] - past.iloc[-1]["meanrec"]) if len(past) else np.nan

    e = eps[pd.to_datetime(eps["statpers"]) <= as_of].sort_values("statpers")
    if len(e):
        now = e.iloc[-1]
        same = e[e["fpedats"] == now["fpedats"]]
        past = same[pd.to_datetime(same["statpers"]) <= as_of - pd.DateOffset(months=3)]
        if len(past) and past.iloc[-1]["meanest"] and abs(past.iloc[-1]["meanest"]) > 1e-9:
            base = past.iloc[-1]["meanest"]
            # Acotada a +-100%: con una estimacion base cerca de cero el cociente
            # explota (Boeing daba -133%) y, con 10 nombres, un solo valor asi
            # dominaria todo el z-score del componente.
            out["eps_revision_3m"] = float(np.clip((now["meanest"] - base) / abs(base), -1.0, 1.0))
        else:
            out["eps_revision_3m"] = np.nan
        out["eps_estimates"] = float(now["numest"])
        out["eps_dispersion"] = float(now["stdev"] / abs(now["meanest"])) if now["meanest"] else np.nan
        out["eps_up_down"] = float(now["numup"] - now["numdown"])
    return pd.Series(out) if out else None


def short_interest_metrics(si: pd.DataFrame, shares_mm: pd.Series, adv_shares: pd.Series) -> pd.DataFrame:
    """% de acciones en corto y dias para cubrir.

    `shortint` va en acciones; `cshoq` en millones. Dias para cubrir = interes
    corto / volumen medio diario en ACCIONES: cuantos dias de volumen entero
    harian falta para que los cortos recompraran.
    """
    if si.empty:
        return pd.DataFrame()
    df = si.copy()
    df["si_pct_float"] = df["shortint"] / (shares_mm.reindex(df.index) * 1e6)
    df["days_to_cover"] = df["shortint"] / adv_shares.reindex(df.index)
    return df


# ---------------------------------------------------------------------------
#  Consultas (red)
# ---------------------------------------------------------------------------


def _in_list(values) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def load(tickers: list[str] | tuple[str, ...], start: pd.Timestamp, as_of: pd.Timestamp) -> WrdsData:
    """Todo lo de WRDS en una sola conexion. Nunca lanza: si falla, lo dice."""
    try:
        from sfc_tfsig.data import wrds  # noqa: PLC0415

        conn_ctx = wrds.connection()
        conn = conn_ctx.__enter__()
    except Exception as exc:  # noqa: BLE001 -- se reporta, el motor sigue sin WRDS
        return WrdsData(available=False, reason=f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}")

    day = as_of.strftime("%Y-%m-%d")
    q = lambda sql, name: wrds.query(sql, cache_name=f"qe_{name}", conn=conn)  # noqa: E731
    try:
        factors = q(f"select date, {', '.join(FF_COLUMNS)}, rf from ff.fivefactors_daily "
                    f"where date >= '{start:%Y-%m-%d}' and date <= '{day}'", "ff5")
        factors["date"] = pd.to_datetime(factors["date"])
        factors = factors.set_index("date").astype(float)

        secs = q(f"select s.tic, s.gvkey, s.iid, s.ibtic, s.excntry, s.secstat, c.gsector, c.gind, c.conm "
                 f"from comp.security s join comp.company c using (gvkey) "
                 f"where s.tic in ({_in_list(tickers)})", f"ids_{day}")
        ids = pick_primary_security(secs)
        unmatched = [t for t in tickers if t not in ids.index]

        gvkeys = list(ids["gvkey"].dropna().unique())
        fund_rows, analyst_rows, si = {}, {}, pd.DataFrame()
        if gvkeys:
            fq = q("select gvkey, datadate, rdq, fyearq, fqtr, saleq, cogsq, niq, oiadpq, xintq, oancfy, capxy, "
                   "ceqq, atq, ltq, dlttq, dlcq, cheq, actq, lctq, cshoq from comp.fundq "
                   f"where gvkey in ({_in_list(gvkeys)}) and indfmt='INDL' and datafmt='STD' and consol='C' "
                   f"and popsrc='D' and datadate >= '{as_of - pd.DateOffset(years=3):%Y-%m-%d}' "
                   f"and datadate <= '{day}'", f"fundq_{day}")
            for t, row in ids.iterrows():
                block = fq[fq["gvkey"] == row["gvkey"]]
                ttm = ttm_fundamentals(block, as_of) if len(block) else None
                if ttm is not None:
                    fund_rows[t] = ttm

            raw_si = q("select gvkey, iid, shortint, shortintadj, datadate from comp.sec_shortint "
                       f"where gvkey in ({_in_list(gvkeys)}) and datadate >= '{as_of - pd.DateOffset(months=3):%Y-%m-%d}' "
                       f"and datadate <= '{day}'", f"shortint_{day}")
            rows = {}
            for t, row in ids.iterrows():
                block = raw_si[(raw_si["gvkey"] == row["gvkey"]) & (raw_si["iid"] == row["iid"])]
                if len(block):
                    last = block.sort_values("datadate").iloc[-1]
                    rows[t] = {"shortint": float(last["shortint"]), "si_date": pd.Timestamp(last["datadate"])}
            si = pd.DataFrame(rows).T

        ibtics = ids["ibtic"].dropna()
        if len(ibtics):
            since = f"{as_of - pd.DateOffset(months=6):%Y-%m-%d}"
            recs = q(f"select ticker, statpers, meanrec, numrec from ibes.recdsum where ticker in "
                     f"({_in_list(ibtics.unique())}) and statpers >= '{since}' and statpers <= '{day}'",
                     f"ibes_rec_{day}")
            eps = q(f"select ticker, statpers, fpedats, meanest, stdev, numest, numup, numdown "
                    f"from ibes.statsum_epsus where ticker in ({_in_list(ibtics.unique())}) and fpi='1' "
                    f"and measure='EPS' and statpers >= '{since}' and statpers <= '{day}'", f"ibes_eps_{day}")
            for t, ib in ibtics.items():
                sig = analyst_signals(recs[recs["ticker"] == ib], eps[eps["ticker"] == ib], as_of)
                if sig is not None:
                    analyst_rows[t] = sig

        return WrdsData(
            available=True,
            factors=factors,
            ids=ids,
            fundamentals=pd.DataFrame(fund_rows).T,
            analysts=pd.DataFrame(analyst_rows).T,
            short_interest=si,
            unmatched=unmatched,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("fallo leyendo WRDS")
        return WrdsData(available=False, reason=f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
    finally:
        conn_ctx.__exit__(None, None, None)
