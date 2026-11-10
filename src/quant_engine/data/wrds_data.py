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
    fundq: pd.DataFrame = field(default_factory=pd.DataFrame)         # trimestres crudos, para la historia
    ibes_recs: pd.DataFrame = field(default_factory=pd.DataFrame)     # consenso mensual crudo
    ibes_eps: pd.DataFrame = field(default_factory=pd.DataFrame)      # estimaciones mensuales crudas
    short_interest_raw: pd.DataFrame = field(default_factory=pd.DataFrame)  # quincenal crudo
    funda: pd.DataFrame = field(default_factory=pd.DataFrame)         # anual (comp.funda), para el DCF
    ibes_sales: pd.DataFrame = field(default_factory=pd.DataFrame)    # consenso de ventas FY1/FY2

    @property
    def sectors(self) -> dict[str, str]:
        if self.ids.empty or "gsector" not in self.ids:
            return {}
        return {t: GICS_SECTORS.get(str(s), "Unknown") for t, s in self.ids["gsector"].items() if pd.notna(s)}


# ---------------------------------------------------------------------------
#  Calculos puros (se prueban sin red)
# ---------------------------------------------------------------------------


CANADIAN_SUFFIXES = (".TO", ".V", ".NE", ".CN")


def compustat_ticker(yahoo: str) -> tuple[str, str]:
    """Ticker de Compustat y pais de la emision para un ticker de Yahoo.

    Compustat guarda los listados de Toronto con su propio `tic`: sin clase,
    con un punto final (Aritzia: Yahoo `ATZ.TO`, Compustat `ATZ.`); con clase,
    punto y clase (Yahoo `TECK-B.TO`, Compustat `TECK.B`). Las clases de EE.UU.
    van igual (`BRK-B` -> `BRK.B`). Buscar el ticker de Yahoo tal cual dejaba
    sin fundamentales ni analistas a todo nombre canadiense.
    """
    t = yahoo.upper()
    for suffix in CANADIAN_SUFFIXES:
        if t.endswith(suffix):
            base = t[: -len(suffix)]
            return (base.replace("-", ".") if "-" in base else base + "."), "CAN"
    return t.replace("-", "."), "USA"


def match_securities(securities: pd.DataFrame, wanted: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Una fila por ticker de Yahoo: su emision en Compustat.

    `securities` trae todas las emisiones de las empresas buscadas. Si la
    emision elegida no tiene ticker de IBES (el listado de Toronto de Aritzia
    no lo tiene; su ADR de EE.UU. si), se toma el de una emision hermana de la
    misma empresa: IBES cubre empresas, no bolsas.
    """
    rows = {}
    for yahoo, (tic, country) in wanted.items():
        candidates = securities[securities["tic"] == tic]
        if candidates.empty:
            continue
        row = pick_primary_security(candidates, country=country).iloc[0].copy()
        if pd.isna(row.get("ibtic")):
            siblings = securities[(securities["gvkey"] == row["gvkey"]) & securities["ibtic"].notna()]
            if len(siblings):
                order = siblings.assign(
                    _same=siblings["excntry"].eq(country),
                    _active=siblings.get("secstat", pd.Series("A", index=siblings.index)).fillna("A").eq("A"))
                row["ibtic"] = order.sort_values(["_same", "_active"], ascending=False).iloc[0]["ibtic"]
        rows[yahoo] = row
    if not rows:
        return pd.DataFrame(columns=[c for c in securities.columns if c != "tic"])
    return pd.DataFrame(rows).T


def price_in_currency(prices_usd: pd.Series, currencies: pd.Series,
                      usd_per_unit: dict[str, float]) -> pd.Series:
    """Precio de cada nombre en la moneda en que reporta sus estados financieros.

    El rendimiento por beneficio es beneficio / capitalizacion: los dos tienen
    que ir en la misma moneda. Aritzia reporta en CAD; Shopify cotiza en Toronto
    y reporta en USD. Sin moneda declarada se supone USD (Compustat EE.UU.); con
    una moneda sin tipo de cambio, NaN: mejor sin ratio de valor que uno 30%
    equivocado.
    """
    out = {}
    for t, price in prices_usd.items():
        ccy = currencies.get(t) if len(currencies) else None
        if ccy is None or pd.isna(ccy) or ccy == "USD":
            out[t] = price
        else:
            out[t] = price / usd_per_unit[ccy] if usd_per_unit.get(ccy) else np.nan
    return pd.Series(out, dtype=float)


def pick_primary_security(securities: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    """Una fila por ticker: la emision principal.

    Un mismo `tic` puede tener varias filas en Compustat (clases, mercados).
    Preferencia: activa, del pais pedido (o de EE.UU. o Canada), emision '01'
    ('01C' en Toronto).
    """
    if securities.empty:
        return securities
    s = securities.copy()
    s["_active"] = s.get("secstat", pd.Series("A", index=s.index)).fillna("A").eq("A")
    s["_main"] = s["iid"].astype(str).str.startswith("01")
    places = [country] if country else ["USA", "CAN"]
    s["_na"] = s.get("excntry", pd.Series("USA", index=s.index)).isin(places)
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
    f = _numeric(fundq, ("gvkey", "datadate", "rdq", "curcdq"))
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
        "currency": latest.get("curcdq") if pd.notna(latest.get("curcdq")) else "USD",
    })


def ttm_panel(fundq: pd.DataFrame, ids: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Conceptos TTM de cada ticker con lo publicado hasta `as_of` (point-in-time).

    La misma funcion sirve para hoy y para cada fecha de la validacion del
    score: un trimestre con `rdq` posterior a `as_of` no existe todavia.
    """
    rows = {}
    if fundq.empty or ids.empty:
        return pd.DataFrame()
    for t, row in ids.iterrows():
        block = fundq[fundq["gvkey"] == row["gvkey"]]
        ttm = ttm_fundamentals(block, as_of) if len(block) else None
        if ttm is not None:
            rows[t] = ttm
    return pd.DataFrame(rows).T


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


# Dias entre la fecha de liquidacion del interes corto y su publicacion. FINRA
# publica unos 8 dias habiles despues; 14 naturales es conservador. Solo para la
# historia: hoy, lo que esta en WRDS ya esta publicado.
SHORT_INTEREST_LAG_DAYS = 14


def analyst_panel(recs: pd.DataFrame, eps: pd.DataFrame, ids: pd.DataFrame,
                  as_of: pd.Timestamp) -> pd.DataFrame:
    """Senales de analistas de cada ticker con los resumenes de IBES hasta `as_of`.

    La misma funcion para hoy y para cada fecha de la validacion del score.
    """
    if ids.empty or "ibtic" not in ids or recs is None or eps is None:
        return pd.DataFrame()
    rows = {}
    for t, ib in ids["ibtic"].dropna().items():
        sig = analyst_signals(recs[recs["ticker"] == ib], eps[eps["ticker"] == ib], as_of)
        if sig is not None:
            rows[t] = sig
    return pd.DataFrame(rows).T


def short_interest_at(raw: pd.DataFrame, ids: pd.DataFrame, as_of: pd.Timestamp,
                      lag_days: int = 0) -> pd.DataFrame:
    """Ultimo interes corto de cada ticker liquidado hasta `as_of - lag_days`.

    Por emision (gvkey + iid): el interes corto es de una bolsa concreta.
    """
    if raw is None or raw.empty or ids.empty:
        return pd.DataFrame()
    cutoff = as_of - pd.Timedelta(days=lag_days)
    dates = pd.to_datetime(raw["datadate"])
    rows = {}
    for t, row in ids.iterrows():
        block = raw[(raw["gvkey"] == row["gvkey"]) & (raw["iid"] == row["iid"]) & (dates <= cutoff)]
        if len(block):
            last = block.sort_values("datadate").iloc[-1]
            rows[t] = {"shortint": float(last["shortint"]), "si_date": pd.Timestamp(last["datadate"])}
    return pd.DataFrame(rows).T


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
        return WrdsData(available=False, reason=f"{type(exc).__name__}: {str(exc).splitlines()[0][:400]}")

    day = as_of.strftime("%Y-%m-%d")
    q = lambda sql, name: wrds.query(sql, cache_name=f"qe_{name}", conn=conn)  # noqa: E731
    try:
        factors = q(f"select date, {', '.join(FF_COLUMNS)}, rf from ff.fivefactors_daily "
                    f"where date >= '{start:%Y-%m-%d}' and date <= '{day}'", "ff5")
        factors["date"] = pd.to_datetime(factors["date"])
        factors = factors.set_index("date").astype(float)

        wanted = {t: compustat_ticker(t) for t in tickers}
        # Todas las emisiones de las empresas buscadas: el ibtic puede estar en una hermana.
        tics = _in_list(sorted({tic for tic, _ in wanted.values()}))
        secs = q(f"select s.tic, s.gvkey, s.iid, s.ibtic, s.excntry, s.secstat, c.gsector, c.gind, c.conm "
                 f"from comp.security s join comp.company c using (gvkey) "
                 f"where s.gvkey in (select gvkey from comp.security where tic in ({tics}))", f"ids_{day}")
        ids = match_securities(secs, wanted)
        unmatched = [t for t in tickers if t not in ids.index]

        gvkeys = list(ids["gvkey"].dropna().unique())
        si, raw_si, analysts = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        recs, eps = pd.DataFrame(), pd.DataFrame()
        fundamentals, fq = pd.DataFrame(), pd.DataFrame()
        # Trimestres desde 15 meses antes del inicio de los datos: la validacion
        # del score necesita un TTM publicado en su primera fecha.
        fq_start = min(start, as_of - pd.DateOffset(years=3)) - pd.DateOffset(months=15)
        if gvkeys:
            fq = q("select gvkey, datadate, rdq, curcdq, fyearq, fqtr, saleq, cogsq, niq, oiadpq, xintq, oancfy, "
                   "capxy, "
                   "ceqq, atq, ltq, dlttq, dlcq, cheq, actq, lctq, cshoq from comp.fundq "
                   f"where gvkey in ({_in_list(gvkeys)}) and indfmt='INDL' and datafmt='STD' and consol='C' "
                   f"and popsrc='D' and datadate >= '{fq_start:%Y-%m-%d}' "
                   f"and datadate <= '{day}'", f"fundq_{day}")
            fundamentals = ttm_panel(fq, ids, as_of)

            # Desde el inicio de los datos: la validacion del score lo necesita.
            raw_si = q("select gvkey, iid, shortint, shortintadj, datadate from comp.sec_shortint "
                       f"where gvkey in ({_in_list(gvkeys)}) and datadate >= '{start:%Y-%m-%d}' "
                       f"and datadate <= '{day}'", f"shortint_{day}")
            si = short_interest_at(raw_si, ids, as_of)

        ibtics = ids["ibtic"].dropna()
        if len(ibtics):
            since = f"{min(start, as_of - pd.DateOffset(months=6)):%Y-%m-%d}"
            recs = q(f"select ticker, statpers, meanrec, numrec from ibes.recdsum where ticker in "
                     f"({_in_list(ibtics.unique())}) and statpers >= '{since}' and statpers <= '{day}'",
                     f"ibes_rec_{day}")
            # Las empresas sin listado en EE.UU. estan en el fichero internacional de IBES.
            where = (f"where ticker in ({_in_list(ibtics.unique())}) and fpi='1' and measure='EPS' "
                     f"and statpers >= '{since}' and statpers <= '{day}'")
            cols = "ticker, statpers, fpedats, meanest, stdev, numest, numup, numdown"
            eps = q(f"select {cols} from ibes.statsum_epsus {where} union all "
                    f"select {cols} from ibes.statsum_epsint {where}", f"ibes_eps_{day}")
            analysts = analyst_panel(recs, eps, ids, as_of)

        # Para el DCF: estados anuales y consenso de ventas. Van al final y cada
        # una en su propio try: si una tabla o columna cambia en WRDS, se pierde
        # solo esa parte, no todo lo anterior.
        funda, sales = pd.DataFrame(), pd.DataFrame()
        if gvkeys:
            try:
                funda = q("select gvkey, datadate, fyear, curcd, revt, oiadp, dp, capx, stkco, txt, pi, xint, "
                          "che, dltt, dlc, mib, pstk, act, lct, ceq, cshfd, csho from comp.funda "
                          f"where gvkey in ({_in_list(gvkeys)}) and indfmt='INDL' and datafmt='STD' "
                          f"and consol='C' and popsrc='D' "
                          f"and datadate >= '{as_of - pd.DateOffset(years=9):%Y-%m-%d}' and datadate <= '{day}'",
                          f"funda_{day}")
            except Exception as exc:  # noqa: BLE001
                log.warning("comp.funda unavailable for the DCF: %s", exc)
        if len(ibtics):
            try:
                where_s = (f"where ticker in ({_in_list(ibtics.unique())}) and measure='SAL' and fpi in ('1', '2') "
                           f"and statpers >= '{as_of - pd.DateOffset(months=3):%Y-%m-%d}' and statpers <= '{day}'")
                sales = q(f"select ticker, statpers, fpi, fpedats, meanest, numest, 'USD' as curcode "
                          f"from ibes.statsum_xepsus {where_s} union all "
                          f"select ticker, statpers, fpi, fpedats, meanest, numest, curcode "
                          f"from ibes.statsum_xepsint {where_s}", f"ibes_sales_{day}")
            except Exception as exc:  # noqa: BLE001
                log.warning("IBES sales consensus unavailable: %s", exc)

        return WrdsData(
            available=True,
            factors=factors,
            ids=ids,
            fundamentals=fundamentals,
            analysts=analysts,
            short_interest=si,
            unmatched=unmatched,
            fundq=fq,
            ibes_recs=recs,
            ibes_eps=eps,
            short_interest_raw=raw_si,
            funda=funda,
            ibes_sales=sales,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("reading WRDS failed")
        return WrdsData(available=False, reason=f"{type(exc).__name__}: {str(exc).splitlines()[0][:400]}")
    finally:
        conn_ctx.__exit__(None, None, None)
