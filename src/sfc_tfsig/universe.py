"""Universo investible: que acciones pueden entrar en la cartera, y por que no las demas.

El universo se define ANTES de mirar ninguna señal. Si se filtra despues -- "me
quedo con las que el modelo puntua alto y ademas son liquidas" -- el filtro deja
de ser una restriccion operativa y se convierte en parte de la señal, sin haberlo
validado como tal.

Dos capas, deliberadamente separadas:

- **Estatica** (aqui): mercado de cotizacion, tipo de emisor, sector. Cambia
  poco y se resuelve una vez.
- **Dinamica** (en `panel.py`, fecha a fecha): precio minimo, volumen en dolares,
  capitalizacion, historia suficiente. Cambia cada mes y DEBE evaluarse con los
  datos de esa fecha, no con los de hoy.

**Sesgo de supervivencia.** La lista de cotizadas de la SEC es la de hoy. Las
empresas que dejaron de cotizar no aparecen, asi que el universo historico que
reconstruimos esta limpio de fracasos que si existieron. Es la limitacion mas
seria del modelo y no se arregla sin datos de pago. Ver README.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd
import requests

from .config import Config
from .data import cache, sec

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# Codigos de "estado" de la SEC que corresponden a provincias canadienses.
# Es como se identifica a un emisor canadiense sin pagar por una base de datos
# de domicilio: A0-A9 y B0 son Canada, el resto del alfabeto es otro pais.
_CANADA_CODES = {"A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "B0"}

# Nombres de mercado tal y como los escribe la SEC en company_tickers_exchange.
_EXCHANGE_ALIASES = {
    "nyse": "NYSE",
    "nasdaq": "Nasdaq",
    "nyseamerican": "NYSEAmerican",
    "nyse american": "NYSEAmerican",
    "nysearca": "NYSEArca",
    "cboe": "CBOE",
    "otc": "OTC",
}


def _normalize_exchange(name: str) -> str:
    return _EXCHANGE_ALIASES.get(str(name).strip().lower().replace(" ", ""), str(name).strip())


# ---------------------------------------------------------------------------
#  SIC -> sector
#
#  El sector no es cosmetico: se usa para estandarizar factores DENTRO del
#  sector (un ROE de banco y uno de software no son comparables) y para limitar
#  la concentracion. Aqui se traduce el SIC de la SEC a un esquema de 11
#  sectores al estilo GICS.
#
#  Es una aproximacion: el SIC es de 1987 y no conoce el software moderno ni el
#  comercio electronico. Se declara como tal. Si algun dia el club paga una
#  clasificacion GICS de verdad, este es el unico sitio que hay que cambiar.
# ---------------------------------------------------------------------------

#  EL ORDEN ES LA REGLA: de lo especifico a lo general. La primera coincidencia
#  gana, asi que los codigos concretos van SIEMPRE antes que los rangos anchos.
#  Con el orden invertido, el rango de quimicas (2800-2899) se tragaba a Johnson
#  & Johnson (2834, farmaceutica) y a Procter & Gamble (2840, consumo) y los
#  mandaba a Materiales -- y con ellos su ROE y sus margenes, que descolocaban
#  la estandarizacion sectorial de los tres sectores implicados.
_SIC_RULES: tuple[tuple[range | tuple[int, ...], str], ...] = (
    # -- energia ----------------------------------------------------------
    ((1311, 1381, 1382, 1389, 2911, 4922, 4923, 4924, 1221, 1220), "Energy"),
    (range(4600, 4620), "Energy"),   # oleoductos y gasoductos: Enbridge, TC Energy
    # -- salud (antes que quimicas) ---------------------------------------
    ((2833, 2834, 2835, 2836, 3826, 3841, 3842, 3843, 3844, 3845, 3851,
      5047, 5122, 8000, 8011, 8050, 8060, 8062, 8071, 8090, 8093, 8731), "HealthCare"),
    # -- consumo basico (antes que quimicas y que comercio) ---------------
    ((2000, 2011, 2013, 2015, 2020, 2024, 2030, 2033, 2040, 2050, 2052, 2060,
      2070, 2080, 2082, 2084, 2086, 2090, 2092, 2100, 2111, 2840, 2844,
      5140, 5141, 5400, 5411, 5412, 5912), "ConsumerStaples"),
    # -- tecnologia -------------------------------------------------------
    (range(3570, 3580), "Technology"),
    (range(3670, 3680), "Technology"),
    (range(7370, 7380), "Technology"),
    ((3559, 3661, 3663, 3669, 3827, 3861), "Technology"),
    # -- comunicaciones ---------------------------------------------------
    ((2711, 2721, 2731, 4813, 4822, 4832, 4833, 4841, 4899, 7311, 7812,
      7819, 7822, 7829, 7841, 7900, 7997), "CommunicationServices"),
    # -- servicios publicos ----------------------------------------------
    (range(4900, 4950), "Utilities"),
    # -- inmobiliario (antes que el rango financiero) ---------------------
    ((6500, 6510, 6512, 6513, 6519, 6531, 6552, 6798), "RealEstate"),
    # -- financiero -------------------------------------------------------
    (range(6000, 6500), "Financials"),
    (range(6700, 6800), "Financials"),
    # -- materiales -------------------------------------------------------
    (range(2800, 2900), "Materials"),
    (range(3300, 3400), "Materials"),
    ((1000, 1040, 1090, 1400, 2600, 2611, 2621, 2631, 3200, 3241), "Materials"),
    # -- consumo discrecional ---------------------------------------------
    ((2300, 2320, 2330, 2340, 3021, 3100, 3140, 3711, 3713, 3714, 3715, 3716,
      3751, 3790, 3942, 3944, 3949, 5700, 7011, 7990, 8200), "ConsumerDiscretionary"),
    (range(5200, 6000), "ConsumerDiscretionary"),
    # -- industrial (lo mas general de todo) ------------------------------
    (range(3400, 3600), "Industrials"),
    (range(3700, 3800), "Industrials"),
    (range(1600, 1800), "Industrials"),
    ((4011, 4100, 4200, 4210, 4213, 4220, 4231, 4400, 4412, 4512, 4513, 4522,
      4581, 4700, 4731, 5000, 5013, 5045, 5063, 5065, 5080, 5084,
      7350, 7359, 7363, 7389, 8711, 8742, 8744), "Industrials"),
)

# Los once sectores, en el orden en que aparecen arriba. Lo leen quienes
# validan nombres de sector escritos a mano (los escenarios de estres de
# config/risk.toml): un sector mal escrito no casaria con ninguna posicion y el
# escenario se aplicaria sin su choque sectorial, sin avisar.
SECTORS: tuple[str, ...] = tuple(dict.fromkeys(sector for _, sector in _SIC_RULES))


def sector_from_sic(sic: object) -> str:
    """SIC -> uno de los 11 sectores. 'Unknown' si no hay SIC o no encaja."""
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    for rule, sector in _SIC_RULES:
        if isinstance(rule, range):
            if code in rule:
                return sector
        elif code in rule:
            return sector
    # Fallback por division del SIC, mas grueso pero mejor que 'Unknown'.
    if 100 <= code < 1000:
        return "ConsumerStaples"
    if 1000 <= code < 1500:
        return "Materials"
    if 2000 <= code < 4000:
        return "Industrials"
    if 4000 <= code < 5000:
        return "Industrials"
    if 5000 <= code < 6000:
        return "ConsumerDiscretionary"
    if 7000 <= code < 9000:
        return "Industrials"
    return "Unknown"


# ---------------------------------------------------------------------------
#  Perfiles de empresa (SIC y pais) desde EDGAR
# ---------------------------------------------------------------------------


def company_profile(cik: int, *, session: requests.Session | None = None) -> dict:
    """SIC, descripcion, pais y nombre. Cacheado: casi nunca cambia."""
    name = f"profile_{cik:010d}"
    cached = cache.read_json(name, max_age_days=180)
    if cached is not None:
        return cached

    own_session = session is None
    session = session or sec._session()
    try:
        payload = sec._get_json(session, SEC_SUBMISSIONS_URL.format(cik=cik)) or {}
    finally:
        if own_session:
            session.close()

    address = (payload.get("addresses") or {}).get("business") or {}
    # Tres sitios, en este orden. Un emisor extranjero deja `stateOrCountry` a
    # null y pone el codigo en `countryCode`: Royal Bank of Canada salia como
    # pais desconocido leyendo solo el primero.
    state = str(
        address.get("stateOrCountry")
        or address.get("countryCode")
        or payload.get("stateOfIncorporation")
        or ""
    ).strip().upper()
    profile = {
        "cik": int(cik),
        "name": payload.get("name"),
        "sic": payload.get("sic"),
        "sic_description": payload.get("sicDescription"),
        "entity_type": payload.get("entityType"),
        "state_or_country": state,
        "country": "CA" if state in _CANADA_CODES else ("US" if state else "Unknown"),
    }
    cache.write_json(name, profile)
    return profile


def company_profiles(ciks: Iterable[int], *, progress: bool = True) -> pd.DataFrame:
    ciks = list(dict.fromkeys(int(c) for c in ciks))
    session = sec._session()
    rows = []
    try:
        for i, cik in enumerate(ciks, start=1):
            try:
                rows.append(company_profile(cik, session=session))
            except sec.SECError:
                rows.append({"cik": cik, "name": None, "sic": None, "sic_description": None,
                             "entity_type": None, "state_or_country": "", "country": "Unknown"})
            if progress and (i % 100 == 0 or i == len(ciks)):
                print(f"  perfiles: {i}/{len(ciks)} empresas", flush=True)
    finally:
        session.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Construccion del universo estatico
# ---------------------------------------------------------------------------


def _clean_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Quita lo que no es una accion ordinaria negociable.

    Los sufijos de Yahoo para warrants (`-WT`), unidades (`-UN`) y derechos
    (`-RT`) no son acciones: son derivados de una SPAC con liquidez y contabilidad
    propias. Las clases (`BRK-B`) SI se quedan.
    """
    ticker = df["ticker"].astype(str).str.upper()
    junk = ticker.str.contains(r"-(?:WT|WS|U|UN|RT|R|P[A-Z]?)$", regex=True, na=False)
    valid = ticker.str.fullmatch(r"[A-Z]{1,5}(-[A-Z]{1,2})?")
    return df[valid & ~junk].copy()


def build_universe(cfg: Config, *, with_profiles: bool = True, progress: bool = True) -> pd.DataFrame:
    """Universo estatico: cotizadas que cumplen mercado, tipo de emisor y sector.

    Columnas: cik, ticker, name, exchange, sic, sector, country.
    """
    listings = sec.listed_companies()
    listings = listings.copy()
    listings["exchange"] = listings["exchange"].map(_normalize_exchange)

    allowed = {_normalize_exchange(e) for e in cfg.get("universe.exchanges")}
    df = listings[listings["exchange"].isin(allowed)]
    df = _clean_tickers(df)

    # Una empresa puede tener varias clases cotizando (GOOG/GOOGL). Se queda una
    # por CIK: duplicar el mismo negocio en la cartera es concentracion
    # disfrazada de diversificacion.
    df = df.sort_values("ticker").drop_duplicates(subset="cik", keep="first")

    if not with_profiles:
        df["sic"] = pd.NA
        df["sector"] = "Unknown"
        df["country"] = "Unknown"
        return df[["cik", "ticker", "name", "exchange", "sic", "sector", "country"]].reset_index(drop=True)

    return attach_profiles(df, cfg, progress=progress)


def attach_profiles(df: pd.DataFrame, cfg: Config, *, progress: bool = True) -> pd.DataFrame:
    """Agrega SIC, sector y pais, y aplica las exclusiones que dependen de ellos.

    Se separa de `build_universe` porque cuesta una peticion a EDGAR por
    empresa. El pipeline lo llama DESPUES del filtro de liquidez: pedir el
    perfil de seis mil cotizadas para acabar usando mil quinientas son cuatro
    mil quinientas peticiones tiradas, y la SEC lleva la cuenta.
    """
    profiles = company_profiles(df["cik"], progress=progress)
    df = df.drop(columns=[c for c in ("sic", "sector", "country") if c in df.columns])
    df = df.merge(profiles[["cik", "sic", "country", "entity_type"]], on="cik", how="left")
    df["sector"] = df["sic"].map(sector_from_sic)

    excluded_sic = {int(s) for s in cfg.get("universe.exclude_sic", [])}
    if excluded_sic:
        sic_numeric = pd.to_numeric(df["sic"], errors="coerce")
        df = df[~sic_numeric.isin(excluded_sic)]

    # NO se filtra por `entityType == "operating"`. La SEC marca como "other" a
    # practicamente todo emisor extranjero -- Royal Bank of Canada incluido --,
    # asi que ese filtro borraba del universo justo la mitad canadiense del
    # mandato, en silencio y sin error. Los vehiculos de inversion se excluyen
    # por SIC, que si es fiable.

    countries = {"US", "CA", "Unknown"}
    df = df[df["country"].isin(countries)]

    return df[["cik", "ticker", "name", "exchange", "sic", "sector", "country"]].reset_index(drop=True)


def ever_liquid(dollar_volume: pd.DataFrame, min_dollar_volume: float) -> set[str]:
    """Tickers que alcanzaron el umbral de liquidez en ALGUN momento del historico.

    Sirve para no descargar fundamentales de empresas que nunca van a entrar en
    la cartera. Es un prefiltro de coste, no de inversion, y por eso usa el
    MAXIMO historico y no el valor de hoy: filtrar por la liquidez actual
    expulsaria a una empresa que fue liquida en 2015 y hoy no lo es, que es
    exactamente el tipo de nombre cuya exclusion sesga el backtest al alza.
    """
    if dollar_volume.empty:
        return set()
    peaks = dollar_volume.max(axis=0, skipna=True)
    return set(peaks[peaks >= min_dollar_volume].index)


def apply_liquidity_filters(
    candidates: pd.DataFrame,
    cfg: Config,
    *,
    price: pd.Series,
    dollar_volume: pd.Series,
    market_cap: pd.Series,
    history_months: pd.Series,
) -> pd.DataFrame:
    """Filtros dinamicos de una fecha concreta. Devuelve el universo de ese mes.

    Las series vienen indexadas por ticker y calculadas CON DATOS DE ESA FECHA.
    Esta funcion no sabe que fecha es, a proposito: asi no puede colarse un
    dato futuro por descuido.

    `history_months` es obligatorio, sin valor por defecto, a proposito. Hasta
    el 2026-09-23 `min_history_months = 24` estaba declarado en el TOML y
    validado en `config.py`, pero esta funcion no lo recibia: el parametro
    existia y no hacia nada. Tres salidas a bolsa de 3 a 4 meses (FRVO, MBGL,
    QNT) encabezaban el ranking de agosto de 2026. Sin historia no tienen
    momentum ni volatilidad; se puntuaban solo con valor y calidad, calculados
    sobre uno o dos reportes, y esos valores extremos las ponian arriba.
    """
    df = candidates.copy()
    df["price"] = df["ticker"].map(price)
    df["dollar_volume"] = df["ticker"].map(dollar_volume)
    df["market_cap"] = df["ticker"].map(market_cap)
    df["history_months"] = df["ticker"].map(history_months)

    mask = (
        (df["price"] >= float(cfg.get("universe.min_price")))
        & (df["dollar_volume"] >= float(cfg.get("universe.min_dollar_volume")))
        & (df["market_cap"] >= float(cfg.get("universe.min_market_cap")))
        & (df["history_months"] >= float(cfg.get("universe.min_history_months")))
    )
    out = df[mask.fillna(False)].copy()

    max_names = int(cfg.get("universe.max_names"))
    if len(out) > max_names:
        # Si sobran nombres se recorta por liquidez, que es el criterio
        # operativo, NO por capitalizacion ni por ninguna señal del modelo.
        out = out.nlargest(max_names, "dollar_volume")

    return out.reset_index(drop=True)


def summarize(universe: pd.DataFrame) -> Mapping[str, object]:
    """Resumen para el log y el reporte del comite."""
    return {
        "n_names": len(universe),
        "n_sectors": universe["sector"].nunique() if "sector" in universe else 0,
        "by_country": universe["country"].value_counts().to_dict() if "country" in universe else {},
        "by_sector": universe["sector"].value_counts().to_dict() if "sector" in universe else {},
    }
