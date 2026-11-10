"""DCF de una empresa de principio a fin, y de toda la canasta del analisis.

`value_company` es puro: recibe la historia, el precio y el coste de capital y
devuelve todo (escenarios, Monte Carlo, DCF inverso, sensibilidades,
comprobaciones). `history_for` y `run_valuations` traen los datos (SEC,
Compustat, Yahoo) y nunca lanzan: si una empresa no se puede valorar, el
resultado dice por que.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import assumptions as asm
from . import data as vdata
from .dcf import (
    Capital,
    cost_of_capital,
    monte_carlo,
    reverse_dcf,
    scenarios,
    sensitivity,
)

log = logging.getLogger(__name__)

STALE_MONTHS = 15
WACC_GRID = [-0.02, -0.01, 0.0, 0.01, 0.02]
G_GRID = [-0.01, -0.005, 0.0, 0.005, 0.01]
GROWTH_GRID = [-0.10, -0.05, 0.0, 0.05, 0.10]
MARGIN_GRID = [-0.04, -0.02, 0.0, 0.02, 0.04]


# Avisos que invalidan el numero: con cualquiera de ellos el DCF no opina sobre
# la senal. Los demas (terminal de mas del 80%, multiplo implicito raro, ROIC
# alto, datos viejos) se leen, pero no lo invalidan.
CRITICAL = ("thin operating margin", "equity value is zero", "WACC ", "negative operating margin",
            "share count from the cover page", "interest is paid but no debt", "revenue changes")


def critical_flags(flags: list[str]) -> list[str]:
    return [f for f in flags if f.startswith(CRITICAL)]


def stance(upside: float, band: float) -> str:
    if upside is None or not np.isfinite(upside):
        return "N/A"
    return "UNDERVALUED" if upside > band else "OVERVALUED" if upside < -band else "FAIR VALUE"


def value_company(ticker: str, hist: pd.DataFrame, currency: str, price: float, beta_raw: float, rf: float,
                  cfg: dict | None = None, *, shares_out: float | None = None, consensus: list[float] | None = None,
                  overrides: dict | None = None, usd_per_unit: float = 1.0, as_of: pd.Timestamp | None = None,
                  data_notes: list[str] | None = None, source: str = "") -> dict[str, object]:
    """Todo el DCF de una empresa. `price` en la moneda de reporte."""
    cfg = cfg or {}
    hist = hist.dropna(subset=["revenue"]).sort_index()
    if len(hist) < 2:
        return {"available": False, "reason": f"only {len(hist)} fiscal year(s) of financials"}
    last = hist.iloc[-1]
    if not last["revenue"] > 0:
        return {"available": False, "reason": "no revenue in the last fiscal year: a cash-flow DCF does not "
                                              "apply (pre-revenue; use a NAV or option model)"}
    diluted, basic = last.get("shares_diluted"), last.get("shares_basic")
    ok = lambda v: v is not None and pd.notna(v) and np.isfinite(v) and v > 0  # noqa: E731
    base_shares = diluted if ok(diluted) else basic if ok(basic) else np.nan
    share_flags: list[str] = []
    if ok(base_shares):
        shares, share_note = float(base_shares), "last-year diluted weighted shares"
        # La portada es mas reciente (recompras, emisiones), pero en empresas con
        # varias clases la SEC publica una sola: solo se usa si cuadra (+-15%).
        if ok(shares_out) and abs(shares_out / base_shares - 1) <= 0.15:
            ratio = diluted / basic if ok(diluted) and ok(basic) else 1.0
            shares, share_note = shares_out * max(ratio, 1.0), "latest shares outstanding x last-year diluted/basic"
    elif ok(shares_out):
        shares, share_note = float(shares_out), "cover-page shares outstanding (no diluted count)"
        share_flags.append("share count from the cover page only: with several share classes it can miss "
                           "some of them; verify before using the value per share")
    else:
        return {"available": False, "reason": "no share count"}
    num = lambda k: float(last[k]) if k in last and pd.notna(last[k]) else 0.0  # noqa: E731
    cap = Capital(price=float(price), shares=float(shares), cash=num("cash"), debt=num("debt"),
                  minority=num("minority"), preferred=num("preferred"))
    marginal_tax = asm.setting(cfg, "marginal_tax")
    marginal = float(marginal_tax.get(currency, marginal_tax.get("default", 0.25))
                     if isinstance(marginal_tax, dict) else marginal_tax)
    o = overrides or {}
    beta_used = float(o.get("beta", beta_raw if np.isfinite(beta_raw) else 1.0))
    coc = cost_of_capital(rf, beta_used, float(o.get("erp", asm.setting(cfg, "erp"))), marginal,
                          num("ebit"), num("interest_expense"), cap, cap.market_cap * usd_per_unit)
    built = asm.build(hist, coc, currency, cfg, consensus, o)
    cases = built["cases"]
    base = cases["base"]
    d = built["derived"]
    sc = scenarios(d["revenue"], cases, built["weights"], cap, d["invested_capital"], hist.index[-1])
    rev = d["revenue"]
    out: dict[str, object] = {
        "available": True, "ticker": ticker, "currency": currency, "source": source,
        "fiscal_end": hist.index[-1], "history": hist, "derived": d, "capital": cap, "cost_of_capital": coc,
        "cases": cases, "scenarios": sc, "weights": sc["weights"], "share_note": share_note,
        "expected": sc["expected"], "expected_upside": sc["expected_upside"],
        "base_value": sc["cases"]["base"]["per_share"], "base_upside": sc["cases"]["base"]["upside"],
        "sens_wacc_g": sensitivity(rev, base, cap, {"wacc": WACC_GRID}, {"g": G_GRID}),
        "sens_growth_margin": sensitivity(rev, base, cap, {"growth": GROWTH_GRID}, {"margin": MARGIN_GRID}),
        "monte_carlo": monte_carlo(rev, base, cap, growth_sd=d["growth_sd"], margin_sd=d["margin_sd"],
                                   seed=int(cfg.get("seed", 0)) if cfg else 0),
        "reverse": reverse_dcf(rev, base, cap),
        "consensus": consensus, "notes": list(data_notes or []) + list(base.notes),
        # Del caso base y no del ponderado: con el suelo en cero y un alcista
        # explosivo, el ponderado se sesga (GOLD: base 0, ponderado +818%).
        "stance": stance(sc["cases"]["base"]["upside"], float(cfg.get("fair_band", 0.20)) if cfg else 0.20),
    }
    flags = list(sc["cases"]["base"]["flags"]) + share_flags
    ratio = (hist["revenue"] / hist["revenue"].shift()).dropna().iloc[-3:]
    jumps = ratio[(ratio > 3) | (ratio < 1 / 3)]
    if len(jumps):
        flags.append(f"revenue changes {float(jumps.iloc[-1]):.1f}x in fiscal {jumps.index[-1]:%Y}: check the "
                     "revenue line (acquisition, spin-off or a mis-tagged filing)")
    margin = built["derived"]["margin"]
    if np.isfinite(margin) and abs(margin) < 0.03:
        flags.append(f"thin operating margin ({margin:.1%}): the value swings with tiny margin changes and revenue "
                     "may follow commodity prices; cross-check with EV/EBITDA or price/book")
    if num("debt") == 0 and num("interest_expense") > 0:
        flags.append("interest is paid but no debt was found in the filings: verify net debt")
    if as_of is not None and (as_of - hist.index[-1]).days > STALE_MONTHS * 30.5:
        flags.append(f"last fiscal year ends {hist.index[-1]:%Y-%m-%d}: more than {STALE_MONTHS} months old")
    out["flags"] = flags
    out["critical"] = critical_flags(flags)
    if out["critical"]:
        out["stance"] = "NOT DECISION-GRADE"
    out["usd_per_unit"] = usd_per_unit
    return out


# ---------------------------------------------------------------------------
#  Datos (red): nunca lanzan
# ---------------------------------------------------------------------------


def _sec_ticker_candidates(ticker: str) -> list[str]:
    t = ticker.upper()
    out = [t]
    if "." in t:
        base = t[: t.rindex(".")]
        out += [base, base.replace("-", ".")]
    return list(dict.fromkeys(out))


def sec_history(ticker: str) -> tuple[pd.DataFrame | None, str | None, list[str], float]:
    """Estados anuales de la SEC (cache 7 dias). (historia, moneda, notas, acciones ultimas)."""
    from sfc_tfsig.data import cache, sec  # noqa: PLC0415

    try:
        listing = sec.listed_companies().drop_duplicates("ticker").set_index("ticker")["cik"]
    except Exception as exc:  # noqa: BLE001
        return None, None, [f"SEC unavailable: {type(exc).__name__}"], np.nan
    cik = next((listing.get(c) for c in _sec_ticker_candidates(ticker) if listing.get(c) is not None), None)
    if cik is None or pd.isna(cik):
        return None, None, ["not an SEC filer (no 10-K, 20-F or 40-F)"], np.nan
    name = f"qe_dcf_sec/cik_{int(cik):010d}"
    cached = cache.read_frame(name, max_age_days=7)
    meta = cache.read_json(name + "_meta") or {}
    if cached is not None and meta:
        frame = cached.set_index(pd.to_datetime(cached["fiscal_end"])).drop(columns="fiscal_end")
        shares_out = meta.get("shares_out")
        return (frame, meta.get("currency"), list(meta.get("notes", [])),
                float(shares_out) if shares_out is not None else np.nan)
    session = sec._session()  # noqa: SLF001 -- mismo User-Agent y ritmo que el modelo multifactor
    try:
        payload = sec._get_json(session, sec.SEC_FACTS_URL.format(cik=int(cik)))  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        return None, None, [f"SEC download failed: {type(exc).__name__}: {exc}"], np.nan
    finally:
        session.close()
    if not payload:
        return None, None, ["no XBRL facts at the SEC"], np.nan
    frame, currency, notes = vdata.annual_from_sec(payload)
    shares_out, _ = vdata.latest_shares_outstanding(payload)
    cache.write_frame(name, frame.reset_index())
    cache.write_json(name + "_meta", {"currency": currency, "notes": notes,
                                      "shares_out": None if not np.isfinite(shares_out) else shares_out})
    return frame, currency, notes, shares_out


def risk_free_rate(as_of: pd.Timestamp, fallback: float) -> tuple[float, str]:
    """Rendimiento del bono del Tesoro a 10 anos (Yahoo ^TNX), o el del YAML."""
    from ..data import loader  # noqa: PLC0415

    try:
        market = loader.load_ohlcv(["^TNX"], as_of - pd.Timedelta(days=20), as_of)
        close = market.ohlcv["^TNX"]["Close"].dropna()
        v = float(close.iloc[-1])
        # Yahoo da el rendimiento en puntos (4,25); el indice de la CBOE, por 10.
        v = v / 10 if v > 20 else v
        rate = v / 100
        if 0.0 < rate < 0.15:
            return rate, f"US 10Y Treasury {rate:.2%} ({close.index[-1]:%Y-%m-%d}, Yahoo ^TNX)"
    except Exception as exc:  # noqa: BLE001
        log.info("no ^TNX: %s", exc)
    return fallback, f"risk-free {fallback:.2%} from valuation.risk_free in the YAML (^TNX unavailable)"


def history_for(ticker: str, wrds) -> tuple[pd.DataFrame | None, str | None, list[str], float, str]:
    """Compustat si WRDS lo trae; si no, SEC. (historia, moneda, notas, acciones, fuente)."""
    if wrds is not None and getattr(wrds, "available", False) and not getattr(wrds, "funda", pd.DataFrame()).empty:
        ids = wrds.ids
        if ticker in ids.index:
            block = wrds.funda[wrds.funda["gvkey"] == ids.loc[ticker, "gvkey"]]
            frame, currency = vdata.annual_from_compustat(block)
            if len(frame) >= 2:
                shares = np.nan
                if not wrds.fundamentals.empty and ticker in wrds.fundamentals.index:
                    shares = float(pd.to_numeric(wrds.fundamentals.loc[ticker, "shares_mm"], errors="coerce"))
                return frame, currency, [], shares, "Compustat (WRDS)"
    frame, currency, notes, shares = sec_history(ticker)
    return frame, currency, notes, shares, "SEC XBRL"


def value_ticker(t: str, clean, beta: float, wrds, fx_rates: dict[str, pd.Series], cfg: dict,
                 rf_usd: float, rf_note: str, as_of: pd.Timestamp, sector: str | None = None) -> dict:
    """DCF de un ticker con los precios ya limpios. Nunca lanza."""
    from ..data import loader  # noqa: PLC0415

    try:
        hist, currency, notes, shares_out, source = history_for(t, wrds)
        if hist is None or len(hist) < 2 or currency is None:
            return {"available": False, "reason": "; ".join(notes) or "no financial statements"}
        # Precio en la moneda de reporte: local -> USD -> moneda de reporte.
        local_close = clean.local_prices(t)[0].dropna()
        usd_per_local = float(clean.fx_rate(t).reindex(local_close.index).ffill().iloc[-1])
        rates = dict(fx_rates)
        if currency != loader.BASE_CURRENCY and currency not in rates:
            rates.update(loader.load_fx([currency], as_of - pd.Timedelta(days=30), as_of)[0])
        usd_per_rep = 1.0 if currency == loader.BASE_CURRENCY else (
            float(rates[currency].loc[:as_of].iloc[-1]) if currency in rates else np.nan)
        if not np.isfinite(usd_per_rep):
            return {"available": False, "reason": f"no {currency}/USD rate to compare the value with the price"}
        price_rep = float(local_close.iloc[-1]) * usd_per_local / usd_per_rep
        rf_cfg = cfg.get("risk_free", {}) or {}
        rf = rf_usd if currency == "USD" else float(rf_cfg.get(currency, rf_usd))
        consensus = None
        sales = getattr(wrds, "ibes_sales", pd.DataFrame()) if wrds is not None else pd.DataFrame()
        if not sales.empty and t in wrds.ids.index and pd.notna(wrds.ids.loc[t, "ibtic"]):
            rows = sales[sales["ticker"] == wrds.ids.loc[t, "ibtic"]]
            consensus = vdata.consensus_growth(rows, hist.index[-1], float(hist["revenue"].iloc[-1]), currency)
        overrides = {str(k).upper(): v for k, v in (cfg.get("overrides", {}) or {}).items()}
        v = value_company(t, hist, currency, price_rep, beta, rf, cfg, shares_out=shares_out, consensus=consensus,
                          overrides=overrides.get(t), usd_per_unit=usd_per_rep, as_of=as_of, data_notes=notes,
                          source=source)
        if not v.get("available"):
            return v
        # Valor en la moneda en que cotiza, para compararlo con la orden.
        to_local = usd_per_rep / usd_per_local
        v["trading_currency"] = clean.currency.get(t, "USD")
        v["price_local"] = float(local_close.iloc[-1])
        v["expected_local"] = v["expected"] * to_local
        v["base_local"] = v["base_value"] * to_local
        v["cases_local"] = {k: r["per_share"] * to_local for k, r in v["scenarios"]["cases"].items()}
        mc = v["monte_carlo"]
        v["mc_local"] = {k: mc[k] * to_local for k in ("p05", "p50", "p95")} if mc.get("n") else {}
        v["rf_note"] = rf_note if currency == "USD" else f"risk-free {rf:.2%} for {currency} (YAML)"
        if sector == "Financials":
            v["flags"].append("financial company: for banks and insurers an FCFF DCF is not meaningful "
                              "(debt is raw material); for asset managers and brokers it is a rough guide")
        return v
    except Exception as exc:  # noqa: BLE001 -- una empresa no tumba el analisis
        log.exception("DCF for %s failed", t)
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def run_valuations(result, wrds, fx_rates: dict[str, pd.Series], as_of: pd.Timestamp,
                   tickers: list[str]) -> dict[str, dict]:
    """DCF de cada nombre de la canasta. Un fallo en uno no para los demas."""
    cfg = result.settings.get("valuation", {}) or {}
    if cfg.get("enabled", True) is False:
        return {}
    rf_usd, rf_note = risk_free_rate(as_of, float((cfg.get("risk_free", {}) or {}).get("USD", 0.0425)))
    out = {}
    for t in tickers:
        beta = float(result.regression.loc[t, "beta"]) if t in result.regression.index else 1.0
        out[t] = value_ticker(t, result.clean, beta, wrds, fx_rates, cfg, rf_usd, rf_note, as_of,
                              result.sectors.get(t))
    return out


def summary(v: dict) -> dict:
    """Lo esencial de un DCF, serializable (JSON) y en la moneda en que cotiza."""
    if not v or not v.get("available"):
        return {"available": False, "reason": (v or {}).get("reason", "not run")}
    c = v["cost_of_capital"]
    rev = v["reverse"]
    mc = v["monte_carlo"]
    base = v["scenarios"]["cases"]["base"]
    return {
        "available": True, "source": v.get("source"), "reporting_currency": v["currency"],
        "trading_currency": v.get("trading_currency", v["currency"]), "fiscal_end": str(pd.Timestamp(v["fiscal_end"]).date()),
        "price": v.get("price_local", v["capital"].price), "value_expected": v.get("expected_local", v["expected"]),
        "value_cases": v.get("cases_local", {k: r["per_share"] for k, r in v["scenarios"]["cases"].items()}),
        "upside_expected": v["expected_upside"], "upside_base": base["upside"], "stance": v["stance"],
        "weights": v["weights"], "critical": list(v.get("critical", [])),
        "wacc": c["wacc"], "cost_of_equity": c["cost_of_equity"], "beta": c["beta"], "rf": c["rf"],
        "rating": c["rating"], "terminal_growth": v["cases"]["base"].terminal_growth,
        "tv_share": base["tv_share"], "terminal_ev_ebitda": base["terminal_ev_ebitda"],
        "mc_prob_above_price": mc.get("prob_above_price"), "mc_range": v.get("mc_local", {}),
        "implied_cagr": rev.get("implied_cagr"), "base_cagr": rev.get("base_cagr"), "reverse_note": rev.get("note"),
        "years": v["cases"]["base"].years, "flags": list(v["flags"]), "notes": list(v["notes"]),
        "consensus": v.get("consensus"),
    }

