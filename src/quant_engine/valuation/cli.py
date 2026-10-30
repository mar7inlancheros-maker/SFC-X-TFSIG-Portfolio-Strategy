"""`python main.py dcf PLTR AAPL`: el DCF automatico sin correr todo el motor.

Descarga tres anos de precios (para la beta frente al benchmark), los estados
financieros (SEC; Compustat con `--wrds`) y el bono a 10 anos, valora cada
ticker, imprime el resumen y escribe un Excel con formulas vivas por ticker en
`output/quant_engine/dcf/`.
"""

from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import pandas as pd
from rich.console import Console
from rich.markup import escape

from sfc_tfsig.console import enable_utf8_stdout


def _print(console: Console, t: str, v: dict) -> None:
    from .engine import summary  # noqa: PLC0415

    console.print(f"[bold]{escape(t)}[/]")
    if not v.get("available"):
        console.print(f"  DCF not available: {escape(str(v.get('reason')))}")
        return
    sm = summary(v)
    c = sm["value_cases"]
    ccy = sm["trading_currency"]
    console.print(f"  Price {sm['price']:,.2f} {ccy} | bear {c['bear']:,.2f} | base {c['base']:,.2f} "
                  f"({sm['upside_base']:+.0%}) | bull {c['bull']:,.2f} | weighted {sm['value_expected']:,.2f} "
                  f"-> {sm['stance']}")
    coc = v["cost_of_capital"]
    console.print(f"  WACC {coc['wacc']:.2%} (Ke {coc['cost_of_equity']:.2%}, beta {coc['beta']:.2f} Blume, "
                  f"Kd {coc['cost_of_debt']:.2%} {coc['rating']}) | terminal g {sm['terminal_growth']:.1%} | "
                  + (f"TV {sm['tv_share']:.0%} of EV" if np.isfinite(sm["tv_share"]) else "EV not positive")
                  + f" | {escape(v.get('rf_note', ''))}")
    if sm.get("implied_cagr") is not None:
        console.print(f"  The price implies {sm['implied_cagr']:+.1%} revenue growth a year for {sm['years']} "
                      f"years (base case {sm['base_cagr']:+.1%}).")
    if sm.get("mc_prob_above_price") is not None:
        r = sm["mc_range"]
        console.print(f"  Monte Carlo: 5%-95% {r['p05']:,.2f} - {r['p95']:,.2f}; value above price in "
                      f"{sm['mc_prob_above_price']:.0%} of draws.")
    for n in sm["notes"]:
        console.print(f"  - {escape(n)}")
    for f in sm["flags"]:
        console.print(f"  [yellow]! {escape(f)}[/]")


def run(args) -> int:
    enable_utf8_stdout()
    from sfc_tfsig.console import keep_awake  # noqa: PLC0415

    with keep_awake():
        return _run(args)


def _run(args) -> int:
    from ..analysis import _load_fx  # noqa: PLC0415
    from ..data import cleaning, loader  # noqa: PLC0415
    from ..data import wrds_data as wd  # noqa: PLC0415
    from ..factors import beta as beta_mod  # noqa: PLC0415
    from ..settings import DEFAULT_YAML, load_yaml  # noqa: PLC0415
    from . import excel  # noqa: PLC0415
    from .engine import risk_free_rate, value_ticker  # noqa: PLC0415

    console = Console(width=args.width)
    raw = load_yaml(DEFAULT_YAML)
    cfg = raw.get("valuation", {}) or {}
    bench = (raw.get("defaults", {}) or {}).get("benchmark", "SPY")
    tickers = [t.strip().upper() for t in args.tickers if t.strip()]
    as_of = pd.Timestamp.today().normalize()
    start = as_of - pd.DateOffset(years=3)
    print("Running automated DCF...", flush=True)
    market = loader.load_ohlcv([*tickers, bench], start, as_of)
    warnings: list[str] = []
    fx, currency, fx_rates = _load_fx([*tickers, bench], start, as_of, warnings, warn_for=set(tickers))
    clean = cleaning.align(market, bench, fx=fx, currency=currency)
    wrds = wd.load(tickers, start, as_of) if args.wrds else None
    if wrds is not None and not wrds.available:
        warnings.append(f"WRDS unavailable, SEC filings used instead: {wrds.reason}")
    rf, rf_note = risk_free_rate(as_of, float((cfg.get("risk_free", {}) or {}).get("USD", 0.0425)))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    from ..settings import build_settings, output_dir  # noqa: PLC0415

    directory = output_dir(build_settings(tuple(tickers[:1]), ())) / "dcf"
    for w in warnings:
        console.print(f"[yellow]! {escape(w)}[/]")
    for t in tickers:
        if t not in clean.close.columns:
            console.print(f"[bold]{escape(t)}[/]\n  no prices: {escape(market.failures.get(t, 'unknown'))}")
            continue
        r = clean.returns[[t, bench]].dropna().iloc[-756:]
        beta = float(args.beta) if args.beta else float(beta_mod.regression(r[t], r[bench], rf).get("beta", 1.0))
        v = value_ticker(t, clean, beta, wrds, fx_rates, cfg, rf, rf_note, as_of)
        _print(console, t, v)
        if v.get("available") and not args.no_excel:
            path = excel.write(v, directory / f"DCF_{t.replace('.', '_')}_{stamp}.xlsx")
            console.print(f"  Excel model: {path}")
        console.print()
    console.print("Automated DCF: a starting point for the analyst, not a recommendation. Change any assumption "
                  "in the Excel (blue cells) or in valuation.overrides of config/quant_engine.yaml.")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("dcf", help="DCF automatico de uno o varios tickers, con Excel de formulas vivas")
    p.add_argument("tickers", nargs="+", help="tickers de Yahoo: PLTR AAPL ATZ.TO")
    p.add_argument("--beta", help="beta a usar en vez de la de la regresion")
    p.add_argument("--wrds", action="store_true", help="Compustat y consenso de IBES (pide aprobacion Duo)")
    p.add_argument("--no-excel", action="store_true")
    p.add_argument("--width", type=int, default=140)
    p.set_defaults(func=run)


if __name__ == "__main__":
    sys.exit(0)
