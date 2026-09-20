"""Interfaz de linea de comandos: los cuatro comandos del modelo.

    python main.py universo     -- construye y describe el universo investible
    python main.py panel        -- descarga datos y ensambla el panel (lo lento)
    python main.py backtest     -- corre la estrategia y escribe el reporte
    python main.py ordenes      -- genera el rebalanceo del mes para ejecutar

`panel` cachea todo lo que descarga, asi que `backtest` y `ordenes` se pueden
repetir sin volver a tocar la red. La primera corrida tarda; las siguientes son
minutos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import panel as panel_mod, report as report_mod, validation as validation_mod
from .backtest import benchmark_nav, run_backtest
from .config import Config, ConfigError, load_config
from .console import enable_utf8_stdout
from .data import cache, prices as prices_mod
from .factors.composite import build_scores
from .metrics import evaluate
from .orders import build_orders, load_positions, render_orders, save_orders
from .paths import CACHE_DIR, ensure_dirs
from .portfolio import build_portfolio
from .universe import build_universe, summarize

PANEL_CACHE = "panel_scored"


def _load(config_path: str | None) -> Config:
    try:
        return load_config(config_path)
    except ConfigError as exc:
        print(f"ERROR de configuracion: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _require_panel(cfg: Config, *, rebuild: bool, progress: bool = True) -> pd.DataFrame:
    """Panel puntuado, de cache o reconstruido.

    La cache guarda el fingerprint de la configuracion con la que se construyo.
    Si no coincide, se reconstruye: usar un panel calculado con otros filtros de
    universo y creer que corresponde a la configuracion actual es como se
    producen los resultados que nadie puede reproducir.
    """
    if not rebuild:
        cached = cache.read_frame(PANEL_CACHE)
        if cached is not None and cached.attrs.get("fingerprint") in (None, cfg.fingerprint):
            stamp = cache.read_json(f"{PANEL_CACHE}_meta") or {}
            if stamp.get("fingerprint") == cfg.fingerprint:
                cached["date"] = pd.to_datetime(cached["date"])
                print(f"panel en cache: {len(cached):,} filas "
                      f"({stamp.get('dates', '?')} fechas, fingerprint {cfg.fingerprint})")
                return cached
            print("la cache del panel es de otra configuracion; se reconstruye")

    raw = panel_mod.build_panel(cfg, progress=progress)
    scored = build_scores(raw, cfg)
    cache.write_frame(PANEL_CACHE, scored)
    cache.write_json(
        f"{PANEL_CACHE}_meta",
        {"fingerprint": cfg.fingerprint, "rows": len(scored),
         "dates": int(scored["date"].nunique())},
    )
    return scored


def _close_prices(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Cierres diarios de los tickers del panel mas el benchmark."""
    tickers = sorted(set(scored["ticker"]) | {str(cfg.get("backtest.benchmark"))})
    start = scored["date"].min() - pd.DateOffset(months=2)
    end = pd.Timestamp.today().normalize()
    long = prices_mod.get_prices(tickers, start, end, progress=False)
    return prices_mod.to_wide(long, "close")


# ---------------------------------------------------------------------------
#  Comandos
# ---------------------------------------------------------------------------


def cmd_universo(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    universe = build_universe(cfg, with_profiles=not args.rapido)
    summary = summarize(universe)

    print(f"\nUniverso estatico: {summary['n_names']:,} empresas")
    print(f"Configuracion: {cfg.fingerprint}\n")
    if not args.rapido:
        print("Por pais:")
        for country, count in summary["by_country"].items():
            print(f"  {country:<8} {count:>5}")
        print("\nPor sector:")
        for sector, count in sorted(summary["by_sector"].items(), key=lambda kv: -kv[1]):
            print(f"  {sector:<24} {count:>5}")
    print(f"\nPrimeras 15:\n{universe.head(15).to_string(index=False)}")
    return 0


def cmd_panel(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    scored = _require_panel(cfg, rebuild=True)

    print(f"\nPanel: {len(scored):,} filas, {scored['date'].nunique()} fechas, "
          f"{scored['ticker'].nunique():,} tickers")
    print(f"Rango: {scored['date'].min().date()} a {scored['date'].max().date()}")
    por_fecha = scored.groupby("date").size()
    print(f"Nombres por fecha: min {por_fecha.min()}, mediana {int(por_fecha.median())}, "
          f"max {por_fecha.max()}")

    print("\nCobertura de metricas (fraccion del panel con dato):")
    coverage = panel_mod.coverage_report(scored)
    print(coverage.to_string(index=False))
    print("\nUna metrica con cobertura baja no mide lo que dice medir: mide "
          "'empresas que reportan esa etiqueta'.")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    scored = _require_panel(cfg, rebuild=args.rebuild)
    close = _close_prices(scored, cfg)

    print("\ncorriendo backtest...")
    result = run_backtest(scored, close, cfg)

    bench = benchmark_nav(close, str(cfg.get("backtest.benchmark")), result.nav.index,
                          float(cfg.get("backtest.initial_capital")))
    perf = evaluate(result.nav, bench, risk_free=float(cfg.get("backtest.risk_free_rate")))

    validation = None
    if not args.sin_validacion:
        print("validando (IC, quintiles, Fama-MacBeth, walk-forward)...")
        with_forward = panel_mod.add_forward_returns(scored, close)
        validation = validation_mod.full_report(with_forward, cfg)

    text = report_mod.build_report(result, perf, cfg, benchmark_nav=bench,
                                   validation=validation)
    path = report_mod.save_report(text)
    artifacts = report_mod.save_artifacts(result)
    chart = report_mod.save_charts(result, bench) if cfg.get("reporting.charts") else None

    print(text)
    print(f"\nreporte: {path}")
    for label, artifact in artifacts.items():
        print(f"{label}: {artifact}")
    if chart:
        print(f"grafico: {chart}")
    return 0


def cmd_ordenes(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    scored = _require_panel(cfg, rebuild=args.rebuild)

    last_date = scored["date"].max()
    cross_section = scored[scored["date"] == last_date]
    current = load_positions()

    nav = args.capital
    if nav is None:
        if current.empty:
            nav = float(cfg.get("backtest.initial_capital"))
            print(f"sin posiciones previas: se usa el capital inicial {nav:,.0f}")
        else:
            close = _close_prices(scored, cfg)
            latest = close.ffill().iloc[-1]
            nav = float((current * latest.reindex(current.index)).sum())
            print(f"valor de las posiciones actuales: {nav:,.2f} "
                  "(pasa --capital para incluir la caja)")

    target = build_portfolio(cross_section, cfg, held=set(current.index))
    orders = build_orders(target, current, nav, cfg)
    text = render_orders(orders, target, nav, cfg, last_date)

    print(text)
    paths = save_orders(orders, text, last_date)
    print(f"\nordenes: {paths['csv']}")
    print(f"hoja: {paths['markdown']}")
    print("\nTras ejecutar, actualiza data/positions.csv con las acciones reales.")
    return 0


def cmd_limpiar(args: argparse.Namespace) -> int:
    removed = cache.clear(args.prefijo or "")
    print(f"{removed} ficheros borrados de {CACHE_DIR}")
    return 0


# ---------------------------------------------------------------------------
#  Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sfc-tfsig",
        description="Modelo multifactor long-only US+Canada (SFC x TFSIG)",
    )
    parser.add_argument("--config", help="ruta de un TOML alternativo")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_universo = sub.add_parser("universo", help="construye y describe el universo")
    p_universo.add_argument("--rapido", action="store_true",
                            help="omite los perfiles de EDGAR (sin sector ni pais)")
    p_universo.set_defaults(func=cmd_universo)

    p_panel = sub.add_parser("panel", help="descarga datos y ensambla el panel")
    p_panel.set_defaults(func=cmd_panel)

    p_backtest = sub.add_parser("backtest", help="corre la estrategia y escribe el reporte")
    p_backtest.add_argument("--rebuild", action="store_true", help="reconstruye el panel")
    p_backtest.add_argument("--sin-validacion", action="store_true",
                            help="omite los contrastes estadisticos (mas rapido)")
    p_backtest.set_defaults(func=cmd_backtest)

    p_ordenes = sub.add_parser("ordenes", help="genera el rebalanceo del mes")
    p_ordenes.add_argument("--rebuild", action="store_true", help="reconstruye el panel")
    p_ordenes.add_argument("--capital", type=float,
                           help="valor total de la cuenta, incluida la caja")
    p_ordenes.set_defaults(func=cmd_ordenes)

    p_limpiar = sub.add_parser("limpiar", help="borra la cache de datos")
    p_limpiar.add_argument("--prefijo", help="borra solo lo que empiece por este prefijo")
    p_limpiar.set_defaults(func=cmd_limpiar)

    return parser


def main(argv: list[str] | None = None) -> int:
    enable_utf8_stdout()
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrumpido", file=sys.stderr)
        return 130
    except (ConfigError, RuntimeError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
