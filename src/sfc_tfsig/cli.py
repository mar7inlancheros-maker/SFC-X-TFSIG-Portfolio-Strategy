"""Interfaz de linea de comandos: los comandos del modelo.

    python main.py universo     -- construye y describe el universo investible
    python main.py panel        -- descarga datos y ensambla el panel (lo lento)
    python main.py backtest     -- corre la estrategia y escribe el reporte
    python main.py ordenes      -- genera el rebalanceo del mes para ejecutar
    python main.py riesgo       -- VaR, Monte Carlo, estres y limites de la cartera

`panel` cachea todo lo que descarga, asi que `backtest` y `ordenes` se pueden
repetir sin volver a tocar la red. La primera corrida tarda; las siguientes son
minutos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import attribution as attribution_mod, panel as panel_mod, report as report_mod, validation as validation_mod
from . import risk_report as risk_report_mod
from .backtest import benchmark_nav, run_backtest
from .config import Config, ConfigError, load_config, load_risk_config
from .console import enable_utf8_stdout
from .data import cache, prices as prices_mod
from .factors.composite import build_scores
from .metrics import evaluate
from .orders import build_orders, load_positions, render_orders, save_orders
from .paths import CACHE_DIR, ensure_dirs
from .portfolio import build_portfolio
from .risk import analysis as risk_mod
from .universe import build_universe, summarize

PANEL_CACHE = "panel_scored"


def _load(config_path: str | None) -> Config:
    try:
        return load_config(config_path)
    except ConfigError as exc:
        print(f"ERROR de configuracion: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _panel_cache_name(cfg: Config) -> str:
    # Fingerprint de la configuracion Y version de la logica del panel: el
    # primero cambia con el TOML, la segunda con el codigo. Hacen falta los dos.
    return f"{PANEL_CACHE}_{panel_mod.PANEL_VERSION}_{cfg.fingerprint}"


def _require_panel(cfg: Config, *, rebuild: bool, progress: bool = True) -> pd.DataFrame:
    """Panel puntuado, de cache o reconstruido.

    La cache lleva el fingerprint de la configuracion EN EL NOMBRE. Antes habia
    un unico panel en cache y se comprobaba el fingerprint al leerlo: correr la
    configuracion trimestral pisaba el panel mensual, volver a la mensual lo
    reconstruia, y comparar dos configuraciones costaba dos reconstrucciones
    completas cada vez. Con un panel por configuracion conviven todos, y sigue
    siendo imposible mezclar el panel de una con el backtest de otra.
    """
    name = _panel_cache_name(cfg)
    if not rebuild:
        cached = cache.read_frame(name)
        if cached is not None:
            cached["date"] = pd.to_datetime(cached["date"])
            print(f"panel en cache: {len(cached):,} filas, "
                  f"{cached['date'].nunique()} fechas (fingerprint {cfg.fingerprint})")
            return cached

    raw = panel_mod.build_panel(cfg, progress=progress)
    scored = build_scores(raw, cfg)
    cache.write_frame(name, scored)
    return scored


def _close_prices(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Cierres diarios de los tickers del panel mas el benchmark."""
    return _price_frames(scored, cfg)[0]


def _price_frames(scored: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cierres y volumenes diarios de los tickers del panel mas el benchmark.

    Una sola lectura de la cache maestra para los dos: son 150 MB y leerla dos
    veces por comando se nota.
    """
    tickers = sorted(set(scored["ticker"]) | {str(cfg.get("backtest.benchmark"))})
    start = scored["date"].min() - pd.DateOffset(months=2)
    end = pd.Timestamp.today().normalize()
    long = prices_mod.get_prices(tickers, start, end, progress=False)
    return prices_mod.to_wide(long, "close"), prices_mod.to_wide(long, "volume")


def _load_risk(path: str | None) -> Config:
    """Politica de riesgo validada, incluidos sectores de escenarios y limites."""
    try:
        risk_cfg = load_risk_config(path)
        risk_mod.validate_policy(risk_cfg)
        return risk_cfg
    except ConfigError as exc:
        print(f"ERROR en la politica de riesgo: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _stress_prices(tickers: list[str], risk_cfg: Config) -> pd.DataFrame | None:
    """Precios largos (desde `stress.history_start`) solo para el estres historico.

    Es la unica descarga del comando: las posiciones actuales y el benchmark
    antes del inicio del panel. Si falla la red, el estres historico sigue con
    los precios que ya hay y los episodios antiguos salen sin datos.
    """
    start = pd.Timestamp(risk_cfg.get("stress.history_start"))
    end = pd.Timestamp.today().normalize()
    try:
        long = prices_mod.get_prices(tickers, start, end, progress=False)
    except Exception as exc:  # noqa: BLE001 - la red no debe tumbar el reporte
        print(f"  aviso: sin historia larga para el estres ({exc}); se usa la del panel")
        return None
    return prices_mod.to_wide(long, "close")


def _scores(scored: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores de factor de la seccion cruzada de `date`: (por ticker, universo)."""
    columns = [c for c in scored.columns if c.startswith("score_") and c != "score_composite"]
    cross = scored[scored["date"] == date]
    return cross.set_index("ticker")[columns], cross[columns]


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

    print("atribucion por nombre, sector y ano...")
    contributions = attribution_mod.position_contributions(
        result.holdings, close, result.rebalances
    )
    yearly_excess = (
        attribution_mod.excess_decomposition(contributions, result.nav, bench)
        if bench is not None else None
    )

    validation = decay_lag = decay_horizon = None
    if not args.sin_validacion:
        print("validando (IC, quintiles, Fama-MacBeth, walk-forward)...")
        with_forward = panel_mod.add_forward_returns(scored, close)
        validation = validation_mod.full_report(with_forward, cfg)
        print("midiendo el decaimiento de la senal...")
        decay_lag = validation_mod.ic_by_lag(scored, close)
        decay_horizon = validation_mod.ic_by_horizon(scored, close)

    text = report_mod.build_report(
        result, perf, cfg, benchmark_nav=bench, validation=validation,
        contributions=contributions, yearly_excess=yearly_excess,
        decay_lag=decay_lag, decay_horizon=decay_horizon,
    )
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

    close = volume = None
    nav = args.capital
    if nav is None:
        if current.empty:
            nav = float(cfg.get("backtest.initial_capital"))
            print(f"sin posiciones previas: se usa el capital inicial {nav:,.0f}")
        else:
            close, volume = _price_frames(scored, cfg)
            latest = close.ffill().iloc[-1]
            nav = float((current * latest.reindex(current.index)).sum())
            print(f"valor de las posiciones actuales: {nav:,.2f} "
                  "(pasa --capital para incluir la caja)")

    target = build_portfolio(cross_section, cfg, held=set(current.index))
    orders = build_orders(target, current, nav, cfg)
    text = render_orders(orders, target, nav, cfg, last_date)

    if not args.sin_riesgo and not target.empty:
        # Antes de operar, no despues: si la cartera propuesta excede un limite,
        # el comite lo sabe cuando todavia puede no aprobarla.
        risk_cfg = _load_risk(args.riesgo_config)
        if close is None:
            close, volume = _price_frames(scored, cfg)
        weights, sectors = risk_mod.book_from_target(target)
        holding_scores, universe_scores = _scores(scored, last_date)
        pre_trade = risk_mod.analyze_book(
            weights, sectors, nav, close, str(cfg.get("backtest.benchmark")), risk_cfg,
            as_of=close.index.max(), volume_wide=volume,
            factor_scores=holding_scores, universe_scores=universe_scores,
            strategy_fingerprint=cfg.fingerprint,
        )
        text = text + "\n" + risk_report_mod.render_pre_trade(pre_trade)

    print(text)
    paths = save_orders(orders, text, last_date)
    print(f"\nordenes: {paths['csv']}")
    print(f"hoja: {paths['markdown']}")
    print("\nTras ejecutar, actualiza data/positions.csv con las acciones reales.")
    return 0


def cmd_riesgo(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    risk_cfg = _load_risk(args.riesgo_config)
    if args.simulaciones:
        risk_cfg = risk_cfg.replace(**{"montecarlo.bootstrap.n_paths": int(args.simulaciones)})
    benchmark = str(cfg.get("backtest.benchmark"))

    scored = _require_panel(cfg, rebuild=args.rebuild)
    close, volume = _price_frames(scored, cfg)

    print("\ncorriendo backtest para la historia de la estrategia...")
    result = run_backtest(scored, close, cfg)
    bench = benchmark_nav(close, benchmark, result.nav.index,
                          float(cfg.get("backtest.initial_capital")))

    weights, sectors, nav, as_of = risk_mod.book_from_backtest(result.holdings, close, result.nav)
    stress_close = None
    if not args.sin_red:
        print("precios largos para el estres historico...")
        stress_close = _stress_prices(sorted(set(weights.index) | {benchmark}), risk_cfg)

    holding_scores, universe_scores = _scores(scored, scored["date"].max())
    print("midiendo riesgo (VaR, Monte Carlo, estres, limites)...")
    analysis = risk_mod.analyze(
        weights, sectors, nav, close, benchmark, risk_cfg,
        strategy_nav=result.nav, benchmark_nav=bench, as_of=as_of,
        volume_wide=volume, stress_close=stress_close,
        factor_scores=holding_scores, universe_scores=universe_scores,
        strategy_fingerprint=cfg.fingerprint,
    )

    text = risk_report_mod.build_risk_report(analysis, cfg, risk_cfg)
    path = risk_report_mod.save_risk_report(text)
    artifacts = risk_report_mod.save_risk_artifacts(analysis)
    chart = risk_report_mod.save_risk_charts(analysis) if cfg.get("reporting.charts") else None

    print(text)
    print(f"\nreporte de riesgo: {path}")
    for label, artifact in artifacts.items():
        print(f"{label}: {artifact}")
    if chart:
        print(f"graficos: {chart}")
    # Codigo de salida propio si hay un limite excedido y se pidio: permite
    # encadenar el comando en un script que avise sin leer el reporte.
    return 3 if args.estricto and analysis.status == "EXCEDIDO" else 0


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
    parser.add_argument("--riesgo-config", help="ruta de una politica de riesgo alternativa")
    # Sin subcomando, `python main.py` abre el motor long/short interactivo
    # (`quant_engine`). Los subcomandos del modelo multifactor no cambian.
    sub = parser.add_subparsers(dest="comando", required=False)

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
    p_ordenes.add_argument("--sin-riesgo", action="store_true",
                           help="omite la comprobacion de riesgo de la cartera objetivo")
    p_ordenes.set_defaults(func=cmd_ordenes)

    p_riesgo = sub.add_parser("riesgo", help="VaR, Monte Carlo, estres y limites")
    p_riesgo.add_argument("--rebuild", action="store_true", help="reconstruye el panel")
    p_riesgo.add_argument("--simulaciones", type=int,
                          help="trayectorias del bootstrap (por defecto, las del TOML)")
    p_riesgo.add_argument("--sin-red", action="store_true",
                          help="no descarga historia larga; el estres usa los precios del panel")
    p_riesgo.add_argument("--estricto", action="store_true",
                          help="sale con codigo 3 si algun limite esta EXCEDIDO")
    p_riesgo.set_defaults(func=cmd_riesgo)

    p_limpiar = sub.add_parser("limpiar", help="borra la cache de datos")
    p_limpiar.add_argument("--prefijo", help="borra solo lo que empiece por este prefijo")
    p_limpiar.set_defaults(func=cmd_limpiar)

    from quant_engine.app import add_parser as add_research_parser

    add_research_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    enable_utf8_stdout()
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.comando is None:
            from quant_engine.app import interactive

            return interactive()
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrumpido", file=sys.stderr)
        return 130
    except (ConfigError, RuntimeError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
