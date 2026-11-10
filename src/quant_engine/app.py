"""Punto de entrada del motor: modo interactivo y modo por argumentos.

    python main.py                      -> interactivo: pide tickers y opciones
    python main.py research --long ...  -> una corrida sin preguntas (scripts, CI)

Tras cada analisis se guardan reporte, CSV, JSON, HTML y graficos. En pantalla
sale el resumen, el plan de operacion y la senal final; el informe del plan
(HTML de una pagina) se abre en el navegador si hay una persona delante. Un
menu permite seguir sin reiniciar el programa.
"""

from __future__ import annotations

import io
import logging
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from sfc_tfsig.console import enable_utf8_stdout

from .analysis import AnalysisResult, run_analysis
from .reporting import terminal
from .settings import (
    CONSTRUCTIONS,
    LOOKBACK_YEARS,
    REBALANCES,
    EngineSettings,
    SettingsError,
    build_settings,
    default_values,
    format_positions,
    output_dir,
    parse_positions,
    parse_tickers,
)

MENU = """
============================================================
ANALYSIS COMPLETE
============================================================

Would you like to:

[1] Change the universe
[2] Change portfolio construction
[3] Run robustness analysis
[4] Run historical backtest
[5] Show the full analysis here (13 sections)
[6] Open the trade plan report in the browser
[0] Exit
"""


def _setup_logging(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=directory / "engine.log", level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True,
    )


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [default: {default}]" if default not in (None, "") else ""
    print(f"{prompt}{suffix}:")
    value = input("> ").strip()
    return value if value else (default or "")


def parse_fraction(text: str) -> float:
    """'0.05%' -> 0.0005 ; '0.0005' -> 0.0005 ; '4%' -> 0.04."""
    t = str(text).strip()
    if t.endswith("%"):
        return float(t[:-1]) / 100.0
    return float(t)


def split_long_short(tickers: tuple[str, ...], long_text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Los que el usuario marca como LONG; el resto, SHORT. Vacio = primera mitad LONG."""
    if not long_text.strip():
        half = (len(tickers) + 1) // 2
        return tickers[:half], tickers[half:]
    longs = parse_tickers(long_text)
    unknown = [t for t in longs if t not in tickers]
    if unknown:
        raise SettingsError(f"LONG tickers not in the ticker list: {unknown}")
    return longs, tuple(t for t in tickers if t not in longs)


def ask_settings(previous: EngineSettings | None = None) -> EngineSettings:
    """Pide los tickers, cuales van LONG, y opciones avanzadas solo si se piden.

    El signo LONG/SHORT lo decide research, no el motor: por eso se pregunta.
    """
    # Defaults sin validar: una configuracion de prueba con un ticker por lado
    # no pasa la validacion de topes, y el programa moria antes de preguntar.
    base = previous if previous is not None else SimpleNamespace(**default_values())
    while True:
        try:
            prev_all = ",".join(previous.longs + previous.shorts) if previous else None
            tickers = parse_tickers(_ask("Enter the tickers, separated by commas", prev_all))
            if not tickers:
                raise SettingsError("at least 1 ticker is needed")
            half = (len(tickers) + 1) // 2
            print(f"  (Enter = the first {half} LONG: {','.join(tickers[:half])}. "
                  "All LONG = long-only basket)")
            longs, shorts = split_long_short(tickers, _ask("Which ones go LONG? The rest go SHORT", ""))
            print(f"  LONG:  {' '.join(longs)}\n  SHORT: {' '.join(shorts) if shorts else '(none: long-only)'}")
            print("  (TICKER:SHARES, negative = short. E.g. AAPL:500,TSLA:-200. Enter = none)")
            prev_pos = format_positions(previous.positions) if previous else ""
            positions = parse_positions(_ask("Current positions", prev_pos))
            if positions:
                print("  NOW: " + "  ".join(f"{t} {q:+,}" for t, q in positions))

            options = {}
            if _ask("Change advanced options? (y/N)", "n").lower().startswith(("y", "s")):
                options = dict(
                    benchmark=_ask("Benchmark", base.benchmark),
                    lookback=_ask(f"Lookback {list(LOOKBACK_YEARS)}", base.lookback),
                    risk_free_rate=parse_fraction(_ask("Risk-free rate", str(base.risk_free_rate))),
                    rebalance=_ask(f"Rebalance {list(REBALANCES)}", base.rebalance),
                    construction=_ask(f"Construction {list(CONSTRUCTIONS)}", base.construction),
                    initial_capital=float(_ask("Capital", f"{base.initial_capital:.0f}").replace(",", "")),
                    transaction_cost=parse_fraction(_ask("Transaction cost", f"{base.transaction_cost:.2%}")),
                )
            elif previous is not None:
                options = dict(benchmark=previous.benchmark, lookback=previous.lookback,
                               risk_free_rate=previous.risk_free_rate, rebalance=previous.rebalance,
                               construction=previous.construction, initial_capital=previous.initial_capital,
                               transaction_cost=previous.transaction_cost)
            return build_settings(longs, shorts, positions=positions, **options)
        except (SettingsError, ValueError) as exc:
            print(f"\n  ERROR: {exc}\n  Try again.\n")


# Ancho del reporte en disco: independiente de la terminal de quien lo corre.
REPORT_WIDTH = 120


def execute(settings: EngineSettings, console: Console, *, signals_csv: Path | None = None,
            robustness: bool = True, charts: bool = True,
            full: bool = False) -> tuple[AnalysisResult, dict[str, Path]]:
    """Corre, guarda el reporte completo y muestra en pantalla el plan.

    En pantalla va el resumen, el plan y la senal final; las trece secciones de
    analisis van a disco (y al menu [5]). Con `full` se imprime todo.
    """
    print("\nRunning quantitative research...", flush=True)
    result = run_analysis(settings, signals_csv=signals_csv, run_robustness=robustness,
                          progress=lambda m: print(f"  - {m}", flush=True))
    paths = save(result, charts=charts)
    (terminal.render if full else terminal.brief)(result, console)
    _print_saved(console, paths)
    return result, paths


def save(result: AnalysisResult, *, charts: bool = True) -> dict[str, Path]:
    """Reporte completo grabado sin imprimir, y todos los ficheros."""
    from .reporting import export

    report = Console(record=True, file=io.StringIO(), width=REPORT_WIDTH)
    terminal.render(result, report)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = output_dir(result.settings)
    chart_paths = []
    if charts and result.settings.get("output.charts", True):
        from .reporting import charts as charts_mod

        chart_paths = charts_mod.render_all(result, directory / "charts", stamp)
    return export.export_all(result, report, directory, stamp, chart_paths)


def _print_saved(console: Console, paths: dict[str, Path]) -> None:
    if not paths:
        return
    directory = next(iter(paths.values())).parent
    console.print()
    console.print(f"Files in {directory}:")
    labels = {"trade_plan_html": "TRADE PLAN REPORT", "orders": "orders (Excel)", "html": "full report",
              "txt": "full report (text)"}
    for kind, label in labels.items():
        if kind in paths:
            console.print(f"  {label:20} {paths[kind].name}")
    for kind, path in paths.items():
        if kind.startswith("dcf_"):
            console.print(f"  {'DCF model (Excel)':20} dcf/{path.name}")
    console.print("  + JSON, CSVs (portfolio, signals, risk, correlation, trade plan) and charts/")


def _open(paths: dict[str, Path]) -> None:
    path = paths.get("trade_plan_html")
    if path is not None:
        webbrowser.open(path.resolve().as_uri())


def interactive() -> int:
    enable_utf8_stdout()
    # La pantalla no se graba: lo que va a disco sale de su propia consola en `save`.
    console = Console()
    print("=" * 60)
    print("        QUANTITATIVE HEDGE FUND RESEARCH ENGINE")
    print("=" * 60)
    print()
    try:
        settings = ask_settings()
    except EOFError:
        return 0
    _setup_logging(output_dir(settings))
    result, paths = execute(settings, console)
    # Solo con una persona delante: redirigido o en un script no se abre nada.
    auto_open = sys.stdin.isatty() and bool(settings.get("output.open_trade_plan", True))
    if auto_open:
        _open(paths)

    while True:
        print(MENU)
        try:
            choice = input("Selection:\n> ").strip()
        except EOFError:
            return 0
        try:
            if choice == "1":
                settings = ask_settings(settings)
                result, paths = execute(settings, console)
                if auto_open:
                    _open(paths)
            elif choice == "2":
                method = _ask(f"Portfolio construction {list(CONSTRUCTIONS)}", settings.construction)
                settings = settings.with_changes(construction=method)
                result, paths = execute(settings, console)
                if auto_open:
                    _open(paths)
            elif choice == "3":
                from .robustness import run_robustness

                result.robustness = run_robustness(result)
                terminal.robustness(console, result)
                paths = save(result, charts=False)
                _print_saved(console, paths)
            elif choice == "4":
                path = Path(_ask("Path to historical signals CSV (date,ticker,signal)"))
                if not path.exists():
                    print(f"  ERROR: {path} not found")
                    continue
                result, paths = execute(settings, console, signals_csv=path)
            elif choice == "5":
                terminal.render(result, console)
            elif choice == "6":
                _open(paths)
                print(f"  {paths.get('trade_plan_html')}")
            elif choice in ("0", "q", "salir", "exit"):
                return 0
            else:
                print("  Choose 0-6.")
        except (SettingsError, ValueError, RuntimeError) as exc:
            print(f"\n  ERROR: {exc}\n")
        except EOFError:
            return 0


def run_once(args) -> int:
    """Una corrida con argumentos, sin preguntas."""
    enable_utf8_stdout()
    settings = build_settings(
        args.long, args.short,
        benchmark=args.benchmark, lookback=args.lookback, rebalance=args.rebalance,
        construction=args.construction,
        risk_free_rate=parse_fraction(args.rf) if args.rf else None,
        initial_capital=float(args.capital) if args.capital else None,
        transaction_cost=parse_fraction(args.tc) if args.tc else None,
        positions=args.positions,
    )
    _setup_logging(output_dir(settings))
    console = Console(width=args.width)
    _, paths = execute(settings, console, signals_csv=Path(args.signals) if args.signals else None,
                       robustness=not args.no_robustness, charts=not args.no_charts, full=args.full)
    if args.open:
        _open(paths)
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("research", help="long/short engine on research lists (no prompts)")
    p.add_argument("--long", required=True, help="LONG tickers, comma-separated")
    p.add_argument("--short", default="", help="SHORT tickers, comma-separated; without them, a long-only basket")
    p.add_argument("--benchmark")
    p.add_argument("--lookback", choices=list(LOOKBACK_YEARS))
    p.add_argument("--rebalance", choices=list(REBALANCES))
    p.add_argument("--construction", choices=list(CONSTRUCTIONS))
    p.add_argument("--capital")
    p.add_argument("--tc", help="transaction cost: 0.0005 or 0.05%%")
    p.add_argument("--rf", help="risk-free rate: 0.04 or 4%%")
    p.add_argument("--positions", help="current positions: AAPL:500,TSLA:-200 (negative = short)")
    p.add_argument("--signals", help="CSV date,ticker,signal for mode B")
    p.add_argument("--no-robustness", action="store_true")
    p.add_argument("--no-charts", action="store_true")
    p.add_argument("--full", action="store_true", help="print all 13 sections, not only the plan")
    p.add_argument("--open", action="store_true", help="open the trade plan report in the browser")
    p.add_argument("--width", type=int, default=160, help="output width")
    p.set_defaults(func=run_once)


if __name__ == "__main__":
    sys.exit(interactive())
