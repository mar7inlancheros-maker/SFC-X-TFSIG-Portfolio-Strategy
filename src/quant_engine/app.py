"""Punto de entrada del motor: modo interactivo y modo por argumentos.

    python main.py                      -> interactivo: pide tickers y opciones
    python main.py research --long ...  -> una corrida sin preguntas (scripts, CI)

Tras cada analisis se guardan reporte, CSV, JSON, HTML y graficos, y se ofrece
un menu para seguir sin reiniciar el programa.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from sfc_tfsig.console import enable_utf8_stdout

from .analysis import AnalysisResult, run_analysis
from .reporting import terminal
from .settings import (CONSTRUCTIONS, LOOKBACK_YEARS, REBALANCES, EngineSettings, SettingsError,
                       build_settings, default_values, output_dir)

MENU = """
============================================================
ANALYSIS COMPLETE
============================================================

Would you like to:

[1] Change the universe
[2] Change portfolio construction
[3] Run robustness analysis
[4] Run historical backtest
[5] Export full report
[6] Exit
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


def ask_settings(previous: EngineSettings | None = None) -> EngineSettings:
    # Defaults sin validar: una configuracion de prueba con un ticker por lado
    # no pasa la validacion de topes, y el programa moria antes de preguntar.
    base = previous if previous is not None else SimpleNamespace(**default_values())
    while True:
        try:
            longs = _ask("Enter LONG tickers", ",".join(previous.longs) if previous else None)
            shorts = _ask("Enter SHORT tickers", ",".join(previous.shorts) if previous else None)
            return build_settings(
                longs, shorts,
                benchmark=_ask("Enter benchmark", base.benchmark),
                lookback=_ask(f"Analysis period {list(LOOKBACK_YEARS)}", base.lookback),
                risk_free_rate=parse_fraction(_ask("Risk-free rate", str(base.risk_free_rate))),
                rebalance=_ask(f"Rebalancing frequency {list(REBALANCES)}", base.rebalance),
                construction=_ask(f"Portfolio construction {list(CONSTRUCTIONS)}", base.construction),
                initial_capital=float(_ask("Initial capital", f"{base.initial_capital:.0f}").replace(",", "")),
                transaction_cost=parse_fraction(_ask("Transaction cost", f"{base.transaction_cost:.2%}")),
            )
        except (SettingsError, ValueError) as exc:
            print(f"\n  ERROR: {exc}\n  Try again.\n")


def execute(settings: EngineSettings, console: Console, *, signals_csv: Path | None = None,
            robustness: bool = True, charts: bool = True) -> tuple[AnalysisResult, dict[str, Path]]:
    print("\nRunning quantitative research...", flush=True)
    result = run_analysis(settings, signals_csv=signals_csv, run_robustness=robustness,
                          progress=lambda m: print(f"  - {m}", flush=True))
    terminal.render(result, console)
    return result, save(result, console, charts=charts)


def save(result: AnalysisResult, console: Console, *, charts: bool = True) -> dict[str, Path]:
    from .reporting import export

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = output_dir(result.settings)
    chart_paths = []
    if charts and result.settings.get("output.charts", True):
        from .reporting import charts as charts_mod

        chart_paths = charts_mod.render_all(result, directory / "charts", stamp)
    paths = export.export_all(result, console, directory, stamp, chart_paths)
    console.print()
    for kind, path in paths.items():
        console.print(f"saved {kind:12} {path}")
    if chart_paths:
        console.print(f"saved {'charts':12} {directory / 'charts'} ({len(chart_paths)} files)")
    return paths


def interactive() -> int:
    enable_utf8_stdout()
    console = Console(record=True)
    print("=" * 60)
    print("        QUANTITATIVE HEDGE FUND RESEARCH ENGINE")
    print("=" * 60)
    print()
    settings = ask_settings()
    _setup_logging(output_dir(settings))
    result, _ = execute(settings, console)

    while True:
        print(MENU)
        choice = input("Selection:\n> ").strip()
        try:
            if choice == "1":
                console = Console(record=True)
                settings = ask_settings(settings)
                result, _ = execute(settings, console)
            elif choice == "2":
                method = _ask(f"Portfolio construction {list(CONSTRUCTIONS)}", settings.construction)
                settings = settings.with_changes(construction=method)
                console = Console(record=True)
                result, _ = execute(settings, console)
            elif choice == "3":
                from .robustness import run_robustness

                result.robustness = run_robustness(result)
                terminal.robustness(console, result)
            elif choice == "4":
                path = Path(_ask("Path to historical signals CSV (date,ticker,signal)"))
                if not path.exists():
                    print(f"  ERROR: {path} not found")
                    continue
                console = Console(record=True)
                result, _ = execute(settings, console, signals_csv=path)
            elif choice == "5":
                save(result, console)
            elif choice == "6":
                return 0
            else:
                print("  Choose 1-6.")
        except (SettingsError, ValueError, RuntimeError) as exc:
            print(f"\n  ERROR: {exc}\n")


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
    )
    _setup_logging(output_dir(settings))
    console = Console(record=True, width=args.width)
    execute(settings, console, signals_csv=Path(args.signals) if args.signals else None,
            robustness=not args.no_robustness, charts=not args.no_charts)
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("research", help="motor long/short sobre listas de research (sin preguntas)")
    p.add_argument("--long", required=True, help="tickers LONG separados por comas")
    p.add_argument("--short", required=True, help="tickers SHORT separados por comas")
    p.add_argument("--benchmark")
    p.add_argument("--lookback", choices=list(LOOKBACK_YEARS))
    p.add_argument("--rebalance", choices=list(REBALANCES))
    p.add_argument("--construction", choices=list(CONSTRUCTIONS))
    p.add_argument("--capital")
    p.add_argument("--tc", help="coste de transaccion: 0.0005 o 0.05%%")
    p.add_argument("--rf", help="tasa libre de riesgo: 0.04 o 4%%")
    p.add_argument("--signals", help="CSV date,ticker,signal para el modo B")
    p.add_argument("--no-robustness", action="store_true")
    p.add_argument("--no-charts", action="store_true")
    p.add_argument("--width", type=int, default=160, help="ancho de la salida")
    p.set_defaults(func=run_once)


if __name__ == "__main__":
    sys.exit(interactive())
