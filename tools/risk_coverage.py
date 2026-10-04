#!/usr/bin/env python
"""Cobertura de lineas del modulo de riesgo, sin dependencias.

    python tools/risk_coverage.py              # informe por fichero
    python tools/risk_coverage.py --min 95     # sale con 1 si queda por debajo

Usa `sys.monitoring` (Python 3.12+) en vez de `coverage`: no hay que instalar
nada, y en esta maquina el Control de aplicaciones de Windows bloquea a veces
extensiones compiladas recien instaladas.

Cuenta las sentencias ejecutables de `src/sfc_tfsig/risk/*.py` y
`risk_report.py` (sin docstrings ni definiciones) y lista las que ningun test de
`tests/test_risk_*.py` ejecuta. Una linea cubierta no prueba que un test
detecte un error en ella: para eso esta `tools/risk_mutation.py`.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfc_tfsig.console import enable_utf8_stdout  # noqa: E402

TARGETS = sorted((ROOT / "src" / "sfc_tfsig" / "risk").glob("*.py")) + [
    ROOT / "src" / "sfc_tfsig" / "risk_report.py"
]


def executable_lines(path: Path) -> set[int]:
    """Lineas con sentencias que se pueden ejecutar, sin docstrings ni cabeceras."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        lines.add(node.lineno)
    return lines


def measure(test_args: list[str]) -> tuple[int, dict[Path, set[int]]]:
    """Corre pytest dentro de este proceso registrando las lineas ejecutadas."""
    if sys.version_info < (3, 12):
        raise SystemExit("hace falta Python 3.12+ para sys.monitoring")
    import pytest  # noqa: PLC0415

    keys = {os.path.normcase(str(t)): t for t in TARGETS}
    hits: dict[Path, set[int]] = {t: set() for t in TARGETS}
    monitoring = sys.monitoring
    tool = monitoring.COVERAGE_ID
    monitoring.use_tool_id(tool, "risk_coverage")

    def on_line(code, line):
        target = keys.get(os.path.normcase(code.co_filename))
        if target is None:
            # Fuera del modulo de riesgo: se desactiva para ese codigo y el
            # resto de la suite corre casi a velocidad normal.
            return monitoring.DISABLE
        hits[target].add(line)
        return None

    monitoring.register_callback(tool, monitoring.events.LINE, on_line)
    monitoring.set_events(tool, monitoring.events.LINE)
    try:
        code = pytest.main(["-o", "addopts=", "-q", "-p", "no:cacheprovider", *test_args])
    finally:
        monitoring.set_events(tool, 0)
        monitoring.free_tool_id(tool)
    return int(code), hits


def main(argv: list[str] | None = None) -> int:
    enable_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min", type=float, default=0.0, help="cobertura minima total, en %%")
    args = parser.parse_args(argv)

    os.chdir(ROOT)
    tests = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_risk_*.py"))
    code, hits = measure(tests)
    if code != 0:
        print("\nlos tests de riesgo fallan: la cobertura de una suite roja no significa nada")
        return 2

    total_lines = total_hit = 0
    print("\nCobertura de lineas del modulo de riesgo")
    for target in TARGETS:
        lines = executable_lines(target)
        if not lines:
            continue
        hit = hits[target] & lines
        total_lines += len(lines)
        total_hit += len(hit)
        missing = sorted(lines - hit)
        name = target.relative_to(ROOT).as_posix()
        print(f"  {name:<40} {len(hit):>4}/{len(lines):<4} {len(hit) / len(lines):6.1%}")
        if missing:
            print(f"      sin ejecutar: {missing}")
    pct = 100.0 * total_hit / total_lines
    print(f"  {'TOTAL':<40} {total_hit:>4}/{total_lines:<4} {pct / 100:6.1%}")
    if pct < args.min:
        print(f"por debajo del minimo pedido ({args.min:.1f}%)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
