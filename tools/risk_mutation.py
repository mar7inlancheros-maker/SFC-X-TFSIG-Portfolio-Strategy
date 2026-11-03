#!/usr/bin/env python
"""Prueba de mutacion del modulo de riesgo: cazan los tests los bugs plausibles?

    python tools/risk_mutation.py              # todo el catalogo
    python tools/risk_mutation.py garch var    # solo los que mencionan esas palabras
    python tools/risk_mutation.py --list       # ver el catalogo
    python tools/risk_mutation.py --jobs 4     # en paralelo, una copia por proceso
    python tools/risk_mutation.py --catalogo qe --jobs 4   # el plan long/short (tools/qe_mutants.py)

**Nunca toca el arbol de trabajo.** Copia `src/`, `tests/` y `config/` a un
directorio temporal y muta ahi. Asi se puede correr con el editor abierto o con
otra sesion trabajando en el repo: restaurar un fichero mutado encima de
cambios ajenos seria perder trabajo sin aviso.

Primero corre los tests de riesgo sin mutar: si ya fallan, el resultado no
significaria nada y se aborta. Despues, cada mutante se aplica, se corren SUS
tests con `-x` y se restaura. Codigo de salida 1 si sobrevive alguno o si un
patron del catalogo ya no existe en el codigo.

La cobertura de lineas dice que codigo se ejecuta; la mutacion dice si algun
test se daria cuenta de que ese codigo esta mal. Son preguntas distintas.
"""

from __future__ import annotations

import argparse
import queue
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TOOLS))

from risk_mutants import MUTANTS, Mutant  # noqa: E402

from sfc_tfsig.console import enable_utf8_stdout  # noqa: E402

_COPY = ("src", "tests", "config")
_IGNORE = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")


def _make_copy(base: Path) -> Path:
    for name in _COPY:
        shutil.copytree(ROOT / name, base / name, ignore=_IGNORE)
    return base


def _pytest(copy: Path, tests, timeout: float) -> tuple[str, str]:
    """Corre pytest en la copia. Devuelve (estado, ultimas lineas de salida)."""
    cmd = [sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-x", "-p", "no:cacheprovider",
           *tests]
    try:
        done = subprocess.run(cmd, cwd=copy, capture_output=True, text=True, timeout=timeout,
                              check=False)  # el codigo de salida se interpreta abajo
    except subprocess.TimeoutExpired:
        return "timeout", ""
    tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-3:])
    return ("pasa" if done.returncode == 0 else "falla"), tail


def apply(text: str, mutant: Mutant) -> str | None:
    """Texto mutado, o None si el patron no esta exactamente una vez.

    Se compara con saltos de linea normalizados: hay ficheros con CRLF y el
    catalogo se escribe con LF.
    """
    crlf = "\r\n" in text
    plain = text.replace("\r\n", "\n")
    if plain.count(mutant.old) != 1:
        return None
    out = plain.replace(mutant.old, mutant.new, 1)
    return out.replace("\n", "\r\n") if crlf else out


def run_mutant(copy: Path, mutant: Mutant, timeout: float) -> tuple[str, str]:
    target = copy / mutant.path
    original = target.read_bytes()
    mutated = apply(original.decode("utf-8"), mutant)
    if mutated is None:
        return "patron", "el fragmento no aparece una sola vez: actualiza tools/risk_mutants.py"
    try:
        target.write_bytes(mutated.encode("utf-8"))
        state, tail = _pytest(copy, mutant.tests, timeout)
    finally:
        target.write_bytes(original)
    # Un test que se cuelga con el bug tambien lo delata, pero se marca aparte:
    # un timeout puede ser lentitud y no deteccion.
    return {"falla": "cazado", "timeout": "cazado (timeout)", "pasa": "SOBREVIVE"}[state], tail


def main(argv: list[str] | None = None) -> int:
    enable_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filtro", nargs="*", help="palabras de la etiqueta o del fichero")
    parser.add_argument("--list", action="store_true", help="lista el catalogo y sale")
    parser.add_argument("--jobs", type=int, default=1, help="procesos en paralelo")
    parser.add_argument("--timeout", type=float, default=240.0, help="segundos por mutante")
    parser.add_argument("--catalogo", choices=["riesgo", "qe"], default="riesgo",
                        help="riesgo: tools/risk_mutants.py; qe: el plan long/short, tools/qe_mutants.py")
    args = parser.parse_args(argv)
    catalog = MUTANTS
    if args.catalogo == "qe":
        from qe_mutants import MUTANTS as catalog  # noqa: PLC0415

    selected = [m for m in catalog
                if not args.filtro or any(f.lower() in (m.label + " " + m.path).lower() for f in args.filtro)]
    if args.list:
        for m in selected:
            print(f"{m.path:<34} {m.label}")
        print(f"{len(selected)} mutantes")
        return 0
    if not selected:
        print("ningun mutante coincide con el filtro; usa --list para verlos")
        return 2

    jobs = max(1, min(args.jobs, len(selected)))
    with tempfile.TemporaryDirectory(prefix="riskmut_") as tmp:
        copies: queue.Queue[Path] = queue.Queue()
        for i in range(jobs):
            copies.put(_make_copy(Path(tmp) / f"c{i}"))

        start = time.time()
        sample = copies.get()
        tests = sorted({t for m in selected for t in m.tests})
        state, tail = _pytest(sample, tests, args.timeout * 2)
        if state == "timeout":
            # La primera importacion en una copia recien creada puede tardar
            # mucho en Windows (el antivirus escanea cada fichero nuevo). Un
            # segundo intento ya encuentra todo escaneado.
            print("la linea base agoto el tiempo; segundo intento...", flush=True)
            state, tail = _pytest(sample, tests, args.timeout * 2)
        copies.put(sample)
        if state != "pasa":
            reason = "agotan el tiempo" if state == "timeout" else "fallan"
            print(f"los tests ya {reason} SIN mutar; arreglalo antes de medir mutantes")
            if tail:
                print(tail)
            return 2
        print(f"linea base: {len(tests)} ficheros de test en verde ({time.time() - start:.0f} s)")

        def work(mutant: Mutant) -> tuple[Mutant, str, str]:
            copy = copies.get()
            try:
                return (mutant, *run_mutant(copy, mutant, args.timeout))
            finally:
                copies.put(copy)

        results = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for mutant, verdict, tail in pool.map(work, selected):
                results.append((mutant, verdict))
                print(f"{verdict:<17} {mutant.label}  [{Path(mutant.path).name}]", flush=True)
                if verdict in ("SOBREVIVE", "patron") and tail:
                    print(f"                  {tail.splitlines()[-1]}")

    caught = sum(v.startswith("cazado") for _, v in results)
    bad = [m.label for m, v in results if v in ("SOBREVIVE", "patron")]
    print(f"\n{caught}/{len(results)} mutantes cazados en {time.time() - start:.0f} s")
    if bad:
        print("sin cazar: " + "; ".join(bad))
        print("Cada superviviente es un bug que la suite no veria: escribe el test que lo caza.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
