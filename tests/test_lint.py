"""Errores que los tests de comportamiento no ven: nombres sin definir e imports rotos.

Un `NameError` en una rama que ningun test ejecuta (el HTML del plan) tumbo una
corrida con 808 tests en verde. `ruff --select F` lo encuentra sin ejecutar
nada. Sin ruff instalado, el test se salta.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_sin_nombres_indefinidos_ni_imports_rotos():
    pytest.importorskip("ruff")
    out = subprocess.run([sys.executable, "-m", "ruff", "check", "--select", "F", "--output-format", "concise",
                          "src", "tools"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
