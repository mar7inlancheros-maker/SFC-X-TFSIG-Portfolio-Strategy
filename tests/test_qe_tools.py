"""El catalogo de mutantes del plan long/short sigue apuntando al codigo real."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import qe_mutants  # noqa: E402
import risk_mutation  # noqa: E402


@pytest.mark.parametrize("mutant", qe_mutants.MUTANTS, ids=lambda m: m.label)
def test_cada_mutante_se_aplica_exactamente_una_vez(mutant):
    source = (ROOT / mutant.path).read_text(encoding="utf-8")
    mutated = risk_mutation.apply(source, mutant)
    assert mutated is not None, (
        f"'{mutant.label}': el fragmento ya no aparece una sola vez en {mutant.path}. "
        "Actualiza tools/qe_mutants.py con el codigo nuevo"
    )
    assert mutated != source
    for test in mutant.tests:
        assert (ROOT / test).exists(), f"'{mutant.label}' apunta a un test que no existe: {test}"


def test_las_etiquetas_no_se_repiten():
    labels = [m.label for m in qe_mutants.MUTANTS]
    assert len(labels) == len(set(labels))
