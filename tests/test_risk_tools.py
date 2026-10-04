"""Las herramientas de calidad de los tests de riesgo no se pudren en silencio.

El catalogo de mutantes describe fragmentos exactos del codigo. Si alguien
cambia una linea y su mutante deja de aplicarse, la prueba de mutacion lo
reportaria como "patron" la proxima vez que alguien la corra, que puede ser
dentro de meses. Este test lo dice en cada `pytest`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import risk_mutants  # noqa: E402
import risk_mutation  # noqa: E402


@pytest.mark.parametrize("mutant", risk_mutants.MUTANTS, ids=lambda m: m.label)
def test_cada_mutante_se_aplica_exactamente_una_vez(mutant):
    source = (ROOT / mutant.path).read_text(encoding="utf-8")
    mutated = risk_mutation.apply(source, mutant)
    assert mutated is not None, (
        f"'{mutant.label}': el fragmento ya no aparece una sola vez en {mutant.path}. "
        "Actualiza tools/risk_mutants.py con el codigo nuevo"
    )
    assert mutated != source
    for test in mutant.tests:
        assert (ROOT / test).exists(), f"'{mutant.label}' apunta a un test que no existe: {test}"


def test_las_etiquetas_no_se_repiten():
    labels = [m.label for m in risk_mutants.MUTANTS]
    assert len(labels) == len(set(labels))


def test_aplicar_respeta_los_saltos_de_linea():
    mutant = risk_mutants.Mutant("x", "f.py", "a = 1\nb = 2", "a = 1\nb = 3", ())
    assert risk_mutation.apply("a = 1\r\nb = 2\r\n", mutant) == "a = 1\r\nb = 3\r\n"
    assert risk_mutation.apply("a = 1\nb = 2\n", mutant) == "a = 1\nb = 3\n"
    assert risk_mutation.apply("a = 1\nb = 2\na = 1\nb = 2\n", mutant) is None


def test_lineas_ejecutables_ignoran_docstrings_y_cabeceras(tmp_path):
    import risk_coverage

    code = tmp_path / "m.py"
    code.write_text('"""doc"""\n\ndef f(x):\n    """doc"""\n    y = x + 1\n    return y\n',
                    encoding="utf-8")
    assert risk_coverage.executable_lines(code) == {5, 6}
