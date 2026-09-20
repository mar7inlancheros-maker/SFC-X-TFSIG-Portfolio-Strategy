#!/usr/bin/env python
"""Punto de entrada sin instalar el paquete.

    python main.py backtest

Inserta `src/` en la ruta de importacion y delega en `cli.py`. Si el paquete se
instala con `pip install -e .`, el import funciona igual y esta linea sobra.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sfc_tfsig.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
