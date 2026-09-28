#!/usr/bin/env python
"""Modelo cuantitativo long/short: pide los tickers y corre todo el analisis.

    python modelo.py

Equivale a `python main.py` sin argumentos. Existe para que quien solo quiera
analizar una lista de tickers no tenga que conocer los subcomandos del modelo
multifactor.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_engine.app import interactive  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(interactive())
