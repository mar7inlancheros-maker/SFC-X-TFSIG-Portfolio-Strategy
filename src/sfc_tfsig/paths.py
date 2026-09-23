"""Rutas del proyecto. UNICO sitio donde se calcula la raiz.

Cualquier modulo que necesite escribir o leer del disco pide la ruta aqui. Si
alguien hace `Path(__file__).parent.parent` por su cuenta, el dia que movamos un
fichero de sitio se rompe en silencio y escribe la cache en el lugar equivocado.
"""

from __future__ import annotations

from pathlib import Path

# src/sfc_tfsig/paths.py -> src/sfc_tfsig -> src -> raiz
ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = ROOT / "output"
REPORT_DIR = OUTPUT_DIR / "reports"

DEFAULT_CONFIG = CONFIG_DIR / "strategy.toml"
RISK_CONFIG = CONFIG_DIR / "risk.toml"


def ensure_dirs() -> None:
    """Crea los directorios de escritura. Idempotente."""
    for d in (DATA_DIR, CACHE_DIR, OUTPUT_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
