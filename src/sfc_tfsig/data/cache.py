"""Cache en disco para todo lo que llega por red.

Por que existe: la SEC y Yahoo son gratuitos pero no son nuestros. Cada corrida
del modelo que vuelve a descargar lo mismo es una peticion que no aporta nada,
y la SEC bloquea por IP a quien abusa. Con cache, la primera corrida tarda y las
siguientes son instantaneas; y cuando alguien del equipo dice "a mi me dio otro
numero", lo primero que se compara es la fecha de la cache.

Formato: parquet si hay pyarrow (conserva tipos y fechas), CSV comprimido si no.
La degradacion es explicita -- `backend()` dice cual esta activo -- porque un
CSV que pierde el tipo de `filed` y lo deja como texto convierte una comparacion
de fechas en una comparacion de cadenas, y eso ordena "2019-1-5" antes que
"2019-12-31" sin avisar.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import CACHE_DIR

try:  # pragma: no cover - depende del entorno
    import pyarrow  # noqa: F401

    _HAS_PARQUET = True
except ImportError:  # pragma: no cover
    _HAS_PARQUET = False


def backend() -> str:
    """'parquet' o 'csv'. Se imprime en los diagnosticos a proposito."""
    return "parquet" if _HAS_PARQUET else "csv"


def _path(name: str) -> Path:
    suffix = ".parquet" if _HAS_PARQUET else ".csv.gz"
    return CACHE_DIR / f"{name}{suffix}"


def age_days(name: str) -> float | None:
    """Antiguedad del fichero en dias, o None si no existe."""
    path = _path(name)
    if not path.exists():
        return None
    return (time.time() - path.stat().st_mtime) / 86400.0


def read_frame(name: str, max_age_days: float | None = None) -> pd.DataFrame | None:
    """Devuelve el DataFrame cacheado, o None si falta o caduco.

    `max_age_days=None` significa "no caduca": se usa para datos historicos que
    ya no cambian (un 10-K de 2015 no se reescribe).
    """
    path = _path(name)
    if not path.exists():
        return None
    if max_age_days is not None:
        age = age_days(name)
        if age is not None and age > max_age_days:
            return None
    try:
        if _HAS_PARQUET:
            return pd.read_parquet(path)
        return pd.read_csv(path, compression="gzip")
    except (OSError, ValueError, pd.errors.ParserError):
        # Cache corrupta (descarga interrumpida, disco lleno). Se ignora y se
        # vuelve a bajar: nunca se propaga un fichero a medias al panel.
        return None


def write_frame(name: str, df: pd.DataFrame) -> Path:
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if _HAS_PARQUET:
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False, compression="gzip")
    # Escritura atomica: si el proceso muere a mitad, la cache anterior sigue
    # intacta en vez de quedar truncada.
    tmp.replace(path)
    return path


def read_json(name: str, max_age_days: float | None = None) -> Any | None:
    path = CACHE_DIR / f"{name}.json"
    if not path.exists():
        return None
    if max_age_days is not None:
        age = (time.time() - path.stat().st_mtime) / 86400.0
        if age > max_age_days:
            return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(name: str, payload: Any) -> Path:
    path = CACHE_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)
    return path


def exists(name: str) -> bool:
    return _path(name).exists()


def clear(prefix: str = "") -> int:
    """Borra la cache (opcionalmente solo lo que empieza por `prefix`)."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    for path in CACHE_DIR.iterdir():
        if path.is_file() and path.name.startswith(prefix):
            path.unlink()
            removed += 1
    return removed
