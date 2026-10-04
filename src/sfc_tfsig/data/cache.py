"""Cache en disco para todo lo que llega por red.

Por que existe: la SEC y Yahoo son gratuitos pero no son nuestros. Cada corrida
del modelo que vuelve a descargar lo mismo es una peticion que no aporta nada,
y la SEC bloquea por IP a quien abusa. Con cache, la primera corrida tarda y las
siguientes son instantaneas; y cuando alguien del equipo dice "a mi me dio otro
numero", lo primero que se compara es la fecha de la cache.

Formato: pickle de pandas comprimido con gzip de nivel 1 (`.pkl.gz`). Conserva
tipos y fechas exactamente: un `filed` sigue siendo fecha y no texto, y una
comparacion de fechas no se convierte en una de cadenas (que ordena "2019-1-5"
antes que "2019-12-31" sin avisar).

Por que no parquet: desde octubre de 2026, Smart App Control de Windows bloquea
en esta maquina la DLL de parquet de pyarrow 25.0.1 (`_parquet.pyd`), y con
ella caia todo el proyecto: ni se leia ni se escribia la cache. pickle es
Python puro sobre pandas y numpy, sin ninguna libreria compilada mas. A cambio
ocupa ~30% mas que parquet y lee unas dos veces mas lento; con estos volumenes
(16 millones de filas de precios en 1,6 s) no importa.

Las caches viejas en parquet se migran solas la primera vez que se leen, si hay
un motor de parquet que cargue. Si no lo hay, cuentan como ausentes y se
vuelven a descargar.

pickle ejecuta codigo al cargar: en `data/cache/` solo debe haber ficheros que
escribio este proyecto. Nunca copiar ahi la cache de otra persona.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import CACHE_DIR

SUFFIX = ".pkl.gz"
_LEGACY_SUFFIX = ".parquet"
# Nivel 1: casi toda la reduccion de tamano de gzip a una fraccion del tiempo
# del nivel 9 por defecto, que escribia el panel cuatro veces mas lento.
_COMPRESSION = {"method": "gzip", "compresslevel": 1}

# Lo que puede romper la lectura de una cache: fichero truncado, gzip corrupto,
# o un pickle de otra version de pandas cuyas clases internas ya no existen.
# Cualquiera de ellos equivale a "no hay cache": se vuelve a bajar.
_READ_ERRORS = (OSError, EOFError, ValueError, pickle.UnpicklingError, AttributeError,
                ImportError, TypeError, gzip.BadGzipFile)


def backend() -> str:
    """Formato activo. Se imprime en los diagnosticos a proposito."""
    return "pickle+gzip"


def _path(name: str) -> Path:
    return CACHE_DIR / f"{name}{SUFFIX}"


@lru_cache(maxsize=1)
def _parquet_readable() -> bool:
    """Si hay un motor de parquet que cargue. Solo para migrar caches viejas.

    Se intenta una vez por proceso: con Smart App Control activo, cada intento
    deja un evento de bloqueo en el registro de Windows.
    """
    try:
        import pyarrow.parquet  # noqa: F401, PLC0415
    except (ImportError, OSError):
        return False
    return True


def migrate_legacy(name: str) -> bool:
    """Copia la cache parquet de `name` a pickle. True si quedo migrada.

    Conserva la fecha de modificacion del fichero original: la caducidad se
    mide con ella, y una migracion no puede hacer pasar por recientes unos
    precios de hace una semana.
    """
    legacy = CACHE_DIR / f"{name}{_LEGACY_SUFFIX}"
    if not legacy.exists() or not _parquet_readable():
        return False
    try:
        df = pd.read_parquet(legacy)
    except (OSError, ValueError, ImportError):
        return False
    stat = legacy.stat()
    path = write_frame(name, df)
    os.utime(path, (stat.st_atime, stat.st_mtime))
    return True


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
    if not path.exists() and not migrate_legacy(name):
        return None
    if max_age_days is not None:
        age = age_days(name)
        if age is not None and age > max_age_days:
            return None
    try:
        df = pd.read_pickle(path, compression="gzip")
    except _READ_ERRORS:
        # Cache corrupta (descarga interrumpida, disco lleno) o de otra version
        # de pandas. Se ignora y se vuelve a bajar: nunca se propaga un fichero
        # a medias al panel.
        return None
    return df if isinstance(df, pd.DataFrame) else None


def write_frame(name: str, df: pd.DataFrame) -> Path:
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_pickle(tmp, compression=_COMPRESSION, protocol=pickle.HIGHEST_PROTOCOL)
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
