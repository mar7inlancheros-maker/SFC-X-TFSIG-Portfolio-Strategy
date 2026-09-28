"""Acceso a WRDS (CRSP, Compustat, Fama-French...) a traves de la cache del proyecto.

La conexion en vivo queda aislada aqui. El resto del modelo lee de la cache
local en disco y no llama a WRDS en cada corrida.

Uso:

    df = fetch_table("crsp.msf", columns="permno, date, prc",
                     where_clause="date >= '2020-01-01'", cache_name="crsp_msf")

    # Varias consultas con UNA sola conexion:
    with connection() as conn:
        a = query("SELECT ...", conn=conn, cache_name="a")
        b = query("SELECT ...", conn=conn, cache_name="b")

Configuracion, una sola vez por maquina:

    1. `.env` en la raiz del repo:  WRDS_USERNAME=tu_usuario
    2. pip install wrds
    3. python -c "import wrds; db = wrds.Connection(); db.create_pgpass_file()"

El paso 3 guarda la contrasena en el fichero pgpass de PostgreSQL. Sin el,
WRDS la pide por teclado en cada conexion, y un proceso sin terminal (un
backtest en segundo plano) se quedaria esperando para siempre. Este modulo lo
comprueba antes de conectar y falla con estas instrucciones.
"""

from __future__ import annotations

import hashlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from ..paths import ROOT
from . import cache
from .sec import _read_env_file

SETUP_HELP = (
    "Configuracion de WRDS, una vez por maquina:\n"
    "  1. en .env (raiz del repo):  WRDS_USERNAME=tu_usuario\n"
    "  2. pip install wrds\n"
    '  3. python -c "import wrds; db = wrds.Connection(); db.create_pgpass_file()"\n'
    "El paso 3 guarda la contrasena para que las corridas sin terminal no se cuelguen."
)


def _env_file() -> Path:
    """`.env` de la RAIZ del repo, no del directorio desde el que se ejecuta.

    La version anterior usaba `Path(".env")`: al correr desde otra carpeta no
    encontraba el usuario aunque estuviera configurado.
    """
    return ROOT / ".env"


def _read_env_value(name: str) -> str | None:
    """Variable de entorno o, si no esta, la linea `NAME=valor` del `.env`.

    Reutiliza el lector de `sec.py`, que abre con utf-8-sig: si el `.env` lo
    escribio PowerShell con `Out-File`, lleva BOM, y leyendo en utf-8 la
    primera clave queda como "\\ufeffWRDS_USERNAME" y no se encuentra.
    """
    value = os.getenv(name)
    if value:
        return value.strip().strip('"').strip("'")
    return _read_env_file(_env_file()).get(name) or None


def _pgpass_path() -> Path:
    """Donde busca PostgreSQL la contrasena guardada."""
    if os.getenv("PGPASSFILE"):
        return Path(os.environ["PGPASSFILE"])
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", "")) / "postgresql" / "pgpass.conf"
    return Path.home() / ".pgpass"


def _import_wrds():
    """Importacion perezosa: el resto del proyecto carga sin el paquete wrds."""
    try:
        import wrds  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"falta el paquete wrds.\n{SETUP_HELP}") from exc
    return wrds


def get_connection():
    """Abre una conexion. Quien la abre es responsable de cerrarla.

    Preferir `connection()`, que la cierra sola.
    """
    username = _read_env_value("WRDS_USERNAME")
    if not username:
        raise RuntimeError(f"WRDS_USERNAME no esta definido.\n{SETUP_HELP}")

    # Sin contrasena guardada, wrds la pide por teclado. Con terminal, que la
    # pida; sin terminal, fallar ahora en vez de colgarse.
    if not _pgpass_path().exists() and not sys.stdin.isatty():
        raise RuntimeError(
            f"no hay contrasena de WRDS guardada ({_pgpass_path()}) y el proceso no "
            f"tiene terminal para pedirla.\n{SETUP_HELP}"
        )

    wrds = _import_wrds()
    return wrds.Connection(wrds_username=username)


@contextmanager
def connection() -> Iterator[Any]:
    """Una conexion que se cierra al salir, tambien si la consulta falla.

    WRDS limita las conexiones simultaneas por usuario. La version anterior
    abria una por consulta y no cerraba ninguna: un bucle de consultas agotaba
    el cupo.
    """
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def _cache_key(cache_name: str, sql: str) -> str:
    """Nombre de cache ligado al TEXTO de la consulta.

    Con solo `cache_name`, cambiar el filtro o las columnas y reutilizar el
    nombre devolvia los datos de la consulta anterior sin avisar. El hash del
    SQL hace que una consulta distinta sea, por construccion, otra entrada.
    """
    digest = hashlib.sha256(" ".join(sql.split()).encode("utf-8")).hexdigest()[:12]
    return f"wrds/{cache_name}_{digest}"


def query(
    sql: str,
    *,
    cache_name: str | None = None,
    refresh: bool = False,
    conn: Any | None = None,
) -> pd.DataFrame:
    """Ejecuta SQL contra WRDS, con cache opcional.

    `conn`: una conexion abierta para reutilizar (no se cierra aqui). Si no se
    pasa, se abre una y se cierra al terminar.
    `refresh`: ignora la cache y vuelve a consultar. La cache no caduca sola:
    un dato de CRSP de 2015 no cambia, pero los meses recientes si.
    """
    key = _cache_key(cache_name, sql) if cache_name is not None else None
    if key is not None and not refresh:
        cached = cache.read_frame(key)
        if cached is not None:
            return cached

    if conn is not None:
        df = conn.raw_sql(sql)
    else:
        with connection() as own:
            df = own.raw_sql(sql)

    if key is not None:
        cache.write_frame(key, df)
    return df


def fetch_table(
    table_name: str,
    *,
    columns: str = "*",
    where_clause: str | None = None,
    limit: int | None = None,
    cache_name: str | None = None,
    refresh: bool = False,
    conn: Any | None = None,
) -> pd.DataFrame:
    """SELECT sobre una tabla de WRDS. Ver `query` para cache y conexion."""
    sql = f"SELECT {columns} FROM {table_name}"
    if where_clause:
        sql += f" WHERE {where_clause}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return query(sql, cache_name=cache_name, refresh=refresh, conn=conn)


def list_libraries(conn: Any | None = None) -> list[str]:
    """Bibliotecas incluidas en la suscripcion (crsp, comp, ff, ibes...)."""
    if conn is not None:
        return sorted(conn.list_libraries())
    with connection() as own:
        return sorted(own.list_libraries())


__all__ = ["connection", "get_connection", "fetch_table", "query", "list_libraries", "SETUP_HELP"]
