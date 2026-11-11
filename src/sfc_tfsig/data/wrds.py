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
    2. instalar wrds (ver `pyproject.toml`, grupo `wrds`)
    3. python main.py wrds --setup

El paso 3 guarda la contrasena en el fichero pgpass de PostgreSQL. Sin el,
WRDS la pide por teclado en cada conexion, y un proceso sin terminal (un
backtest en segundo plano) se quedaria esperando para siempre. Este modulo lo
comprueba antes de conectar y falla con estas instrucciones.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import ROOT
from . import cache
from .sec import _read_env_file

SETUP_HELP = (
    "WRDS setup, once per machine:\n"
    "  1. in .env (repo root):  WRDS_USERNAME=your_user\n"
    "  2. install wrds (see pyproject.toml, wrds group)\n"
    "  3. python main.py wrds --setup   (asks for the password and saves it; create_pgpass_file)\n"
    "Step 3 saves the password so runs without a terminal do not hang."
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
        value = value.strip().strip('"').strip("'")
    else:
        value = _read_env_file(_env_file()).get(name) or None
    # PostgreSQL distingue mayusculas en el usuario y WRDS los crea en
    # minusculas: "Armaankumar006" fallaba con "PAM authentication failed".
    if value and name == "WRDS_USERNAME":
        value = value.lower()
    return value


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
        raise RuntimeError(f"the wrds package is missing.\n{SETUP_HELP}") from exc
    return wrds


def get_connection():
    """Abre una conexion. Quien la abre es responsable de cerrarla.

    Preferir `connection()`, que la cierra sola.
    """
    username = _read_env_value("WRDS_USERNAME")
    if not username:
        raise RuntimeError(f"WRDS_USERNAME is not set.\n{SETUP_HELP}")

    # Sin contrasena guardada, wrds la pide por teclado. Con terminal, que la
    # pida; sin terminal, fallar ahora en vez de colgarse.
    if not _pgpass_path().exists() and not sys.stdin.isatty():
        raise RuntimeError(
            f"no saved WRDS password ({_pgpass_path()}) and the process has no "
            f"terminal to ask for it.\n{SETUP_HELP}"
        )

    # El sondeo solo tiene sentido con contrasena guardada: sin ella, con
    # terminal, `wrds` la pide por teclado y eso es lo que se quiere.
    if _pgpass_path().exists():
        _probe(username)
    wrds = _import_wrds()
    return wrds.Connection(wrds_username=username)


_HOST, _PORT, _DB = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"


def _probe(username: str, timeout: int | None = None) -> None:
    """Prueba usuario y contrasena guardada antes de dar el control a `wrds`.

    Si la autenticacion falla, `wrds.Connection` no lanza: pide usuario y
    contrasena por teclado. Sin terminal eso es un EOFError a mitad del
    analisis; con terminal, una pregunta inesperada. Y si WRDS deja la
    conexion esperando (aprobacion Duo pendiente en el celular del titular),
    se queda colgada sin limite. Aqui se falla con un mensaje que dice que
    hacer, y en `timeout` segundos como mucho.
    """
    import psycopg2  # noqa: PLC0415

    timeout = timeout or int(_read_env_value("WRDS_CONNECT_TIMEOUT") or 45)
    try:
        psycopg2.connect(host=_HOST, port=_PORT, dbname=_DB, user=username, sslmode="require",
                         connect_timeout=timeout).close()
    except psycopg2.OperationalError as exc:
        msg = str(exc).strip()
        low = msg.lower()
        if "timeout" in low:
            raise RuntimeError(
                f"WRDS did not answer within {timeout}s (the network is reachable). It is usually a "
                "pending Duo approval on the account holder's phone: approve it and run again. "
                "To wait longer, set WRDS_CONNECT_TIMEOUT=90 in .env."
            ) from None
        if "no password supplied" in low:
            raise RuntimeError(
                f"pgpass has no line for {username} at {_HOST}. Create it with: "
                "python main.py wrds --setup"
            ) from None
        if "authentication" in low or "password" in low:
            raise RuntimeError(
                f"WRDS rejected the saved password for {username}. If it changed, "
                f"update it with: python main.py wrds --setup"
            ) from None
        raise RuntimeError(f"could not connect to WRDS: {msg.splitlines()[0]}") from None


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


def setup_pgpass(password: str | None = None) -> Path:
    """Conecta una vez con la contrasena escrita por teclado y la guarda en pgpass.

    Por que no basta `wrds.Connection(); db.create_pgpass_file()` a secas: sin
    usuario explicito, wrds lo PREGUNTA ofreciendo como valor por defecto el
    usuario de Windows ("usuario"). Quien pulsa Enter se autentica con un
    usuario que no existe en WRDS, la conexion falla y el pgpass nunca se crea.
    Aqui el usuario sale de `.env` y la contrasena se pide una sola vez.

    WRDS puede pedir ademas la aprobacion de Duo en el movil la primera vez.
    """
    import getpass  # noqa: PLC0415

    username = _read_env_value("WRDS_USERNAME")
    if not username:
        raise RuntimeError(f"WRDS_USERNAME is not set.\n{SETUP_HELP}")
    if password is None:
        # WRDS_PASSWORD en .env evita escribirla en la terminal. Tras crear el
        # pgpass conviene borrarla de .env: pgpass vive fuera del repo.
        password = _read_env_value("WRDS_PASSWORD")
    if not password:
        password = getpass.getpass(f"WRDS password for {username} (hidden while typing): ")

    wrds = _import_wrds()
    db = wrds.Connection(wrds_username=username, wrds_password=password)
    try:
        db.create_pgpass_file()
    finally:
        db.close()
    return _pgpass_path()


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


__all__ = ["connection", "get_connection", "fetch_table", "query", "list_libraries",
           "setup_pgpass", "SETUP_HELP"]
