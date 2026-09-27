"""Access to WRDS data through the project cache.

This module keeps the live database connection isolated from the rest of the
strategy code. The rest of the app should read from the local parquet/csv cache
rather than call WRDS directly on every run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from . import cache


def _read_env_value(name: str) -> str | None:
    """Read an env value from the process env or a local .env file.

    Accepts the common pattern ``KEY = "value"`` used by editor env files.
    """
    value = os.getenv(name)
    if value:
        return value.strip().strip('"').strip("'")

    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            if key.strip() == name:
                return raw_value.strip().strip('"').strip("'")
    return None


def _import_wrds():
    """Import WRDS lazily so that the rest of the project can load without it."""
    try:
        import wrds  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "falta wrds. Instala con: pip install wrds"
        ) from exc
    return wrds


def get_connection():
    """Open a WRDS connection using the environment variable WRDS_USERNAME."""
    username = _read_env_value("WRDS_USERNAME")
    if not username:
        raise RuntimeError(
            "WRDS_USERNAME no esta definido. Exporta la variable antes de llamar a WRDS o guarda la linea WRDS_USERNAME=... en .env."
        )

    wrds = _import_wrds()
    return wrds.Connection(wrds_username=username)


def fetch_table(
    table_name: str,
    *,
    columns: str = "*",
    where_clause: str | None = None,
    limit: int | None = None,
    cache_name: str | None = None,
) -> pd.DataFrame:
    """Read a WRDS table, optionally reading from cache first."""
    if cache_name is not None:
        cached = cache.read_frame(cache_name)
        if cached is not None:
            return cached

    query = f"SELECT {columns} FROM {table_name}"
    if where_clause:
        query += f" WHERE {where_clause}"
    if limit is not None:
        query += f" LIMIT {limit}"

    conn = get_connection()
    df = conn.raw_sql(query)

    if cache_name is not None:
        cache.write_frame(cache_name, df)

    return df


def query(
    sql: str,
    *,
    cache_name: str | None = None,
) -> pd.DataFrame:
    """Execute a raw SQL query against WRDS."""
    if cache_name is not None:
        cached = cache.read_frame(cache_name)
        if cached is not None:
            return cached

    conn = get_connection()
    df = conn.raw_sql(sql)

    if cache_name is not None:
        cache.write_frame(cache_name, df)

    return df


__all__ = ["get_connection", "fetch_table", "query"]
