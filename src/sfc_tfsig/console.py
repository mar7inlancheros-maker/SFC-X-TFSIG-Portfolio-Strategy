"""Salida de consola segura en Windows.

Los reportes dibujan con caracteres fuera de cp1252 (guiones largos, bloques,
flechas). Interactivamente funciona; en cuanto alguien redirige la salida a un
fichero, Windows cae a la pagina de codigos ANSI y revienta con
UnicodeEncodeError a mitad del reporte -- el fichero queda truncado y parece que
el modelo fallo, cuando lo que fallo fue imprimirlo.

Se llama SOLO desde los puntos de entrada (`main.py`, scripts, tests que
redirigen). Nunca desde un modulo de libreria: reconfigurar stdout por sorpresa
dentro de una importacion es exactamente el tipo de efecto secundario que luego
nadie encuentra.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

# Banderas de SetThreadExecutionState (winbase.h).
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


def enable_utf8_stdout() -> None:
    """Fuerza UTF-8 en stdout/stderr si el flujo lo permite."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Flujo ya cerrado o no reconfigurable (pytest capturando, por
            # ejemplo). No es motivo para tumbar el proceso.
            pass


@contextmanager
def keep_awake() -> Iterator[None]:
    """Pide a Windows que no suspenda el equipo mientras dura el bloque.

    El 3 y el 4 de octubre de 2026 dos corridas quedaron congeladas horas: con
    la pantalla apagada por inactividad, el portatil entro en suspension
    moderna y Windows congela los programas de escritorio. Se pide pantalla
    encendida porque en suspension moderna es la pantalla apagada lo que
    suspende. Al salir se libera; cerrar la tapa sigue suspendiendo. Fuera de
    Windows no hace nada.
    """
    kernel32 = None
    if os.name == "nt":
        try:
            import ctypes  # noqa: PLC0415

            kernel32 = ctypes.windll.kernel32
            if not kernel32.SetThreadExecutionState(
                _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
            ):
                kernel32 = None
        except (AttributeError, OSError):
            kernel32 = None
    try:
        yield
    finally:
        if kernel32 is not None:
            kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
