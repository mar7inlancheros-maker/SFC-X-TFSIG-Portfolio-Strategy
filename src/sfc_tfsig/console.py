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

import sys


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
