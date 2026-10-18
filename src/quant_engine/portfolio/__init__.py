"""portfolio del motor long/short."""

import logging


class _SolverImportNoise(logging.Filter):
    """Calla solo el aviso de cvxpy sobre solvers que no cargan.

    Al importarse, cvxpy prueba cada solver instalado y avisa en stderr de los
    que fallan. Smart App Control bloquea parte de OSQP en esta maquina (octubre
    de 2026) y el aviso salia en cada corrida, en medio del plan. Se resuelve
    con CLARABEL primero; si un solver falla al resolver, `construction._solve`
    lo reporta en la nota del metodo. Es un filtro y no un nivel porque cvxpy
    reinicia el nivel de su logger al importarse.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "importing solver" not in record.getMessage()


logging.getLogger("__cvxpy__").addFilter(_SolverImportNoise())
