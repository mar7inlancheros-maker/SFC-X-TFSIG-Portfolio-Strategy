"""Volatilidad condicional GARCH(1,1) y Monte Carlo filtrado sobre ella.

**Por que GARCH ademas de EWMA.** La EWMA de RiskMetrics es un GARCH sin
reversion: la volatilidad de hoy se proyecta constante a cualquier horizonte.
A un dia da igual; a un mes o a un trimestre no. Despues de un pico de
volatilidad, la EWMA con raiz del tiempo exagera el VaR mensual; en calma, lo
subestima. GARCH(1,1) revierte a una varianza de largo plazo con una vida media
medible, y eso es lo que hace creible un VaR a 21 o 63 sesiones.

**Estimacion.** Maxima verosimilitud gaussiana (cuasi-MV: los parametros son
consistentes aunque los residuos no sean normales, Bollerslev y Wooldridge
1992) con objetivo de varianza (Engle y Mezrich 1996): la varianza de largo
plazo se fija en la muestral y solo se estiman alfa y beta. Busqueda en rejilla
vectorizada y un refinado local; sin scipy, que no es dependencia del proyecto.

**Monte Carlo filtrado** (Barone-Adesi, Giannopoulos y Vosper 1999). Se
remuestrean los residuos estandarizados HISTORICOS -- con su asimetria y sus
colas reales -- y se reinyectan en la recursion GARCH partiendo de la
volatilidad de hoy. Cada trayectoria tiene racimos de volatilidad: un dia malo
sube la varianza del siguiente. Es lo que un bootstrap simple o una t de
Student con covarianza fija no pueden producir.

Funciones puras. Semilla explicita.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..metrics import TRADING_DAYS

_MIN_OBS = 250


@dataclass(frozen=True)
class GarchFit:
    mu: float
    omega: float
    alpha: float
    beta: float
    long_run_var: float
    loglik: float
    sigma: pd.Series        # prevision de sigma para cada dia, con datos hasta el anterior
    sigma_next: float       # prevision para la proxima sesion
    residuals: np.ndarray   # residuos estandarizados (r - mu) / sigma

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def half_life_d(self) -> float:
        """Sesiones para que se disipe la mitad de un choque de varianza."""
        p = self.persistence
        return math.log(0.5) / math.log(p) if 0.0 < p < 1.0 else float("inf")

    @property
    def long_run_vol(self) -> float:
        return math.sqrt(self.long_run_var * TRADING_DAYS)

    @property
    def current_vol(self) -> float:
        return self.sigma_next * math.sqrt(TRADING_DAYS)


def _variance_paths(eps: np.ndarray, long_run: float, alpha: np.ndarray,
                    beta: np.ndarray) -> np.ndarray:
    """Varianza condicional para una rejilla de (alfa, beta). Forma (T, G)."""
    omega = (1.0 - alpha - beta) * long_run
    out = np.empty((len(eps), len(alpha)))
    out[0] = long_run
    for t in range(1, len(eps)):
        out[t] = omega + alpha * eps[t - 1] ** 2 + beta * out[t - 1]
    return out


def _loglik(eps: np.ndarray, variance: np.ndarray) -> np.ndarray:
    variance = np.maximum(variance, 1e-18)
    return -0.5 * np.sum(np.log(2.0 * np.pi * variance) + (eps ** 2)[:, None] / variance, axis=0)


def _grid(alpha_lo, alpha_hi, beta_lo, beta_hi, n) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.meshgrid(np.linspace(alpha_lo, alpha_hi, n), np.linspace(beta_lo, beta_hi, n))
    a, b = a.ravel(), b.ravel()
    # Estacionariedad estricta: con alfa + beta >= 1 no hay varianza de largo
    # plazo a la que revertir y el objetivo de varianza pierde sentido.
    keep = (a > 0) & (b >= 0) & (a + b < 0.999)
    return a[keep], b[keep]


def fit_garch(returns: pd.Series) -> GarchFit:
    """GARCH(1,1) con objetivo de varianza por cuasi-maxima verosimilitud."""
    r = returns.dropna().astype(float)
    if len(r) < _MIN_OBS:
        raise ValueError(
            f"{len(r)} retornos para un GARCH: hacen falta al menos {_MIN_OBS} "
            "(un ano) para separar la persistencia del ruido"
        )
    mu = float(r.mean())
    eps = r.to_numpy() - mu
    long_run = float(np.var(eps, ddof=1))

    alpha, beta = _grid(0.005, 0.30, 0.50, 0.995, 40)
    ll = _loglik(eps, _variance_paths(eps, long_run, alpha, beta))
    best = int(np.nanargmax(ll))
    # Refinado local alrededor del mejor punto de la rejilla gruesa.
    step_a, step_b = 0.30 / 40, 0.495 / 40
    alpha, beta = _grid(max(alpha[best] - step_a, 1e-4), alpha[best] + step_a,
                        max(beta[best] - step_b, 0.0), min(beta[best] + step_b, 0.998), 25)
    ll = _loglik(eps, _variance_paths(eps, long_run, alpha, beta))
    best = int(np.nanargmax(ll))
    a, b = float(alpha[best]), float(beta[best])

    variance = _variance_paths(eps, long_run, np.array([a]), np.array([b]))[:, 0]
    omega = (1.0 - a - b) * long_run
    next_var = omega + a * eps[-1] ** 2 + b * variance[-1]
    sigma = np.sqrt(variance)
    return GarchFit(
        mu=mu, omega=omega, alpha=a, beta=b, long_run_var=long_run,
        loglik=float(ll[best]),
        sigma=pd.Series(sigma, index=r.index),
        sigma_next=float(math.sqrt(next_var)),
        residuals=eps / sigma,
    )


def vol_term_structure(fit: GarchFit, horizons_d) -> pd.DataFrame:
    """Volatilidad anualizada prevista para cada horizonte.

    La varianza esperada del dia k revierte a la de largo plazo al ritmo de la
    persistencia: E[s2_k] = V + p^(k-1) (s2_1 - V). La del horizonte es la
    suma de las diarias. Frente a la EWMA, la diferencia es justo ese termino.
    """
    p = fit.persistence
    first = fit.sigma_next ** 2
    rows = []
    for h in horizons_d:
        k = np.arange(int(h))
        total = float(np.sum(fit.long_run_var + p ** k * (first - fit.long_run_var)))
        rows.append({
            "horizon_d": int(h),
            "vol_annual": math.sqrt(total / h * TRADING_DAYS),
            "ewma_like_vol_annual": fit.current_vol,
            "long_run_vol_annual": fit.long_run_vol,
        })
    return pd.DataFrame(rows).set_index("horizon_d")


@dataclass
class FilteredSimulation:
    horizons_d: list[int]
    terminal: dict[int, np.ndarray]      # retornos acumulados por horizonte
    max_drawdown: dict[int, np.ndarray]
    vol_path: np.ndarray                 # mediana de la vol anualizada simulada por dia


def simulate_filtered(fit: GarchFit, horizons_d, *, n_paths: int, seed: int) -> FilteredSimulation:
    """Trayectorias GARCH con residuos historicos remuestreados."""
    horizons = sorted({int(h) for h in horizons_d})
    rng = np.random.default_rng(seed)
    z = fit.residuals[np.isfinite(fit.residuals)]
    if len(z) == 0:
        raise ValueError("sin residuos estandarizados: el ajuste GARCH no dejo datos")
    variance = np.full(n_paths, fit.sigma_next ** 2)
    log_wealth = np.zeros(n_paths)
    peak = np.zeros(n_paths)
    worst = np.zeros(n_paths)
    terminal: dict[int, np.ndarray] = {}
    drawdowns: dict[int, np.ndarray] = {}
    vol_path = np.empty(horizons[-1])

    for step in range(1, horizons[-1] + 1):
        vol_path[step - 1] = float(np.median(np.sqrt(variance * TRADING_DAYS)))
        eps = np.sqrt(variance) * z[rng.integers(0, len(z), n_paths)]
        r = np.maximum(fit.mu + eps, -0.99)
        log_wealth += np.log1p(r)
        peak = np.maximum(peak, log_wealth)
        worst = np.minimum(worst, np.expm1(log_wealth - peak))
        variance = fit.omega + fit.alpha * eps ** 2 + fit.beta * variance
        if step in horizons:
            terminal[step] = np.expm1(log_wealth)
            drawdowns[step] = worst.copy()
    return FilteredSimulation(horizons, terminal, drawdowns, vol_path)


def simulation_var_table(sim: FilteredSimulation, confidences, method: str = "garch_fhs") -> pd.DataFrame:
    """VaR y ES por horizonte y confianza, con el mismo esquema que `var_table`."""
    rows = []
    for h in sim.horizons_d:
        paths = sim.terminal[h]
        for c in confidences:
            cutoff = float(np.quantile(paths, 1.0 - c))
            rows.append({"method": method, "confidence": c, "horizon_d": h,
                         "var": -cutoff, "es": float(-paths[paths <= cutoff].mean())})
    return pd.DataFrame(rows)
