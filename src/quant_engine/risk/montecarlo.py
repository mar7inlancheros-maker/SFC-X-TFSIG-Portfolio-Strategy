"""Monte Carlo. SIMULADO, y se etiqueta como tal en el reporte.

Dos metodos, porque cada uno falla de una forma distinta:

1. **Bootstrap por bloques estacionario** (Politis-Romano 1994) sobre los
   retornos diarios de la cartera. Conserva colas gruesas y la autocorrelacion
   de corto plazo. Solo puede generar lo que ya paso: si la historia no tuvo
   un crash, la simulacion tampoco.
2. **Normal multivariante** con la covarianza estimada. Puede generar
   escenarios nuevos, pero con colas finas: subestima los extremos.

Semilla fija: el mismo input da el mismo output. Un Monte Carlo que cambia en
cada corrida no se puede discutir en un comite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stationary_bootstrap(returns: pd.Series, n_paths: int, horizon: int, block: int, seed: int) -> np.ndarray:
    r = returns.dropna().to_numpy(float)
    n = len(r)
    rng = np.random.default_rng(seed)
    p = 1.0 / block
    idx = np.empty((n_paths, horizon), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_paths)
    jumps = rng.random((n_paths, horizon)) < p
    fresh = rng.integers(0, n, (n_paths, horizon))
    for t in range(1, horizon):
        idx[:, t] = np.where(jumps[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return r[idx]


def parametric_normal(weights: pd.Series, mean: pd.Series, cov: pd.DataFrame,
                      n_paths: int, horizon: int, seed: int, drift: float = 0.0) -> np.ndarray:
    """Normal con media = mean' w + drift y varianza = w' Sigma w.

    `drift` es lo que rinde la caja de la cartera. Con medias de activo en cero
    (deriva neutra) y drift = rf - prestamo, la cartera rinde exactamente la
    tasa libre de riesgo neta. NO se pone rf en cada activo: con signos, eso da
    bruta x rf (8% en una 100/100 con rf 4%), no rf sobre el capital.
    """
    names = list(weights.index)
    w = weights.to_numpy(float)
    mu = float(mean.reindex(names).fillna(0.0).to_numpy() @ w) + drift
    sd = float(np.sqrt(max(w @ cov.loc[names, names].to_numpy(float) @ w, 0.0)))
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sd, (n_paths, horizon))


def summarize(paths: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    wealth = np.cumprod(1.0 + paths, axis=1)
    terminal = wealth[:, -1] - 1.0
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(wealth), 1)), wealth], axis=1), axis=1)[:, 1:]
    max_dd = (wealth / peak - 1.0).min(axis=1)
    out = {
        "paths": float(len(paths)),
        "expected_terminal_wealth": float(1.0 + terminal.mean()),
        "median_return": float(np.median(terminal)),
        "p05_return": float(np.quantile(terminal, 0.05)),
        "p95_return": float(np.quantile(terminal, 0.95)),
        "prob_loss": float((terminal < 0).mean()),
        "median_max_dd": float(np.median(max_dd)),
        # El 5% de trayectorias con PEOR drawdown queda por debajo de esto.
        "worst_5pct_max_dd": float(np.quantile(max_dd, 0.05)),
    }
    for th in thresholds:
        out[f"prob_dd_worse_{int(th * 100)}pct"] = float((max_dd <= -th).mean())
    return out


def terminal_distribution(paths: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + paths, axis=1)[:, -1] - 1.0
