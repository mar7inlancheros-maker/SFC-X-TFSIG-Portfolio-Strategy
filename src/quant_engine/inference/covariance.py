"""Estimadores de covarianza y su estabilidad. Funciones puras.

Con 10 activos y 1.250 dias la covarianza muestral es razonable; con 10 activos
y 60 dias, o con dos activos casi identicos, esta mal condicionada, y un
optimizador que la invierte amplifica el ruido hasta dar pesos absurdos. Por eso
se reportan varios estimadores y el NUMERO DE CONDICION de cada uno:

- muestral;
- EWMA (lambda del YAML), reacciona antes a un cambio de regimen;
- Ledoit-Wolf hacia correlacion constante (la de `sfc_tfsig.risk.exposure`);
- Ledoit-Wolf hacia identidad escalada (scikit-learn), la contraccion
  "diagonal": encoge las covarianzas fuera de la diagonal.

Todo en unidades DIARIAS; `annualize` multiplica por 252 al final.

Numero de condicion = autovalor maximo / minimo. Por encima de ~1.000 la
inversa amplifica el ruido tres ordenes de magnitud: el optimizador usa
entonces Ledoit-Wolf, y el reporte lo dice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from sfc_tfsig.risk.exposure import ledoit_wolf as ledoit_wolf_constant_corr

ILL_CONDITIONED = 1_000.0


def sample(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.dropna().cov()


def ewma(returns: pd.DataFrame, lam: float) -> pd.DataFrame:
    r = returns.dropna()
    x = r.to_numpy(float)
    x = x - x.mean(axis=0)
    weights = (1.0 - lam) * lam ** np.arange(len(x) - 1, -1, -1)
    weights /= weights.sum()
    cov = (x * weights[:, None]).T @ x
    return pd.DataFrame(cov, index=r.columns, columns=r.columns)


def ledoit_wolf_identity(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    r = returns.dropna()
    lw = LedoitWolf().fit(r.to_numpy(float))
    return pd.DataFrame(lw.covariance_, index=r.columns, columns=r.columns), float(lw.shrinkage_)


def condition_number(cov: pd.DataFrame) -> float:
    eig = np.linalg.eigvalsh(cov.to_numpy(float))
    return float(eig.max() / eig.min()) if eig.min() > 0 else float("inf")


def is_psd(cov: pd.DataFrame, tol: float = -1e-12) -> bool:
    return bool(np.linalg.eigvalsh(cov.to_numpy(float)).min() >= tol)


def annualize(cov: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    return cov * periods


def estimate_all(returns: pd.DataFrame, lam: float) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    s = sample(returns)
    out["sample"] = {"cov": s, "condition": condition_number(s), "shrinkage": 0.0}
    e = ewma(returns, lam)
    out["ewma"] = {"cov": e, "condition": condition_number(e), "shrinkage": float("nan")}
    cc, delta = ledoit_wolf_constant_corr(returns.dropna())
    out["ledoit_wolf_cc"] = {"cov": cc, "condition": condition_number(cc), "shrinkage": delta}
    li, delta_i = ledoit_wolf_identity(returns)
    out["ledoit_wolf_identity"] = {"cov": li, "condition": condition_number(li), "shrinkage": delta_i}
    for v in out.values():
        v["psd"] = is_psd(v["cov"])
    return out


def for_optimization(returns: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """La que usa el optimizador: muestral si esta bien condicionada; si no, LW.

    Se prefiere la muestral cuando es estable porque es la que no impone ningun
    supuesto; la contraccion se reserva para cuando la muestral no se puede
    invertir sin amplificar ruido.
    """
    s = sample(returns)
    if condition_number(s) <= ILL_CONDITIONED and is_psd(s):
        return s, "sample"
    cc, _ = ledoit_wolf_constant_corr(returns.dropna())
    return cc, "ledoit_wolf_cc"
