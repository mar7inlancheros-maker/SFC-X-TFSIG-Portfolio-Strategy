"""Cinco metodos de construccion long/short. Todos respetan el signo de research.

| Metodo | Que iguala u optimiza | Neta |
|---|---|---|
| equal_weight | mismo peso dentro de cada pata | fijada (N) |
| inverse_vol | peso proporcional a 1/sigma dentro de cada pata | fijada (N) |
| risk_parity | misma contribucion al riesgo de SU PATA (ERC por pata) | fijada (N) |
| min_variance | minima varianza con las patas fijadas | fijada (N) |
| max_sharpe | maximo Sharpe ex-ante con las patas fijadas | fijada (N) |

**Paridad de riesgo long/short, dentro de cada pata.** Cada pata se resuelve
como un ERC long-only sobre la covarianza de sus nombres (Spinu 2013: minimizar
0,5 x' Sigma x - (1/n) sum ln x_i, que es convexo) y se escala a su tamano:
L = (G + N) / 2 y S = (G - N) / 2. Asi la neta queda fijada, como pide la
seccion 16 de la especificacion y `net_exposure` del YAML.

Antes se igualaban las contribuciones al riesgo TOTAL de la cartera (ERC sobre
S Sigma S, con S = diag(signos)). Eso deja la neta libre: +0,23 en la corrida
de referencia, entre -0,12 y +0,32 en el backtest. Una cartera "neutral en
dolares" con un 23% neto largo no lo es. Auditoria #12.

**Maximo Sharpe** es el metodo menos robusto de los cinco: depende de retornos
esperados, y la media historica es un estimador pesimo. Se contrae con
James-Stein hacia la media de seccion cruzada, y aun asi el resultado se
reporta con esa advertencia. Se resuelve con la transformacion de
Charnes-Cooper (min y'Sigma y  s.a.  mu'y = 1), valida porque todas las
restricciones son homogeneas.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constraints import ConstructionParams, cap_leg, exposures

METHODS = ("equal_weight", "inverse_vol", "risk_parity", "min_variance", "max_sharpe")
_SOLVERS = ("CLARABEL", "OSQP", "SCS")


@dataclass
class ConstructionResult:
    weights: pd.Series
    method: str
    status: str = "ok"
    note: str = ""


def _solve(problem: cp.Problem) -> str:
    last = "no resuelto"
    for solver in _SOLVERS:
        try:
            problem.solve(solver=solver)
        except (cp.error.SolverError, ValueError) as exc:
            last = f"{solver}: {exc}"
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return problem.status
        last = f"{solver}: {problem.status}"
    return last


def _split(signs: pd.Series) -> tuple[pd.Index, pd.Index]:
    return signs.index[signs > 0], signs.index[signs < 0]


def equal_weight(signs: pd.Series, params: ConstructionParams) -> pd.Series:
    longs, shorts = _split(signs)
    w = pd.Series(0.0, index=signs.index)
    w[longs] = cap_leg(pd.Series(1.0, index=longs), params.long_leg, params.max_position)
    w[shorts] = -cap_leg(pd.Series(1.0, index=shorts), params.short_leg, params.max_position)
    return w


def inverse_vol(signs: pd.Series, vol: pd.Series, params: ConstructionParams) -> pd.Series:
    longs, shorts = _split(signs)
    inv = 1.0 / vol.reindex(signs.index).replace(0.0, np.nan)
    inv = inv.fillna(inv.median())
    w = pd.Series(0.0, index=signs.index)
    w[longs] = cap_leg(inv[longs], params.long_leg, params.max_position)
    w[shorts] = -cap_leg(inv[shorts], params.short_leg, params.max_position)
    return w


def _erc_long_only(sigma: np.ndarray) -> np.ndarray:
    """Pesos (sin escalar) con contribuciones iguales al riesgo de `sigma`."""
    n = len(sigma)

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        sx = sigma @ x
        return 0.5 * x @ sx - np.log(x).sum() / n, sx - 1.0 / (n * x)

    x0 = 1.0 / np.sqrt(np.diag(sigma))
    res = minimize(objective, x0, jac=True, method="L-BFGS-B",
                   bounds=[(1e-10, None)] * n, options={"maxiter": 1000, "ftol": 1e-14})
    return res.x


def risk_parity(signs: pd.Series, cov: pd.DataFrame, params: ConstructionParams) -> tuple[pd.Series, str]:
    w = pd.Series(0.0, index=signs.index)
    capped = False
    for names, total, sign in ((list(signs.index[signs > 0]), params.long_leg, 1.0),
                               (list(signs.index[signs < 0]), params.short_leg, -1.0)):
        if not names:
            continue
        x = pd.Series(_erc_long_only(cov.loc[names, names].to_numpy(float)), index=names)
        x = x / x.sum() * total
        if (x > params.max_position + 1e-9).any():
            x = cap_leg(x, total, params.max_position)
            capped = True
        w[names] = sign * x
    # El tope rompe la igualdad exacta de contribuciones; se dice.
    note = "tope por nombre activo: contribuciones aproximadamente iguales, no exactas" if capped else ""
    return w, note


def _leg_constraints(w: cp.Variable, signs: pd.Series, params: ConstructionParams,
                     scale: cp.Expression | float = 1.0, betas: pd.Series | None = None) -> list:
    s = signs.to_numpy(float)
    long_mask = s > 0
    cons = [
        cp.multiply(s, w) >= 0,                                   # signo de research
        cp.abs(w) <= params.max_position * scale,                 # tope por nombre
    ]
    if params.beta_neutral and betas is not None:
        # Beta cero Y neta fijada a la vez suelen no caber: si los largos tienen
        # mas beta que los cortos, igualar dolares deja beta positiva. Se fija la
        # BRUTA (lineal con signos conocidos: suma de s_i w_i) y la neta queda
        # libre, que es el precio declarado de neutralizar la beta.
        cons += [s @ w == params.gross * scale,
                 betas.reindex(signs.index).to_numpy(float) @ w == 0]
    else:
        cons += [cp.sum(w[np.where(long_mask)[0]]) == params.long_leg * scale,
                 cp.sum(w[np.where(~long_mask)[0]]) == -params.short_leg * scale]
    return cons


def min_variance(signs: pd.Series, cov: pd.DataFrame, params: ConstructionParams,
                 betas: pd.Series | None = None) -> tuple[pd.Series | None, str]:
    names = list(signs.index)
    sigma = cov.loc[names, names].to_numpy(float)
    sigma = 0.5 * (sigma + sigma.T)
    w = cp.Variable(len(names))
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))),
                      _leg_constraints(w, signs, params, betas=betas))
    status = _solve(prob)
    if w.value is None:
        return None, f"min_variance sin solucion ({status})"
    return pd.Series(w.value, index=names), ""


def shrink_means(returns: pd.DataFrame) -> pd.Series:
    """James-Stein hacia la media de seccion cruzada.

    c = min(1, (N - 3) x var(media estimada) / suma((mu_i - media)^2)),
    var(media estimada) = media de sigma_i^2 / T. Con N <= 3 no hay contraccion
    posible y se devuelve la media muestral.
    """
    r = returns.dropna()
    mu = r.mean()
    n, t = len(mu), len(r)
    if n <= 3 or t < 2:
        return mu
    grand = mu.mean()
    dispersion = float(((mu - grand) ** 2).sum())
    noise = float(r.var(ddof=1).mean() / t)
    c = min(1.0, (n - 3) * noise / dispersion) if dispersion > 0 else 1.0
    return grand + (1.0 - c) * (mu - grand)


def max_sharpe(signs: pd.Series, cov: pd.DataFrame, mu: pd.Series, params: ConstructionParams,
               betas: pd.Series | None = None) -> tuple[pd.Series | None, str]:
    names = list(signs.index)
    sigma = cov.loc[names, names].to_numpy(float)
    sigma = 0.5 * (sigma + sigma.T)
    m = mu.reindex(names).to_numpy(float)
    y = cp.Variable(len(names))
    kappa = cp.Variable(nonneg=True)
    cons = _leg_constraints(y, signs, params, scale=kappa, betas=betas) + [m @ y == 1]
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(sigma))), cons)
    status = _solve(prob)
    if y.value is None or kappa.value is None or kappa.value <= 1e-12:
        return None, (
            "max_sharpe sin solucion: con los retornos esperados contraidos, ninguna "
            f"cartera con estos signos tiene Sharpe ex-ante positivo ({status})"
        )
    return pd.Series(y.value / kappa.value, index=names), (
        "depende de retornos esperados historicos contraidos: el menos robusto de los cinco"
    )


def build(
    method: str,
    signs: pd.Series,
    *,
    cov: pd.DataFrame,
    vol: pd.Series,
    params: ConstructionParams,
    mu: pd.Series | None = None,
    betas: pd.Series | None = None,
) -> ConstructionResult:
    if method not in METHODS:
        raise ValueError(f"metodo '{method}' desconocido. Opciones: {METHODS}")
    signs = signs.astype(float)
    note = ""
    if method == "equal_weight":
        w = equal_weight(signs, params)
    elif method == "inverse_vol":
        w = inverse_vol(signs, vol, params)
    elif method == "risk_parity":
        w, note = risk_parity(signs, cov, params)
    elif method == "min_variance":
        w, note = min_variance(signs, cov, params, betas)
    else:
        if mu is None:
            raise ValueError("max_sharpe necesita retornos esperados")
        w, note = max_sharpe(signs, cov, mu, params, betas)

    if w is None:
        return ConstructionResult(pd.Series(dtype=float), method, status="failed", note=note)

    # Neutralidad de beta: min_variance y max_sharpe la llevan como restriccion
    # dentro del problema; los metodos de formula cerrada la aplican despues,
    # escalando las patas (ver `neutral.beta_neutral_legs`).
    if params.beta_neutral and method in ("equal_weight", "inverse_vol", "risk_parity"):
        if betas is None:
            raise ValueError("beta_neutral necesita betas")
        from .neutral import beta_neutral_legs  # evita import circular

        w, beta_note = beta_neutral_legs(w, betas, params)
        note = "; ".join(n for n in (note, beta_note) if n)

    w = w.where(w.abs() > 1e-10, 0.0)
    return ConstructionResult(w, method, note=note)


def summary(w: pd.Series) -> dict[str, float]:
    return exposures(w)
