"""Restricciones comunes a todos los metodos de construccion.

Convencion de pesos: fraccion del CAPITAL, con signo. Largo > 0, corto < 0.

    exposicion larga   L = suma de w > 0
    exposicion corta   S = suma de |w < 0|
    bruta              G = L + S
    neta               N = L - S

Dados G y N, cada pata queda fijada:  L = (G + N) / 2,  S = (G - N) / 2.
Con G = 2 y N = 0: 100% largo y 100% corto, neutral en dolares.

**El signo lo pone research y ningun optimizador lo cambia.** Por eso la bruta
es LINEAL en los pesos (L - S con signos conocidos) y todos los problemas de
optimizacion son convexos. Dejar que el optimizador elija el signo convertiria
la bruta en una norma L1 y, peor, le permitiria dar la vuelta a la tesis de
research para minimizar varianza.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConstructionParams:
    gross: float = 2.0
    net: float = 0.0
    max_position: float = 0.20
    beta_neutral: bool = False

    @property
    def long_leg(self) -> float:
        return (self.gross + self.net) / 2.0

    @property
    def short_leg(self) -> float:
        return (self.gross - self.net) / 2.0


def cap_leg(raw: pd.Series, total: float, cap: float) -> pd.Series:
    """Escala una pata a `total` sin que ningun nombre pase de `cap`.

    Llenado por niveles: se topan los que se pasan y el exceso se reparte entre
    los demas en proporcion a su peso. Lanza error si es imposible, en vez de
    devolver una pata que incumple el tope en silencio.
    """
    x = raw.clip(lower=0.0).astype(float)
    if x.sum() <= 0:
        x = pd.Series(1.0, index=raw.index)
    if len(x) * cap < total - 1e-12:
        raise ValueError(
            f"{len(x)} nombres con tope {cap:.0%} no suman {total:.0%}: la pata no cabe"
        )
    x = x / x.sum() * total
    for _ in range(100):
        over = x > cap + 1e-12
        if not over.any():
            break
        excess = float((x[over] - cap).sum())
        x[over] = cap
        free = ~over & (x < cap)
        if not free.any():
            break
        x[free] += excess * x[free] / x[free].sum()
    return x


def exposures(weights: pd.Series) -> dict[str, float]:
    w = weights.fillna(0.0)
    long = float(w[w > 0].sum())
    short = float(-w[w < 0].sum())
    return {"long": long, "short": short, "gross": long + short, "net": long - short}


def check(weights: pd.Series, signs: pd.Series, params: ConstructionParams, tol: float = 1e-6) -> list[str]:
    """Lista de violaciones. Vacia = la cartera cumple todo."""
    problems = []
    w = weights.reindex(signs.index).fillna(0.0)
    wrong_sign = w[(np.sign(w) != signs) & (w.abs() > tol)]
    if len(wrong_sign):
        problems.append(f"signo contrario al research: {list(wrong_sign.index)}")
    over = w[w.abs() > params.max_position + tol]
    if len(over):
        problems.append(f"sobre el tope de {params.max_position:.0%}: {list(over.index)}")
    e = exposures(w)
    if abs(e["gross"] - params.gross) > 1e-4:
        problems.append(f"bruta {e['gross']:.4f} != {params.gross}")
    return problems
