"""Correlaciones y grupos de activos que se mueven juntos. Funciones puras.

**Por que importa en una cartera long/short:** el objetivo es que el corto cubra
el largo. Si los largos estan muy correlacionados ENTRE SI, la cartera larga es
en realidad una o dos apuestas repetidas, no cinco. Y si largos y cortos estan
poco correlacionados entre ellos, la "cobertura" no cubre nada.

Clustering jerarquico con distancia d = sqrt(0,5 x (1 - rho)), la de Mantegna
(1999): es una metrica de verdad (cumple la desigualdad triangular), cosa que
1 - rho no hace. Enlace medio. Se corta en `rho_threshold`: activos con
correlacion media >= 0,6 quedan en el mismo grupo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def average_pairwise(corr: pd.DataFrame) -> float:
    values = corr.to_numpy()
    mask = ~np.eye(len(values), dtype=bool)
    return float(np.nanmean(values[mask])) if mask.any() else float("nan")


def rolling_average_correlation(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """Correlacion media entre pares, en ventana movil. Sube en las crisis."""
    out = {}
    clean = returns.dropna(how="all")
    for end in range(window, len(clean) + 1, 5):
        block = clean.iloc[end - window:end].dropna(axis=1, thresh=int(window * 0.8))
        if block.shape[1] >= 2:
            out[clean.index[end - 1]] = average_pairwise(block.corr())
    return pd.Series(out)


def rolling_pair_correlation(a: pd.Series, b: pd.Series, window: int = 63) -> pd.Series:
    return a.rolling(window).corr(b)


def clusters(corr: pd.DataFrame, rho_threshold: float = 0.6) -> dict[int, list[str]]:
    if len(corr) < 2:
        return {1: list(corr.index)}
    distance = np.sqrt(np.clip(0.5 * (1.0 - corr.to_numpy()), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    z = linkage(squareform(distance, checks=False), method="average")
    cut = float(np.sqrt(0.5 * (1.0 - rho_threshold)))
    labels = fcluster(z, t=cut, criterion="distance")
    groups: dict[int, list[str]] = {}
    for ticker, label in zip(corr.index, labels):
        groups.setdefault(int(label), []).append(ticker)
    return groups


def linkage_matrix(corr: pd.DataFrame) -> np.ndarray:
    distance = np.sqrt(np.clip(0.5 * (1.0 - corr.to_numpy()), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    return linkage(squareform(distance, checks=False), method="average")


def correlation_summary(returns: pd.DataFrame, longs: list[str], shorts: list[str]) -> dict[str, object]:
    cols = [t for t in (*longs, *shorts) if t in returns.columns]
    r = returns[cols].dropna(how="all")
    pearson = r.corr(method="pearson")
    spearman = r.corr(method="spearman")
    L = [t for t in longs if t in cols]
    S = [t for t in shorts if t in cols]
    cross = pearson.loc[L, S].to_numpy() if L and S else np.array([[np.nan]])
    return {
        "pearson": pearson,
        "spearman": spearman,
        "avg_pairwise": average_pairwise(pearson),
        "avg_within_long": average_pairwise(pearson.loc[L, L]) if len(L) > 1 else float("nan"),
        "avg_within_short": average_pairwise(pearson.loc[S, S]) if len(S) > 1 else float("nan"),
        "avg_long_short": float(np.nanmean(cross)),
        "clusters": clusters(pearson),
    }
