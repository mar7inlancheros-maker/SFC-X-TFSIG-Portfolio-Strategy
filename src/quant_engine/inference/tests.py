"""Contrastes estadisticos LONG contra SHORT.

**Advertencia que va en el reporte, no solo aqui:** con 5 nombres por lado, un
contraste de seccion cruzada tiene poquisima potencia. Una diferencia real y
grande puede salir con p = 0,3; una diferencia nula puede salir con p = 0,04 por
azar. Se reportan cuatro contrastes distintos para ver si coinciden, no para
elegir el que mas convenga.

1. **t de Welch**: diferencia de medias sin suponer varianzas iguales.
2. **U de Mann-Whitney**: rangos, robusto a colas gruesas. Con 5 y 5, el p mas
   pequeno posible es 0,008: ningun resultado puede ser mas "significativo".
3. **Permutacion exacta**: con 5 y 5 hay C(10,5) = 252 reparticiones posibles;
   se enumeran TODAS. p = fraccion con diferencia al menos tan extrema como la
   observada.
4. **IC bootstrap** de la diferencia de medias, percentil, semilla fija.

Y un contraste con MUCHA mas potencia: la serie diaria del spread
(cesta larga - cesta corta), con t de Newey-West. Usa ~1.250 observaciones en
vez de 10. Pero ojo: ese contraste mide el pasado de una seleccion hecha HOY, y
es tan in-sample como el resto del modo instantanea.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from sfc_tfsig.validation import newey_west_se, two_sided_p_value

_EXACT_LIMIT = 50_000


def permutation_p(a: np.ndarray, b: np.ndarray, *, seed: int = 0, samples: int = 20_000) -> float:
    """p de dos colas de la diferencia de medias. Exacto si es viable."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.concatenate([a, b])
    observed = abs(a.mean() - b.mean())
    n, k = len(pooled), len(a)

    if math.comb(n, k) <= _EXACT_LIMIT:
        total = pooled.sum()
        extreme = count = 0
        for idx in combinations(range(n), k):
            s = pooled[list(idx)].sum()
            diff = abs(s / k - (total - s) / (n - k))
            extreme += diff >= observed - 1e-12
            count += 1
        return extreme / count

    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(samples):
        perm = rng.permutation(pooled)
        extreme += abs(perm[:k].mean() - perm[k:].mean()) >= observed - 1e-12
    return (extreme + 1) / (samples + 1)


def bootstrap_ci(a: np.ndarray, b: np.ndarray, *, level: float = 0.95,
                 n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = rng.choice(a, (n, len(a))).mean(axis=1) - rng.choice(b, (n, len(b))).mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(diffs, tail)), float(np.quantile(diffs, 1.0 - tail))


def compare_groups(long_values: pd.Series, short_values: pd.Series, *, seed: int = 0) -> dict[str, float]:
    a = long_values.dropna().to_numpy(float)
    b = short_values.dropna().to_numpy(float)
    out = {
        "n_long": len(a),
        "n_short": len(b),
        "long_mean": float(a.mean()) if len(a) else float("nan"),
        "short_mean": float(b.mean()) if len(b) else float("nan"),
        "long_median": float(np.median(a)) if len(a) else float("nan"),
        "short_median": float(np.median(b)) if len(b) else float("nan"),
    }
    out["difference"] = out["long_mean"] - out["short_mean"]
    if len(a) >= 2 and len(b) >= 2:
        t = stats.ttest_ind(a, b, equal_var=False)
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        lo, hi = bootstrap_ci(a, b, seed=seed)
        out.update({
            "welch_t": float(t.statistic), "welch_p": float(t.pvalue),
            "mw_u": float(u.statistic), "mw_p": float(u.pvalue),
            "perm_p": permutation_p(a, b, seed=seed),
            "ci_low": lo, "ci_high": hi,
        })
    return out


def spread_test(long_returns: pd.Series, short_returns: pd.Series, lags: int = 5) -> dict[str, float]:
    """Spread diario largo - corto: media anualizada y t de Newey-West."""
    spread = (long_returns - short_returns).dropna()
    if len(spread) < 60:
        return {}
    se = newey_west_se(spread, lags)
    t = float(spread.mean() / se) if se and se > 0 else float("nan")
    return {
        "observations": int(len(spread)),
        "mean_daily": float(spread.mean()),
        "mean_annual": float(spread.mean() * 252),
        "t_nw": t,
        "p_value": two_sided_p_value(t),
        "hit_rate": float((spread > 0).mean()),
    }
