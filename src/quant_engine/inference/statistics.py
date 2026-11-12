"""Inferencia por nombre y para la canasta: lo que un numero puntual no dice.

- **Paseo aleatorio** (Lo y MacKinlay, 1988): ratio de varianzas con el z*
  robusto a heterocedasticidad. VR > 1 con p < 0,05 es tendencia (los
  retornos se encadenan); VR < 1, reversion. Sin rechazo, el precio no deja
  huella explotable a ese horizonte, y un stop por ATR es tan bueno como otro.
- **Normalidad** (Jarque-Bera): con colas gordas el VaR normal se queda corto;
  se da junto al historico y al de Cornish-Fisher.
- **Regimen de volatilidad** (GARCH(1,1), el de `sfc_tfsig.risk.garch`): la
  vol condicional de hoy frente a la de largo plazo. Un ATR medido en un pico
  de volatilidad pone el stop lejos y achica la posicion; en calma, al reves.
- **Intervalo del E(R) de la regla**: bootstrap por bloques moviles. Las
  entradas diarias se solapan; remuestrear operaciones sueltas daria un
  intervalo falsamente estrecho.
- **Sharpe probabilistico y deflactado** (Bailey y Lopez de Prado, 2012 y
  2014): probabilidad de que el Sharpe verdadero supere un umbral, con la
  asimetria y la curtosis de la muestra, y descontando cuantas variantes se
  probaron antes de mirar este.
- **Pruebas multiples**: Holm (controla la probabilidad de un solo falso
  positivo; para decidir una regla) y Benjamini-Hochberg (controla la
  proporcion de falsos positivos; para leer muchos nombres a la vez). Con 50
  nombres y dos horizontes, un p < 0,05 suelto saldria unas cinco veces por
  pura suerte.

Funciones puras: sin red ni disco. scipy no es dependencia del proyecto: la
normal sale de `statistics.NormalDist`.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from sfc_tfsig.risk import garch as garch_mod
from sfc_tfsig.risk import var as var_mod

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329
TRADING_DAYS = 252


def variance_ratio_test(returns: pd.Series, q: int) -> dict[str, float]:
    """Ratio de varianzas de Lo y MacKinlay con z* robusto a heterocedasticidad.

    VR(q) = var(retorno de q dias) / (q x var(retorno diario)), con sumas
    solapadas y la correccion de sesgo del articulo. z* = sqrt(T)(VR - 1) /
    sqrt(theta), theta = sum_j [2(q - j)/q]^2 delta(j): con retornos iid,
    delta(j) = 1 y theta es la varianza homocedastica 2(2q - 1)(q - 1)/(3q).
    """
    r = pd.Series(returns).dropna().to_numpy(float)
    n = len(r)
    if q < 2 or n < 20 * q:
        return {}
    mu = r.mean()
    d = r - mu
    d2 = d ** 2
    var1 = d2.sum() / (n - 1)
    if var1 <= 0:
        return {}
    rq = np.convolve(r, np.ones(q), mode="valid")
    m = q * (n - q + 1) * (1 - q / n)
    vr = float(((rq - q * mu) ** 2).sum() / m / var1)
    denom = d2.sum() ** 2
    theta = sum((2 * (q - j) / q) ** 2 * n * float((d2[j:] * d2[:-j]).sum()) / denom for j in range(1, q))
    z = math.sqrt(n) * (vr - 1) / math.sqrt(theta) if theta > 0 else float("nan")
    p = 2 * (1 - _N.cdf(abs(z))) if np.isfinite(z) else float("nan")
    return {"q": q, "vr": vr, "z": z, "p": p, "n": n}


def jarque_bera(returns: pd.Series) -> dict[str, float]:
    """JB = n/6 (S^2 + K^2/4), K en exceso; con 2 grados de libertad, p = exp(-JB/2)."""
    r = pd.Series(returns).dropna().to_numpy(float)
    n = len(r)
    if n < 30:
        return {}
    d = r - r.mean()
    s2 = (d ** 2).mean()
    if s2 <= 0:
        return {}
    skew = float((d ** 3).mean() / s2 ** 1.5)
    kurt = float((d ** 4).mean() / s2 ** 2 - 3.0)
    jb = n / 6.0 * (skew ** 2 + kurt ** 2 / 4.0)
    return {"jb": jb, "p": math.exp(-jb / 2.0), "skew": skew, "excess_kurtosis": kurt, "n": n}


def tail_risk(returns: pd.Series, confidence: float = 0.95) -> dict[str, float]:
    """VaR y ES a 1 dia: historico, normal y Cornish-Fisher (perdidas positivas)."""
    r = pd.Series(returns).dropna()
    if len(r) < 100:
        return {}
    return {
        "confidence": confidence,
        "var_hist": var_mod.historical_var(r, confidence),
        "es_hist": var_mod.historical_es(r, confidence),
        "var_normal": var_mod.normal_var(r, confidence),
        "var_cf": var_mod.cornish_fisher_var(r, confidence),
    }


def volatility_regime(returns: pd.Series) -> dict[str, float | str]:
    """GARCH(1,1): vol condicional de hoy, de largo plazo, persistencia y vida media."""
    r = pd.Series(returns).dropna()
    try:
        fit = garch_mod.fit_garch(r)
    except ValueError:
        return {}
    ratio = fit.current_vol / fit.long_run_vol if fit.long_run_vol > 0 else float("nan")
    label = ("elevated" if ratio >= 1.2 else "calm" if ratio <= 0.8 else "normal") if np.isfinite(ratio) else ""
    return {"current_vol": fit.current_vol, "long_run_vol": fit.long_run_vol, "ratio": ratio,
            "persistence": fit.persistence, "half_life_d": fit.half_life_d, "label": label}


def block_bootstrap_mean(x, block: int, *, n_boot: int = 2000, level: float = 0.90,
                         seed: int = 0) -> dict[str, float]:
    """Intervalo de la media por bootstrap de bloques moviles (Kunsch, 1989).

    `block` es el solape de las observaciones (en la regla, el horizonte de la
    operacion). Devuelve el intervalo y la fraccion de remuestras con media
    positiva: no es un p-valor, es cuan a menudo el signo se sostiene.
    """
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(v)
    block = int(max(1, min(block, n // 2 if n >= 2 else 1)))
    if n < 2 * block or n < 20:
        return {}
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = v[idx].mean(axis=1)
    tail = (1 - level) / 2
    return {"mean": float(v.mean()), "lo": float(np.quantile(means, tail)),
            "hi": float(np.quantile(means, 1 - tail)), "p_positive": float((means > 0).mean()),
            "level": level, "block": block}


def holm(pvalues) -> np.ndarray:
    """p-valores ajustados de Holm (1979), en el orden de entrada; NaN se respeta."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    if not m:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    out[order] = np.maximum.accumulate(adj)
    return out


def benjamini_hochberg(pvalues) -> np.ndarray:
    """q-valores de Benjamini y Hochberg (1995), en el orden de entrada; NaN se respeta."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    if not m:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    adj = np.minimum(1.0, m * p[order] / np.arange(1, m + 1))
    out[order] = np.minimum.accumulate(adj[::-1])[::-1]
    return out


def adjust_random_walk(stats: dict[str, dict]) -> dict[str, dict]:
    """Anade a cada prueba de paseo aleatorio su q-valor BH sobre toda la canasta.

    Una prueba por nombre y horizonte: la familia es la canasta entera. El
    veredicto de la ficha sale del q-valor, no del p suelto.
    """
    cells = [(t, q) for t, s in stats.items() for q, v in (s.get("random_walk") or {}).items() if v]
    qv = benjamini_hochberg([stats[t]["random_walk"][q]["p"] for t, q in cells])
    for (t, q), value in zip(cells, qv, strict=True):
        stats[t]["random_walk"][q]["q_value"] = float(value)
        stats[t]["random_walk"][q]["family"] = len(cells)
    return stats


def _sr_moments(returns) -> tuple[float, float, float, int]:
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(r)
    if n < 30:
        return float("nan"), float("nan"), float("nan"), n
    sd = r.std(ddof=1)
    if sd <= 0:
        return float("nan"), float("nan"), float("nan"), n
    d = r - r.mean()
    s2 = (d ** 2).mean()
    skew = float((d ** 3).mean() / s2 ** 1.5)
    kurt = float((d ** 4).mean() / s2 ** 2)          # no en exceso: la formula lo pide asi
    return float(r.mean() / sd), skew, kurt, n


def probabilistic_sharpe(returns, sr_benchmark: float = 0.0) -> float:
    """P(Sharpe verdadero > sr_benchmark), Sharpe por periodo (Bailey y Lopez de Prado, 2012).

    PSR = Phi((SR - SR*) sqrt(n - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)).
    Con asimetria negativa y colas gordas, el mismo Sharpe merece menos fe.
    """
    sr, g3, g4, n = _sr_moments(returns)
    if not np.isfinite(sr):
        return float("nan")
    var = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if var <= 0:
        return float("nan")
    return float(_N.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(var)))


def expected_max_sharpe(trial_sharpes) -> float:
    """Sharpe maximo esperado entre N pruebas sin habilidad (Bailey y Lopez de Prado, 2014)."""
    s = np.asarray(pd.Series(trial_sharpes).dropna(), dtype=float)
    n = len(s)
    if n < 2:
        return 0.0
    sd = s.std(ddof=1)
    return float(sd * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n)
                       + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n * math.e))))


def deflated_sharpe(returns, trial_sharpes) -> dict[str, float]:
    """PSR con el umbral en el Sharpe que N pruebas darian por suerte.

    `trial_sharpes` son los Sharpe POR PERIODO de todas las variantes que se
    probaron (incluida esta). Con una sola variante, DSR = PSR contra 0.
    """
    sr_star = expected_max_sharpe(trial_sharpes)
    return {"sr_star": sr_star, "dsr": probabilistic_sharpe(returns, sr_star),
            "psr": probabilistic_sharpe(returns, 0.0), "trials": int(pd.Series(trial_sharpes).notna().sum())}


def name_statistics(returns: pd.Series) -> dict[str, dict]:
    """El paquete de un nombre con sus retornos diarios en USD."""
    r = pd.Series(returns).dropna()
    return {
        "random_walk": {q: variance_ratio_test(r, q) for q in (5, 21)},
        "normality": jarque_bera(r),
        "tail": tail_risk(r, 0.95),
        "vol_regime": volatility_regime(r),
    }


def describe(stats: dict) -> list[str]:
    """El paquete en frases, para la ficha."""
    lines: list[str] = []
    rw = [v for v in (stats.get("random_walk") or {}).values() if v]
    if rw:
        parts = []
        # Con q-valor (canasta), el veredicto lo controla la tasa de falsos
        # positivos de toda la familia; sin el, el p suelto.
        family = max((v.get("family", 1) for v in rw), default=1)
        for v in rw:
            sig = v.get("q_value", v["p"])
            verdict = ("trending" if v["vr"] > 1 else "mean-reverting") if sig < 0.05 else "random walk"
            parts.append(f"VR({v['q']}) {v['vr']:.2f}, z {v['z']:+.1f}, "
                         + (f"q {sig:.2f}" if "q_value" in v else f"p {sig:.2f}") + f" ({verdict})")
        tail = (f" q is the p-value adjusted by Benjamini-Hochberg over the {family} tests in this basket."
                if any("q_value" in v for v in rw) else "")
        lines.append("Random-walk test (Lo-MacKinlay, heteroskedasticity-robust): " + "; ".join(parts) + "." + tail)
    nm, tail = stats.get("normality") or {}, stats.get("tail") or {}
    if nm and tail:
        fat = nm["p"] < 0.01
        lines.append(f"Daily returns {'are not' if fat else 'are close to'} normal (Jarque-Bera p {nm['p']:.3f}; "
                     f"skew {nm['skew']:+.2f}, excess kurtosis {nm['excess_kurtosis']:.1f}): 1-day 95% VaR "
                     f"{tail['var_hist']:.1%} historical, {tail['var_normal']:.1%} normal, {tail['var_cf']:.1%} "
                     f"Cornish-Fisher; ES {tail['es_hist']:.1%}.")
    vr = stats.get("vol_regime") or {}
    if vr:
        hl = f"{vr['half_life_d']:.0f}" if np.isfinite(vr["half_life_d"]) else "no"
        lines.append(f"Volatility regime (GARCH(1,1)): {vr['current_vol']:.0%} today vs {vr['long_run_vol']:.0%} "
                     f"long run, {vr['label']}; persistence {vr['persistence']:.2f}, half-life {hl} sessions.")
    return lines
