"""VaR y Expected Shortfall: cuanto se puede perder en un dia malo, y en uno peor.

**Convenio de signo.** VaR y ES se reportan como PERDIDA POSITIVA en fraccion
del NAV. "VaR 99% a 1 dia = 3.1%" quiere decir que uno de cada cien dias se
espera perder MAS del 3.1%. El ES es la perdida media de ese uno de cada cien:
el VaR dice donde empieza la cola, el ES cuanto pesa.

Cuatro metodos, porque cada uno falla de una forma distinta y conocida:

- **Historico.** Cuantil empirico. No supone distribucion, pero no puede
  imaginar un dia peor que el peor de la muestra, y pesa igual 2013 que hoy.
- **Normal.** Media y desviacion. Subestima la cola en renta variable, que es
  leptocurtica. Se reporta como SUELO, no como estimacion.
- **Cornish-Fisher.** Corrige el cuantil normal por asimetria y curtosis. Mejor
  en la cola moderada (95%); con curtosis extrema se vuelve inestable.
- **Simulacion historica filtrada** (FHS, Barone-Adesi et al. 1999). Residuos
  estandarizados por la volatilidad EWMA de cada dia, reescalados con la de
  HOY. Conserva la forma de la cola historica y responde al regimen actual. Es
  el VaR de referencia del reporte.

**Horizontes de mas de un dia.** Historico y Cornish-Fisher usan retornos
compuestos de `h` sesiones solapados: capturan la autocorrelacion, a cambio de
observaciones que no son independientes. Normal y FHS escalan con la raiz del
tiempo, que supone retornos independientes: si la volatilidad de hoy es alta y
revierte, el VaR mensual FHS la exagera.

Un VaR sin backtest es una opinion. `var_backtest` reestima el VaR cada dia con
la informacion del dia anterior, cuenta las excepciones y aplica Kupiec
(cobertura), Christoffersen (independencia) y el semaforo de Basilea.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

_NORMAL = NormalDist()
_MIN_OBS = 30

METHODS = ("historico", "normal", "cornish_fisher", "fhs")


def _clean(returns: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(pd.Series(returns).dropna(), dtype=float)
    return values[np.isfinite(values)]


def _tail(confidence: float) -> float:
    if not 0.5 < confidence < 1.0:
        raise ValueError(
            f"confianza {confidence}: debe estar entre 0.5 y 1 (0.99, no 99)"
        )
    return 1.0 - confidence


def compound(returns: pd.Series, horizon_d: int) -> pd.Series:
    """Retornos compuestos de `horizon_d` sesiones, solapados."""
    if horizon_d <= 1:
        return returns.dropna()
    log = np.log1p(returns.dropna().astype(float))
    return np.expm1(log.rolling(horizon_d).sum()).dropna()


# ---------------------------------------------------------------------------
#  Estimadores puntuales
# ---------------------------------------------------------------------------


def historical_var(returns, confidence: float) -> float:
    r = _clean(returns)
    if len(r) < _MIN_OBS:
        return float("nan")
    return float(-np.quantile(r, _tail(confidence)))


def historical_es(returns, confidence: float) -> float:
    r = _clean(returns)
    if len(r) < _MIN_OBS:
        return float("nan")
    cutoff = np.quantile(r, _tail(confidence))
    return float(-r[r <= cutoff].mean())


def normal_var(returns, confidence: float) -> float:
    r = _clean(returns)
    if len(r) < _MIN_OBS:
        return float("nan")
    z = _NORMAL.inv_cdf(_tail(confidence))
    return float(-(r.mean() + z * r.std(ddof=1)))


def normal_es(returns, confidence: float) -> float:
    r = _clean(returns)
    if len(r) < _MIN_OBS:
        return float("nan")
    p = _tail(confidence)
    z = _NORMAL.inv_cdf(p)
    return float(-(r.mean() - r.std(ddof=1) * _NORMAL.pdf(z) / p))


def _moments(r: np.ndarray) -> tuple[float, float, float, float]:
    mu = r.mean()
    sd = r.std(ddof=1)
    centered = (r - mu) / sd if sd > 0 else np.zeros_like(r)
    skew = float(np.mean(centered ** 3))
    excess_kurt = float(np.mean(centered ** 4) - 3.0)
    return float(mu), float(sd), skew, excess_kurt


def _cf_quantile(u: np.ndarray, skew: float, kurt: float) -> np.ndarray:
    z = np.array([_NORMAL.inv_cdf(float(x)) for x in np.atleast_1d(u)])
    return (z + (z ** 2 - 1) * skew / 6 + (z ** 3 - 3 * z) * kurt / 24
            - (2 * z ** 3 - 5 * z) * skew ** 2 / 36)


def cornish_fisher_var(returns, confidence: float) -> float:
    r = _clean(returns)
    if len(r) < _MIN_OBS:
        return float("nan")
    mu, sd, skew, kurt = _moments(r)
    z_cf = _cf_quantile(np.array([_tail(confidence)]), skew, kurt)[0]
    return float(-(mu + z_cf * sd))


def cornish_fisher_es(returns, confidence: float, grid: int = 400) -> float:
    """Media del cuantil Cornish-Fisher sobre la cola: la integral, numerica."""
    r = _clean(returns)
    if len(r) < _MIN_OBS:
        return float("nan")
    mu, sd, skew, kurt = _moments(r)
    p = _tail(confidence)
    u = p * (np.arange(grid) + 0.5) / grid
    return float(-(mu + _cf_quantile(u, skew, kurt).mean() * sd))


# ---------------------------------------------------------------------------
#  Volatilidad EWMA y simulacion historica filtrada
# ---------------------------------------------------------------------------


def ewma_sigma(returns: pd.Series, lam: float) -> tuple[pd.Series, float]:
    """Volatilidad EWMA diaria.

    Devuelve `(sigma, sigma_next)`. `sigma[t]` usa solo retornos hasta `t-1`:
    es la prevision que se tenia al cierre anterior, la que sirve para el VaR
    del dia `t`. `sigma_next` es la prevision para la proxima sesion.

    La semilla es la varianza de las primeras 60 observaciones. Eso mira 60
    dias hacia delante al principio de la serie; el backtest del VaR empieza
    despues de su ventana (250 sesiones) y no lo hereda.
    """
    r = returns.dropna().astype(float)
    values = r.to_numpy()
    if len(values) == 0:
        return pd.Series(dtype="float64"), float("nan")
    seed_n = min(len(values), 60)
    variance = np.empty(len(values) + 1)
    variance[0] = np.var(values[:seed_n], ddof=1) if seed_n > 1 else values[0] ** 2
    for t in range(1, len(values) + 1):
        variance[t] = lam * variance[t - 1] + (1.0 - lam) * values[t - 1] ** 2
    sigma = np.sqrt(variance)
    return pd.Series(sigma[:-1], index=r.index), float(sigma[-1])


def fhs_var_es(returns: pd.Series, confidence: float, lam: float,
               horizon_d: int = 1) -> tuple[float, float]:
    """VaR y ES por simulacion historica filtrada, reescalados a la vol de hoy."""
    r = returns.dropna().astype(float)
    if len(r) < _MIN_OBS:
        return float("nan"), float("nan")
    sigma, sigma_next = ewma_sigma(r, lam)
    standardized = (r / sigma.replace(0.0, np.nan)).dropna()
    scaled = standardized * sigma_next
    root = math.sqrt(horizon_d)
    return historical_var(scaled, confidence) * root, historical_es(scaled, confidence) * root


def var_table(returns: pd.Series, confidences, horizons_d, lam: float) -> pd.DataFrame:
    """VaR y ES por metodo, confianza y horizonte, en fraccion de NAV."""
    r = returns.dropna().astype(float)
    rows = []
    for h in horizons_d:
        overlapping = compound(r, h)
        mu, sd = r.mean(), r.std(ddof=1)
        for c in confidences:
            p = _tail(c)
            z = _NORMAL.inv_cdf(p)
            normal_v = -(mu * h + z * sd * math.sqrt(h))
            normal_e = -(mu * h - sd * math.sqrt(h) * _NORMAL.pdf(z) / p)
            fhs_v, fhs_e = fhs_var_es(r, c, lam, h)
            for method, v, e in (
                ("historico", historical_var(overlapping, c), historical_es(overlapping, c)),
                ("normal", normal_v, normal_e),
                ("cornish_fisher", cornish_fisher_var(overlapping, c),
                 cornish_fisher_es(overlapping, c)),
                ("fhs", fhs_v, fhs_e),
            ):
                rows.append({"method": method, "confidence": c, "horizon_d": int(h),
                             "var": float(v), "es": float(e)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Estadisticas de cola
# ---------------------------------------------------------------------------


def tail_statistics(returns: pd.Series) -> dict[str, object]:
    """Lo que el VaR resume y el comite deberia ver sin resumir."""
    r = returns.dropna().astype(float)
    if len(r) < _MIN_OBS:
        return {}
    mu, sd, skew, kurt = _moments(r.to_numpy())
    threshold = mu - 3.0 * sd
    q05, q95 = np.quantile(r, 0.05), np.quantile(r, 0.95)
    return {
        "n_days": int(len(r)),
        "skew": skew,
        "excess_kurtosis": kurt,
        "worst_day": float(r.min()),
        "worst_day_date": r.idxmin(),
        "best_day": float(r.max()),
        "best_day_date": r.idxmax(),
        # Frecuencia de dias por debajo de -3 sigma frente a la normal (0.135%).
        # Si sale cinco veces mayor, el VaR normal esta mintiendo por la cola.
        "freq_below_3sigma": float((r < threshold).mean()),
        "normal_freq_below_3sigma": float(_NORMAL.cdf(-3.0)),
        "tail_ratio": float(abs(q95) / abs(q05)) if q05 != 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
#  Backtest del VaR
# ---------------------------------------------------------------------------


def rolling_var_forecasts(returns: pd.Series, confidence: float, window: int,
                          lam: float) -> pd.DataFrame:
    """VaR de cada dia con informacion hasta el cierre anterior.

    Tres modelos: historico de ventana movil, normal con volatilidad EWMA (el
    RiskMetrics clasico) y FHS. Todos con `shift(1)` o con una sigma que ya es
    prevision: ningun VaR del dia `t` ve el retorno de `t`.
    """
    r = returns.dropna().astype(float)
    p = _tail(confidence)
    z = _NORMAL.inv_cdf(p)
    sigma, _ = ewma_sigma(r, lam)

    historical = -r.rolling(window).quantile(p).shift(1)
    ewma_normal = -z * sigma
    standardized = r / sigma.replace(0.0, np.nan)
    fhs = -standardized.rolling(window).quantile(p).shift(1) * sigma

    frame = pd.DataFrame({
        "return": r,
        "historico": historical,
        "ewma_normal": ewma_normal,
        "fhs": fhs,
    })
    # Se descarta el arranque: la ventana tiene que estar llena para los tres
    # modelos, o se compararian con muestras distintas.
    return frame.iloc[window + 1:].dropna()


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * math.log(y)


def _chi2_sf(stat: float, df: int) -> float:
    """Cola derecha de la chi cuadrado para 1 y 2 grados de libertad."""
    if not np.isfinite(stat) or stat < 0:
        return float("nan")
    if df == 1:
        return math.erfc(math.sqrt(stat / 2.0))
    if df == 2:
        return math.exp(-stat / 2.0)
    raise ValueError("solo se usan 1 o 2 grados de libertad")


def kupiec_pof(n_obs: int, n_exceptions: int, tail: float) -> tuple[float, float]:
    """Proportion of failures (Kupiec 1995). Devuelve (LR, valor p).

    H0: la frecuencia de excepciones es la que promete el nivel de confianza.
    Un valor p bajo significa que el VaR falla demasiado -- o demasiado poco,
    que tambien es un error: un VaR que nunca se excede es capital parado.
    """
    if n_obs <= 0:
        return float("nan"), float("nan")
    x, n = n_exceptions, n_obs
    observed = x / n
    log_null = _xlogy(n - x, 1.0 - tail) + _xlogy(x, tail)
    log_alt = _xlogy(n - x, 1.0 - observed) + _xlogy(x, observed)
    lr = -2.0 * (log_null - log_alt)
    return float(lr), _chi2_sf(lr, 1)


def christoffersen_independence(exceptions) -> tuple[float, float]:
    """Independencia de las excepciones (Christoffersen 1998). (LR, valor p).

    Un VaR puede acertar en la frecuencia y fallar en la forma: si las
    excepciones llegan en racimos, el modelo no reacciona a los cambios de
    volatilidad y las perdidas grandes vienen juntas, que es cuando duelen.
    """
    hits = np.asarray(exceptions, dtype=bool)
    if len(hits) < 2:
        return float("nan"), float("nan")
    prev, curr = hits[:-1], hits[1:]
    n00 = int(np.sum(~prev & ~curr))
    n01 = int(np.sum(~prev & curr))
    n10 = int(np.sum(prev & ~curr))
    n11 = int(np.sum(prev & curr))
    if n01 + n11 == 0:
        return 0.0, 1.0
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    log_null = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    log_alt = (_xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01)
               + _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11))
    lr = max(-2.0 * (log_null - log_alt), 0.0)
    return float(lr), _chi2_sf(lr, 1)


def basel_zone(n_exceptions: int, n_obs: int, tail: float) -> str:
    """Semaforo de Basilea por probabilidad binomial acumulada.

    Verde si un modelo correcto produce tantas excepciones o menos con
    probabilidad < 95%; amarillo hasta 99.99%; rojo por encima. Con 250
    sesiones al 99% da los cortes clasicos: verde 0-4, amarillo 5-9, rojo 10+.
    """
    cumulative = sum(
        math.comb(n_obs, k) * tail ** k * (1.0 - tail) ** (n_obs - k)
        for k in range(n_exceptions + 1)
    )
    # Tolerancia de redondeo: la tabla de Basilea redondea a 99.99%.
    if cumulative < 0.95:
        return "verde"
    if cumulative < 0.9999 - 1e-6:
        return "amarillo"
    return "rojo"


def var_backtest(returns: pd.Series, confidence: float, window: int, lam: float,
                 recent_d: int = 250) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest de los tres modelos de VaR. Devuelve (resumen, previsiones)."""
    forecasts = rolling_var_forecasts(returns, confidence, window, lam)
    tail = _tail(confidence)
    rows = []
    for model in ("historico", "ewma_normal", "fhs"):
        if forecasts.empty:
            break
        hits = (forecasts["return"] < -forecasts[model]).to_numpy()
        n = len(hits)
        x = int(hits.sum())
        lr_pof, p_pof = kupiec_pof(n, x, tail)
        lr_ind, p_ind = christoffersen_independence(hits)
        recent = hits[-recent_d:]
        rows.append({
            "model": model,
            "n_obs": n,
            "expected": n * tail,
            "exceptions": x,
            "rate": x / n if n else float("nan"),
            "kupiec_p": p_pof,
            "christoffersen_p": p_ind,
            # Cobertura condicional: las dos pruebas juntas, chi2 con 2 g.l.
            "conditional_p": _chi2_sf(lr_pof + lr_ind, 2),
            "recent_obs": int(len(recent)),
            "recent_exceptions": int(recent.sum()),
            "zone": basel_zone(int(recent.sum()), int(len(recent)), tail),
        })
    return pd.DataFrame(rows), forecasts
