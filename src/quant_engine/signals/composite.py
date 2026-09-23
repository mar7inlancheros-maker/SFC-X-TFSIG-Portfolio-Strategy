"""Quant Score: z-scores de seccion cruzada combinados con pesos declarados.

Cada componente es la media de uno o mas z-scores, con signo tal que MAS ALTO =
MAS FAVORABLE PARA UNA POSICION LARGA. La tabla de abajo es la definicion
completa; los pesos estan en `config/quant_engine.yaml`.

| Componente | Caracteristica | Signo | Por que |
|---|---|---|---|
| momentum | 12-1, 6M, 3M, precio/SMA200, histograma MACD | + | Jegadeesh-Titman |
| risk_adjusted_return | Sharpe de 1 ano | + | retorno por unidad de riesgo |
| volatility | volatilidad anual | - | anomalia de baja volatilidad |
| mean_reversion | z-score del precio a 20 dias | - | estirado al alza tiende a devolver |
| liquidity | log(volumen medio en dolares) | + | coste de entrar y salir |
| beta | beta frente al benchmark | - | Betting Against Beta (Frazzini-Pedersen) |
| statistical | calidad de tendencia (signo x R2) | + | informacion continua (Da et al.) |

**Con 10 activos, los z-scores son ruidosos**: media y desviacion se estiman con
10 puntos. Un z de +1,2 frente a +0,9 no distingue nada. El score sirve para
ordenar y comparar con research, no para afinar pesos.

**El Quant Score NO sustituye la senal de research.** Se guardan por separado y
se comparan en `agreement.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS: dict[str, list[tuple[str, int]]] = {
    "momentum": [("mom_12_1", 1), ("mom_6m", 1), ("mom_3m", 1),
                 ("price_to_sma_200", 1), ("macd_hist_pct", 1)],
    "risk_adjusted_return": [("sharpe_1y", 1)],
    "volatility": [("vol_annual", -1)],
    "mean_reversion": [("zscore_20d", -1)],
    "liquidity": [("log_adv", 1)],
    "beta": [("beta", -1)],
    "statistical": [("trend_quality", 1)],
}

# Matriz de caracteristicas estandarizadas del apartado de seccion cruzada. Sin
# signo: aqui se describe, no se puntua.
FEATURE_MATRIX_COLUMNS = {
    "momentum": "mom_12_1",
    "volatility": "vol_annual",
    "sharpe": "sharpe_1y",
    "beta": "beta",
    "drawdown": "max_dd",
    "liquidity": "log_adv",
    "return_1y": "ret_1y",
    "correlation": "avg_corr",
}


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=1)
    if s.notna().sum() < 3 or not std or np.isnan(std):
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def component_scores(features: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name, parts in COMPONENTS.items():
        pieces = [zscore(features[col]) * sign for col, sign in parts if col in features.columns]
        out[name] = pd.concat(pieces, axis=1).mean(axis=1) if pieces else pd.Series(np.nan, index=features.index)
    return pd.DataFrame(out)


def normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    active = {k: float(v) for k, v in weights.items() if float(v) > 0}
    total = sum(active.values())
    if total <= 0:
        raise ValueError("todos los pesos del Quant Score son cero")
    unknown = set(active) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"componentes desconocidos en signal_weights: {sorted(unknown)}")
    return {k: v / total for k, v in active.items()}


def quant_score(features: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Componentes, score compuesto y rango (1 = mas favorable a un largo)."""
    comps = component_scores(features)
    w = normalized_weights(weights)
    block = comps[list(w)]
    weight_row = pd.Series(w)
    # Renormaliza sobre los componentes disponibles de cada activo.
    available = block.notna().mul(weight_row, axis=1)
    score = (block.fillna(0.0) * weight_row).sum(axis=1) / available.sum(axis=1).replace(0, np.nan)
    out = comps.copy()
    out["quant_score"] = score
    out["rank"] = score.rank(ascending=False, method="min").astype("Int64")
    return out


def feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    cols = {k: v for k, v in FEATURE_MATRIX_COLUMNS.items() if v in features.columns}
    return pd.DataFrame({k: zscore(features[v]) for k, v in cols.items()})


def rank_stability(features: pd.DataFrame, weights: dict[str, float],
                   draws: int, seed: int) -> dict[str, float]:
    """Cuanto cambia el ranking si los pesos fueran otros razonables.

    Se sortean pesos de una Dirichlet centrada en los del YAML y se mide la
    correlacion de rangos (Spearman) con el ranking base. Si la mediana baja de
    ~0,8, el ranking depende de los pesos elegidos mas que de los datos.
    """
    base = quant_score(features, weights)["quant_score"]
    w = normalized_weights(weights)
    keys = list(w)
    alpha = np.array([w[k] for k in keys]) * 20.0
    rng = np.random.default_rng(seed)
    correlations = []
    for _ in range(draws):
        sample = dict(zip(keys, rng.dirichlet(alpha)))
        alt = quant_score(features, sample)["quant_score"]
        correlations.append(base.rank().corr(alt.rank()))
    arr = np.array(correlations, dtype=float)
    return {"median_spearman": float(np.nanmedian(arr)),
            "p05_spearman": float(np.nanquantile(arr, 0.05)),
            "draws": draws}
