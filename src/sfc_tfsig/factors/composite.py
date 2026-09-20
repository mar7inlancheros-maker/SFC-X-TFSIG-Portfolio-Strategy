"""Los cuatro factores del modelo y su combinacion en un solo score.

- **Valor.** Rendimientos, no multiplos. Compra flujo por dolar de precio.
- **Calidad.** Quality Minus Junk (Asness, Frazzini, Pedersen 2019): rentabilidad
  del capital, solidez del balance y prudencia contable. Existe para no comprar
  lo barato que esta barato por buenas razones.
- **Momentum.** 12-1 (Jegadeesh y Titman 1993). Es el unico de los cuatro que no
  mira al negocio, solo al precio, y el que mas turnover genera.
- **Baja volatilidad.** La anomalia de la baja volatilidad (Frazzini-Pedersen,
  Betting Against Beta). Entra con peso pequeno: su funcion aqui es amortiguar
  el drawdown, no generar alfa.

**Por que combinarlos ponderados y no en cascada.** Un embudo (primero filtro por
calidad, luego ordeno por valor) descarta de forma irreversible a una empresa
excelente en tres factores que falla el corte del cuarto por poco. La suma
ponderada -- como en Two Sigma, señales combinadas y no secuenciales -- deja que
las fortalezas compensen debilidades y que la cartera exprese lo que realmente
piensa el modelo. El coste es que un mal score en un factor se puede esconder;
por eso el reporte muestra SIEMPRE los cuatro por separado para cada posicion.
"""

from __future__ import annotations

import pandas as pd

from ..config import Config
from ..financials import METRIC_DIRECTION
from .scoring import combine_factors, factor_score

FACTOR_METRICS: dict[str, tuple[str, ...]] = {
    "momentum": ("momentum",),
    "lowvol": ("volatility",),
}


def _metrics_for(cfg: Config, factor: str) -> dict[str, int]:
    """Metricas de un factor con su direccion (+1 / -1)."""
    if factor in FACTOR_METRICS:
        names = FACTOR_METRICS[factor]
    else:
        names = tuple(cfg.get(f"factors.{factor}.metrics"))

    missing = [n for n in names if n not in METRIC_DIRECTION]
    if missing:
        raise KeyError(
            f"metricas sin direccion declarada en METRIC_DIRECTION: {missing}. "
            "Sin direccion no se sabe si mas es mejor o peor, y el modelo "
            "compraria exactamente lo contrario de lo que se pretende."
        )
    return {name: METRIC_DIRECTION[name] for name in names}


def build_scores(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Panel con metricas -> panel con los scores de cada factor y el compuesto.

    No ordena, no filtra y no decide nada: solo puntua. Quien compra es
    `portfolio.py`.
    """
    proc = cfg.section("factors")["processing"]
    weights = cfg.factor_weights

    scores = pd.DataFrame(index=panel.index)
    for factor in weights:
        scores[factor] = factor_score(
            panel,
            _metrics_for(cfg, factor),
            winsorize_pct=float(proc["winsorize_pct"]),
            min_coverage=float(proc["min_coverage"]),
            sector_neutral=bool(proc["sector_neutral"]),
            min_sector_names=int(proc["min_sector_names"]),
        )

    scores["composite"] = combine_factors(scores, weights)

    out = panel.copy()
    for column in scores.columns:
        out[f"score_{column}"] = scores[column]
    return out


def rank_within_date(panel: pd.DataFrame, score_col: str = "score_composite") -> pd.Series:
    """Rango 1 = mejor, dentro de cada fecha."""
    return panel.groupby("date")[score_col].rank(ascending=False, method="first")
