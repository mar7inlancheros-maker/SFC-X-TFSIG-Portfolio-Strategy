"""Estandarizacion de metricas en seccion cruzada. Funciones puras.

El problema: un ROIC del 18% y un rendimiento por flujo de caja del 6% no se
pueden sumar. Estan en unidades distintas, con dispersiones distintas y con
colas distintas. Hay que llevarlos a una escala comun antes de combinarlos, y
como se haga esa conversion cambia la cartera resultante.

Tres decisiones, las tres discutibles y las tres explicitas en el TOML:

1. **Winsorizar antes de estandarizar.** Una sola empresa con un rendimiento del
   400% (denominador diminuto tras una ampliacion) mueve la media y la
   desviacion de toda la seccion cruzada, y hunde el z-score de las 499
   restantes. Se recorta el 2% de cada cola. No se elimina el dato: se acota,
   porque la empresa sigue siendo barata, solo que no 400% barata.

2. **Estandarizar DENTRO del sector.** Los bancos operan con apalancamientos que
   en una industrial serian de quiebra; las tecnologicas cotizan a multiplos que
   en un servicio publico serian delirantes. Un z-score global no mide calidad
   ni valor: mide sector. Con `sector_neutral = true` el modelo compara cada
   empresa con sus pares, y el resultado es una seleccion DENTRO de cada sector
   en vez de una apuesta sectorial encubierta.

   Contrapartida honesta: si un sector entero esta barato, la neutralizacion
   impide aprovecharlo. Es el precio de no tener la cartera entera en energia
   cada vez que el petroleo cae.

3. **Cobertura minima por metrica.** Una metrica que solo existe para el 30% del
   universo no se promedia con las demas: quien la tiene compite contra quien no
   la tiene, y el score deja de significar lo mismo para cada empresa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(series: pd.Series, pct: float) -> pd.Series:
    """Acota la serie a sus percentiles [pct, 1-pct]."""
    if pct <= 0 or series.notna().sum() < 3:
        return series
    lower = series.quantile(pct)
    upper = series.quantile(1.0 - pct)
    if pd.isna(lower) or pd.isna(upper):
        return series
    return series.clip(lower=lower, upper=upper)


def zscore(series: pd.Series) -> pd.Series:
    """z-score. Si no hay dispersion, devuelve 0: todos empatan, nadie destaca."""
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    mean = valid.mean()
    std = valid.std(ddof=0)
    if std == 0 or pd.isna(std):
        return series.notna().astype(float) * 0.0
    return (series - mean) / std


def rank_score(series: pd.Series) -> pd.Series:
    """Rango normalizado a [-1, 1]. Alternativa robusta al z-score.

    No se usa por defecto, pero esta aqui porque es el contraste natural: si el
    modelo funciona con z-score y deja de funcionar con rangos, lo que funciona
    son los valores extremos, no la ordenacion -- y eso es mucho mas fragil de
    lo que un backtest con z-scores deja ver.
    """
    ranks = series.rank(pct=True)
    return (ranks - 0.5) * 2.0


def standardize_group(
    frame: pd.DataFrame,
    metrics: dict[str, int],
    *,
    winsorize_pct: float,
    min_coverage: float,
    method: str = "zscore",
) -> pd.Series:
    """Promedia las metricas estandarizadas de un grupo (una fecha, un sector).

    `metrics` mapea nombre -> direccion (+1 mas es mejor, -1 menos es mejor).
    Devuelve una serie con el score medio, o NaN donde no haya ninguna metrica.
    """
    standardizer = zscore if method == "zscore" else rank_score
    n = len(frame)
    pieces: list[pd.Series] = []

    for metric, direction in metrics.items():
        if metric not in frame.columns:
            continue
        raw = pd.to_numeric(frame[metric], errors="coerce")
        if n == 0 or raw.notna().sum() / n < min_coverage:
            # Cobertura insuficiente: la metrica se descarta ENTERA para este
            # grupo. Rellenar los huecos con la media seria inventarse el dato
            # justo para las empresas de las que menos se sabe.
            continue
        clipped = winsorize(raw, winsorize_pct)
        pieces.append(standardizer(clipped) * direction)

    if not pieces:
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    stacked = pd.concat(pieces, axis=1)
    # Media ignorando NaN: una empresa a la que le falta una metrica de cinco se
    # puntua con las otras cuatro en vez de quedar fuera del factor.
    return stacked.mean(axis=1, skipna=True)


def factor_score(
    panel: pd.DataFrame,
    metrics: dict[str, int],
    *,
    winsorize_pct: float,
    min_coverage: float,
    sector_neutral: bool,
    min_sector_names: int,
    method: str = "zscore",
    date_col: str = "date",
    sector_col: str = "sector",
) -> pd.Series:
    """Score de un factor para todo el panel, fecha a fecha.

    Con `sector_neutral`, los sectores con menos de `min_sector_names` nombres
    caen al grupo global de esa fecha: estandarizar dentro de un sector de tres
    empresas produce z-scores de +-1.2 mecanicos que no dicen nada.
    """
    out = pd.Series(np.nan, index=panel.index, dtype="float64")

    for _, day in panel.groupby(date_col, sort=False):
        if sector_neutral and sector_col in day.columns:
            counts = day[sector_col].value_counts()
            big = counts[counts >= min_sector_names].index
            in_big = day[sector_col].isin(big)

            for sector in big:
                mask = day[sector_col] == sector
                block = day[mask]
                out.loc[block.index] = standardize_group(
                    block, metrics, winsorize_pct=winsorize_pct,
                    min_coverage=min_coverage, method=method,
                )

            rest = day[~in_big]
            if len(rest) > 0:
                out.loc[rest.index] = standardize_group(
                    rest, metrics, winsorize_pct=winsorize_pct,
                    min_coverage=min_coverage, method=method,
                )
        else:
            out.loc[day.index] = standardize_group(
                day, metrics, winsorize_pct=winsorize_pct,
                min_coverage=min_coverage, method=method,
            )

    return out


def combine_factors(
    scores: pd.DataFrame,
    weights: dict[str, float],
    *,
    min_factors: int = 2,
) -> pd.Series:
    """Score compuesto: media ponderada de los factores disponibles.

    Los pesos se renormalizan sobre los factores que esa empresa SI tiene. Una
    empresa con valor y calidad pero sin momentum (recien salida a bolsa) se
    puntua con los dos que tiene, no se le asigna un cero en el tercero -- un
    cero seria "momentum promedio", que es una afirmacion que nadie ha
    comprobado.

    Por debajo de `min_factors` disponibles, el score es NaN: no hay base
    suficiente para poner capital detras.
    """
    available = [f for f in weights if f in scores.columns]
    if not available:
        return pd.Series(np.nan, index=scores.index, dtype="float64")

    block = scores[available]
    weight_row = pd.Series({f: weights[f] for f in available}, dtype="float64")

    mask = block.notna()
    effective_weights = mask.mul(weight_row, axis=1)
    total = effective_weights.sum(axis=1)

    weighted = (block.fillna(0.0) * effective_weights).sum(axis=1)
    composite = weighted.div(total.where(total > 0))

    return composite.where(mask.sum(axis=1) >= min_factors)
