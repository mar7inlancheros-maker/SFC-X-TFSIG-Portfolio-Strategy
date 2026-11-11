"""Modelo factorial de riesgo: cuanto del riesgo es sector, cuanto estilo y cuanto propio.

La covarianza de Ledoit-Wolf dice CUANTO riesgo tiene la cartera. No dice DE
DONDE viene. Un 25% de volatilidad puede ser una apuesta al mercado, una
apuesta a momentum disfrazada de treinta nombres, o treinta riesgos propios que
se diversifican. Cada una exige una conversacion distinta en el comite.

**Construccion, al estilo Barra y en pequeno.** Cada dia, regresion transversal
de los retornos de todo el universo sobre:

- una variable ficticia por sector (sin constante: el factor de un sector
  incluye el mercado, y el "riesgo sectorial" que se reporta incluye el de
  mercado);
- los scores de estilo del propio modelo (valor, calidad, momentum, baja
  volatilidad), que ya son z-scores sector-neutrales;
- tamano: log de la capitalizacion, estandarizado ese dia.

Minimos cuadrados ponderados por la raiz de la capitalizacion: las grandes
tienen residuos menos ruidosos y el mercado las pesa mas. Los coeficientes de
cada dia son los retornos de los factores; lo que queda, el retorno especifico
de cada nombre.

**Sin mirar al futuro.** El retorno del dia `d` se explica con las exposiciones
del ultimo panel ANTERIOR a `d`: las que se conocian al abrir.

**Riesgo de la cartera.** var = x' F x + sum w_i^2 s_i^2, con x las exposiciones
de la cartera, F la covarianza de los factores y s_i la volatilidad especifica.
Descomposicion de Euler sobre los factores: suma exacta.

Funcion pura: entran precios y panel, sale un `FactorRisk`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..metrics import TRADING_DAYS

STYLE_COLUMNS = ("score_value", "score_quality", "score_momentum", "score_lowvol")
_Z_CLIP = 3.0
_RETURN_CLIP = 0.5
_MIN_NAMES = 50
_MIN_SPECIFIC_OBS = 60


@dataclass
class FactorRisk:
    total_vol: float
    factor_vol: float
    specific_vol: float
    decomposition: pd.DataFrame      # grupo -> fraccion de varianza
    factor_contributions: pd.DataFrame
    exposures: pd.Series             # exposicion de la cartera a cada factor
    factor_vols: pd.Series           # volatilidad anual de cada factor
    mean_r2: float
    n_days: int
    factor_cov: pd.DataFrame
    specific_var: pd.Series
    warnings: list[str] = field(default_factory=list)


def _exposures(cross: pd.DataFrame, sectors: list[str]) -> pd.DataFrame:
    """Matriz de exposiciones de una seccion cruzada del panel."""
    frame = cross.drop_duplicates("ticker").set_index("ticker")
    out = pd.DataFrame(index=frame.index)
    sector = frame["sector"].fillna("Unknown") if "sector" in frame else pd.Series("Unknown", index=frame.index)
    for s in sectors:
        out[f"sector_{s}"] = (sector == s).astype(float)
    for column in STYLE_COLUMNS:
        if column in frame:
            out[column.removeprefix("score_")] = (
                pd.to_numeric(frame[column], errors="coerce").clip(-_Z_CLIP, _Z_CLIP).fillna(0.0)
            )
    if "market_cap" in frame:
        log_cap = np.log(pd.to_numeric(frame["market_cap"], errors="coerce").where(lambda s: s > 0))
        sd = log_cap.std()
        size = (log_cap - log_cap.mean()) / sd if sd and np.isfinite(sd) else log_cap * 0.0
        out["size"] = size.clip(-_Z_CLIP, _Z_CLIP).fillna(0.0)
        cap = pd.to_numeric(frame["market_cap"], errors="coerce")
        out["_weight"] = np.sqrt(cap.where(cap > 0).fillna(cap.median()))
    else:
        out["_weight"] = 1.0
    return out


def estimate_factor_model(
    close_wide: pd.DataFrame,
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    window: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[pd.Timestamp, pd.DataFrame]]:
    """Retornos de los factores, residuos, R2 diario y exposiciones por fecha de panel."""
    as_of = pd.Timestamp(as_of)
    panel = panel[pd.to_datetime(panel["date"]) <= as_of].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel_dates = np.array(sorted(panel["date"].unique()), dtype="datetime64[ns]")
    if len(panel_dates) == 0:
        raise ValueError("the panel has no dates before the analysis date")
    sectors = sorted(panel["sector"].dropna().unique()) if "sector" in panel else ["Unknown"]

    prices = close_wide.loc[close_wide.index <= as_of].tail(window + 1)
    returns = prices.pct_change(fill_method=None).iloc[1:].clip(-_RETURN_CLIP, _RETURN_CLIP)

    exposures_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    factor_rows: dict[pd.Timestamp, pd.Series] = {}
    residual_rows: dict[pd.Timestamp, pd.Series] = {}
    r2: dict[pd.Timestamp, float] = {}

    for day, y_all in returns.iterrows():
        position = np.searchsorted(panel_dates, np.datetime64(day), side="left") - 1
        if position < 0:
            continue
        exp_date = pd.Timestamp(panel_dates[position])
        if exp_date not in exposures_by_date:
            exposures_by_date[exp_date] = _exposures(panel[panel["date"] == exp_date], sectors)
        x_all = exposures_by_date[exp_date]
        y = y_all.reindex(x_all.index).dropna()
        if len(y) < _MIN_NAMES:
            continue
        x = x_all.loc[y.index]
        weight = x.pop("_weight").to_numpy()
        # Un sector sin miembros ese dia no se puede estimar: fuera de la
        # regresion en vez de dejar la matriz singular.
        x = x.loc[:, (x != 0).any(axis=0)]
        root = np.sqrt(weight)[:, None]
        coef, *_ = np.linalg.lstsq(x.to_numpy() * root, y.to_numpy() * root[:, 0], rcond=None)
        fitted = x.to_numpy() @ coef
        resid = y.to_numpy() - fitted
        total = np.sum(weight * (y.to_numpy() - np.average(y.to_numpy(), weights=weight)) ** 2)
        r2[day] = float(1.0 - np.sum(weight * resid ** 2) / total) if total > 0 else float("nan")
        factor_rows[day] = pd.Series(coef, index=x.columns)
        residual_rows[day] = pd.Series(resid, index=y.index)

    if not factor_rows:
        raise ValueError(
            "no day with enough names for the cross-sectional regression: "
            "check that prices and the panel cover the model window"
        )
    factor_returns = pd.DataFrame(factor_rows).T.sort_index().fillna(0.0)
    residuals = pd.DataFrame(residual_rows).T.sort_index()
    return factor_returns, residuals, pd.Series(r2), exposures_by_date


def factor_risk(
    weights: pd.Series,
    sectors: pd.Series,
    close_wide: pd.DataFrame,
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    window: int = 252,
) -> FactorRisk:
    """Descomposicion del riesgo de la cartera en sectores, estilos y especifico."""
    factor_returns, residuals, r2, _ = estimate_factor_model(close_wide, panel, as_of, window)
    warnings: list[str] = []

    last = pd.to_datetime(panel["date"])
    latest = last[last <= pd.Timestamp(as_of)].max()
    all_sectors = sorted(panel["sector"].dropna().unique()) if "sector" in panel else ["Unknown"]
    exposures_now = _exposures(panel[pd.to_datetime(panel["date"]) == latest], all_sectors)
    exposures_now = exposures_now.drop(columns="_weight").reindex(columns=factor_returns.columns, fill_value=0.0)

    names = list(weights.index)
    x_book = exposures_now.reindex(names)
    missing = x_book.index[x_book.isna().all(axis=1)]
    if len(missing):
        # Sin fila en el panel: se le da su sector y estilo cero (la media).
        for ticker in missing:
            column = f"sector_{sectors.get(ticker, 'Unknown')}"
            x_book.loc[ticker] = 0.0
            if column in x_book.columns:
                x_book.loc[ticker, column] = 1.0
        warnings.append(
            f"no exposures in the panel for {', '.join(map(str, missing))}: "
            "they are modeled with their sector and a neutral style"
        )
    x_book = x_book.fillna(0.0)

    f_cov = factor_returns.cov()
    w = weights.astype(float)
    x = x_book.T @ w                                    # exposicion de la cartera por factor
    fx = f_cov.to_numpy() @ x.to_numpy()
    factor_var = float(x.to_numpy() @ fx)

    counts = residuals.notna().sum()
    spec = residuals.var(ddof=1).where(counts >= _MIN_SPECIFIC_OBS)
    # Nombres con poca historia: la varianza especifica mediana de su sector.
    by_sector = spec.groupby(panel.drop_duplicates("ticker").set_index("ticker")["sector"]
                             .reindex(spec.index).fillna("Unknown")).median()
    spec_book = spec.reindex(names)
    fill = sectors.reindex(names).map(by_sector).fillna(spec.median())
    short = spec_book.isna()
    if short.any():
        warnings.append(
            f"specific variance from the sector median for {', '.join(map(str, spec_book.index[short]))} "
            f"(fewer than {_MIN_SPECIFIC_OBS} days of residuals)"
        )
    spec_book = spec_book.fillna(fill)
    specific_var = float(np.sum(w.to_numpy() ** 2 * spec_book.to_numpy()))
    total_var = factor_var + specific_var

    contrib = pd.DataFrame({
        "exposure": x,
        "factor_vol": np.sqrt(np.diag(f_cov.to_numpy()) * TRADING_DAYS),
        "variance_share": x.to_numpy() * fx / total_var if total_var > 0 else np.nan,
    }, index=x.index)
    is_sector = contrib.index.str.startswith("sector_")
    groups = {"sectors (incl. market)": float(contrib.loc[is_sector, "variance_share"].sum())}
    for factor in contrib.index[~is_sector]:
        groups[factor] = float(contrib.loc[factor, "variance_share"])
    groups["specific"] = specific_var / total_var if total_var > 0 else float("nan")
    decomposition = pd.DataFrame({"variance_share": pd.Series(groups)})
    decomposition["vol_contribution"] = decomposition["variance_share"] * math.sqrt(total_var * TRADING_DAYS)

    return FactorRisk(
        total_vol=math.sqrt(max(total_var, 0.0) * TRADING_DAYS),
        factor_vol=math.sqrt(max(factor_var, 0.0) * TRADING_DAYS),
        specific_vol=math.sqrt(max(specific_var, 0.0) * TRADING_DAYS),
        decomposition=decomposition,
        factor_contributions=contrib,
        exposures=x,
        factor_vols=contrib["factor_vol"],
        mean_r2=float(r2.mean()),
        n_days=int(len(factor_returns)),
        factor_cov=f_cov,
        specific_var=spec_book,
        warnings=warnings,
    )
