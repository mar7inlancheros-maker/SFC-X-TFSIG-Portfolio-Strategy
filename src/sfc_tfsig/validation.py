"""Validacion estadistica: probar que el modelo NO funciona, y ver si aguanta.

Un backtest con una curva bonita no es evidencia. Es una hipotesis que ha
sobrevivido a un solo experimento, elegido a posteriori entre todos los que se
podrian haber hecho. Este modulo aplica los contrastes que la pueden refutar:

- **Coeficiente de informacion (IC).** Correlacion de rangos entre el score de
  hoy y el retorno del mes que viene, fecha a fecha. Un IC medio de 0.02-0.05
  con t significativo es lo normal en un buen factor de renta variable; si sale
  0.15, casi siempre hay una fuga de datos futuros, no un descubrimiento.

- **Carteras por quintiles.** Si el score ordena de verdad, el retorno medio
  debe crecer monotonamente del quintil 5 al 1. Un spread grande impulsado solo
  por el peor quintil dice que el modelo sabe detectar basura, no encontrar
  ganadores -- distincion importante para una estrategia long-only, que no puede
  vender en corto la basura.

- **Fama-MacBeth con errores Newey-West.** Regresion de seccion cruzada mes a
  mes y despues media temporal de los coeficientes. Newey-West corrige la
  autocorrelacion: los coeficientes mensuales de un factor lento se parecen
  entre meses consecutivos, y tratarlos como independientes infla el t hasta el
  doble. Es la diferencia entre "significativo" y "significativo de verdad".

- **Walk-forward.** La ventana de prueba siempre es POSTERIOR a la de
  calibracion, y avanza. Es el unico modo de medicion que el comite puede
  defender ante alguien que pregunte si los parametros se eligieron sabiendo el
  resultado.

Sin scipy a proposito: el valor p sale de la normal acumulada via `math.erfc`,
con error despreciable para estos tamanos muestrales, y el repo se queda con
pandas y numpy como unicas dependencias de calculo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def two_sided_p_value(t_stat: float) -> float:
    """p de dos colas bajo normalidad asintotica."""
    if t_stat is None or pd.isna(t_stat):
        return float("nan")
    return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def newey_west_se(series: pd.Series, lags: int) -> float:
    """Error estandar de la media con correccion de Newey-West.

    Con `lags=0` es el error estandar clasico. Cada rezago anade la
    autocovarianza ponderada por el nucleo de Bartlett.
    """
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    demeaned = x - x.mean()
    variance = float(demeaned @ demeaned) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cov = float(demeaned[lag:] @ demeaned[:-lag]) / n
        variance += 2.0 * weight * cov
    if variance <= 0:
        return float("nan")
    return math.sqrt(variance / n)


@dataclass(frozen=True)
class MeanTest:
    """Media de una serie temporal con su significancia."""

    mean: float
    std: float
    n: int
    se: float
    t_stat: float
    p_value: float
    annualized_ir: float

    def to_dict(self) -> dict:
        return {
            "mean": self.mean, "std": self.std, "n": self.n, "se": self.se,
            "t_stat": self.t_stat, "p_value": self.p_value,
            "annualized_ir": self.annualized_ir,
        }


def test_mean(series: pd.Series, lags: int = 6, periods_per_year: int = 12) -> MeanTest:
    """Contrasta si la media de una serie mensual es distinta de cero."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    n = len(values)
    if n < 3:
        return MeanTest(float("nan"), float("nan"), n, float("nan"),
                        float("nan"), float("nan"), float("nan"))
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    se = newey_west_se(values, lags)
    t_stat = mean / se if se and not pd.isna(se) and se > 0 else float("nan")
    ir = (mean / std * math.sqrt(periods_per_year)) if std > 0 else float("nan")
    return MeanTest(mean, std, n, se, t_stat, two_sided_p_value(t_stat), ir)


# ---------------------------------------------------------------------------
#  Coeficiente de informacion
# ---------------------------------------------------------------------------


def information_coefficient(
    panel: pd.DataFrame,
    *,
    score_col: str = "score_composite",
    forward_col: str = "forward_return",
    date_col: str = "date",
    min_names: int = 20,
) -> pd.Series:
    """IC de rangos (Spearman) por fecha.

    De rangos y no de Pearson: los retornos tienen colas gruesas y una sola
    accion que se duplica dominaria una correlacion lineal. Lo que se quiere
    saber es si el score ORDENA bien.
    """
    results: dict[pd.Timestamp, float] = {}
    for date, group in panel.groupby(date_col, sort=True):
        pair = group[[score_col, forward_col]].dropna()
        if len(pair) < min_names:
            continue
        ranks = pair.rank()
        correlation = ranks[score_col].corr(ranks[forward_col])
        if pd.notna(correlation):
            results[pd.Timestamp(date)] = float(correlation)
    return pd.Series(results).sort_index()


def ic_summary(ic: pd.Series, lags: int = 6) -> dict[str, float]:
    """Resumen del IC: media, t Newey-West, y fraccion de meses positivos."""
    test = test_mean(ic, lags=lags)
    summary = test.to_dict()
    summary["hit_rate"] = float((ic > 0).mean()) if len(ic) else float("nan")
    return summary


# ---------------------------------------------------------------------------
#  Carteras por quantiles
# ---------------------------------------------------------------------------


def quantile_returns(
    panel: pd.DataFrame,
    *,
    n_quantiles: int = 5,
    score_col: str = "score_composite",
    forward_col: str = "forward_return",
    date_col: str = "date",
    min_names: int = 20,
) -> pd.DataFrame:
    """Retorno medio del mes siguiente por quantil de score, fecha a fecha.

    Quantil 1 = mejores scores. Equiponderado dentro de cada quantil.
    """
    rows: list[dict] = []
    for date, group in panel.groupby(date_col, sort=True):
        valid = group[[score_col, forward_col]].dropna()
        if len(valid) < max(min_names, n_quantiles * 3):
            continue
        labels = list(range(1, n_quantiles + 1))
        buckets = pd.qcut(
            valid[score_col].rank(ascending=False, method="first"),
            q=n_quantiles, labels=labels,
        )
        means = valid.groupby(buckets, observed=True)[forward_col].mean()
        row = {"date": pd.Timestamp(date)}
        row.update({f"Q{int(q)}": float(means.get(q, np.nan)) for q in labels})
        row["spread"] = row.get("Q1", np.nan) - row.get(f"Q{n_quantiles}", np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def monotonicity(quantiles: pd.DataFrame, n_quantiles: int = 5) -> float:
    """Correlacion entre numero de quantil y retorno medio. -1 es perfecto.

    Perfecto es -1 porque Q1 (mejor score) deberia tener el mayor retorno. Un
    valor cercano a 0 con un spread grande delata que el resultado depende de
    uno o dos grupos extremos, no de una ordenacion que se sostenga.
    """
    columns = [f"Q{i}" for i in range(1, n_quantiles + 1) if f"Q{i}" in quantiles.columns]
    if len(columns) < 3:
        return float("nan")
    means = quantiles[columns].mean()
    return float(pd.Series(range(1, len(columns) + 1)).corr(pd.Series(means.to_numpy())))


# ---------------------------------------------------------------------------
#  Fama-MacBeth
# ---------------------------------------------------------------------------


def _ols(y: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    """Minimos cuadrados con constante. None si la matriz es singular."""
    design = np.column_stack([np.ones(len(x)), x])
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return beta


def fama_macbeth(
    panel: pd.DataFrame,
    factors: list[str],
    *,
    forward_col: str = "forward_return",
    date_col: str = "date",
    lags: int = 6,
    standardize: bool = True,
    min_names: int = 20,
) -> pd.DataFrame:
    """Dos etapas: regresion de corte transversal por fecha, media temporal despues.

    Con `standardize`, cada factor se lleva a z-score dentro de la fecha, asi
    que los coeficientes se leen como "retorno mensual por desviacion tipica de
    exposicion" y son comparables entre factores.
    """
    coefficients: list[dict] = []
    for date, group in panel.groupby(date_col, sort=True):
        block = group[[forward_col] + factors].dropna()
        if len(block) < max(min_names, len(factors) * 5):
            continue
        y = block[forward_col].to_numpy(dtype=float)
        x = block[factors].to_numpy(dtype=float)
        if standardize:
            mean = x.mean(axis=0)
            std = x.std(axis=0, ddof=0)
            std = np.where(std > 0, std, np.nan)
            x = (x - mean) / std
            if np.isnan(x).any():
                continue
        beta = _ols(y, x)
        if beta is None:
            continue
        row = {"date": pd.Timestamp(date), "alpha": float(beta[0])}
        row.update({factor: float(b) for factor, b in zip(factors, beta[1:])})
        coefficients.append(row)

    if not coefficients:
        return pd.DataFrame()

    series = pd.DataFrame(coefficients).set_index("date")
    summary = []
    for column in series.columns:
        test = test_mean(series[column], lags=lags)
        summary.append({"term": column, **test.to_dict()})
    return pd.DataFrame(summary).set_index("term")


# ---------------------------------------------------------------------------
#  Walk-forward
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __str__(self) -> str:
        return (f"train {self.train_start.date()}..{self.train_end.date()} | "
                f"test {self.test_start.date()}..{self.test_end.date()}")


def walk_forward_windows(
    dates: pd.DatetimeIndex, train_years: float, test_years: float
) -> list[Window]:
    """Ventanas deslizantes: entrenar atras, probar delante, avanzar y repetir.

    No hay solape entre entrenamiento y prueba en ninguna ventana, y la prueba
    de una ventana puede ser entrenamiento de la siguiente -- que es lo que pasa
    en la realidad segun avanza el tiempo.
    """
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates)))
    if len(dates) == 0:
        return []

    windows: list[Window] = []
    train_start = dates[0]
    while True:
        train_end = train_start + pd.DateOffset(years=int(train_years))
        test_end = train_end + pd.DateOffset(years=int(test_years))
        if train_end > dates[-1]:
            break
        test_slice = dates[(dates > train_end) & (dates <= test_end)]
        if len(test_slice) == 0:
            break
        windows.append(
            Window(
                train_start=train_start,
                train_end=train_end,
                test_start=test_slice[0],
                test_end=test_slice[-1],
            )
        )
        train_start = train_start + pd.DateOffset(years=int(test_years))
    return windows


def out_of_sample_ic(
    panel: pd.DataFrame,
    windows: list[Window],
    *,
    score_col: str = "score_composite",
    forward_col: str = "forward_return",
) -> pd.DataFrame:
    """IC medido SOLO en los tramos de prueba de cada ventana.

    Si el IC dentro de muestra es alto y fuera de muestra es cero, el modelo
    esta memorizando el pasado. Esa comparacion es el contraste mas util de
    todo el modulo.
    """
    rows = []
    for window in windows:
        test_block = panel[
            (panel["date"] >= window.test_start) & (panel["date"] <= window.test_end)
        ]
        train_block = panel[
            (panel["date"] >= window.train_start) & (panel["date"] <= window.train_end)
        ]
        ic_test = information_coefficient(test_block, score_col=score_col, forward_col=forward_col)
        ic_train = information_coefficient(train_block, score_col=score_col, forward_col=forward_col)
        rows.append(
            {
                "window": str(window),
                "train_ic": float(ic_train.mean()) if len(ic_train) else np.nan,
                "test_ic": float(ic_test.mean()) if len(ic_test) else np.nan,
                "test_months": len(ic_test),
            }
        )
    return pd.DataFrame(rows)


def full_report(
    panel: pd.DataFrame,
    cfg,
    *,
    score_col: str = "score_composite",
    forward_col: str = "forward_return",
) -> dict[str, object]:
    """Todos los contrastes de una vez. Lo que consume `report.py`."""
    n_quantiles = int(cfg.get("validation.n_quantiles"))
    lags = int(cfg.get("validation.newey_west_lag"))

    ic = information_coefficient(panel, score_col=score_col, forward_col=forward_col)
    quantiles = quantile_returns(
        panel, n_quantiles=n_quantiles, score_col=score_col, forward_col=forward_col
    )
    factor_columns = [c for c in panel.columns if c.startswith("score_") and c != score_col]
    windows = walk_forward_windows(
        pd.DatetimeIndex(sorted(panel["date"].unique())),
        float(cfg.get("validation.train_years")),
        float(cfg.get("validation.test_years")),
    )

    return {
        "ic": ic,
        "ic_summary": ic_summary(ic, lags=lags),
        "quantiles": quantiles,
        "quantile_means": quantiles.mean() if not quantiles.empty else pd.Series(dtype=float),
        "spread_test": test_mean(quantiles["spread"], lags=lags).to_dict()
        if "spread" in quantiles else {},
        "monotonicity": monotonicity(quantiles, n_quantiles) if not quantiles.empty else float("nan"),
        "fama_macbeth": fama_macbeth(panel, factor_columns, forward_col=forward_col, lags=lags)
        if factor_columns else pd.DataFrame(),
        "walk_forward": out_of_sample_ic(panel, windows, score_col=score_col, forward_col=forward_col),
    }
