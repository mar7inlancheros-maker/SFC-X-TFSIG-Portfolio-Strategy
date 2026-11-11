"""Historia del riesgo: como ha cambiado, y si el modelo de riesgo acierta.

Dos preguntas que el reporte de un solo dia no contesta:

1. **Riesgo movil.** Volatilidad, beta y tracking error en ventanas de 63
   sesiones. Un 25% de volatilidad hoy es alarmante si la media es 15% y
   normal si es 24%.

2. **Calibracion del riesgo ex-ante.** En cada rebalanceo del backtest se
   calcula la volatilidad que el modelo de riesgo habria PREVISTO para la
   cartera recien comprada (Ledoit-Wolf sobre el ano anterior, sin datos
   posteriores) y se compara con el retorno que luego tuvo. Si el modelo esta
   bien calibrado, retorno / prevista tiene desviacion tipica 1: es el
   estadistico de sesgo de Barra. Por encima, el modelo SUBESTIMA el riesgo y
   todos los VaR del reporte son optimistas; por debajo, lo exagera.

Funciones puras.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..metrics import TRADING_DAYS
from .exposure import ledoit_wolf


def rolling_risk(nav: pd.Series, benchmark_nav: pd.Series | None, window: int = 63) -> pd.DataFrame:
    """Volatilidad, beta y tracking error moviles, anualizados."""
    r = nav.dropna().astype(float).pct_change()
    out = pd.DataFrame({"vol": r.rolling(window).std() * math.sqrt(TRADING_DAYS)})
    if benchmark_nav is not None:
        b = benchmark_nav.reindex(nav.index).ffill().pct_change()
        cov = r.rolling(window).cov(b)
        out["beta"] = cov / b.rolling(window).var()
        out["tracking_error"] = (r - b).rolling(window).std() * math.sqrt(TRADING_DAYS)
        out["benchmark_vol"] = b.rolling(window).std() * math.sqrt(TRADING_DAYS)
    return out.dropna(how="all")


def rolling_summary(rolling: pd.DataFrame) -> pd.DataFrame:
    """Hoy frente a la historia: valor actual, mediana, percentil del actual."""
    rows = {}
    for column in rolling.columns:
        series = rolling[column].dropna()
        if series.empty:
            continue
        current = float(series.iloc[-1])
        rows[column] = {
            "current": current,
            "median": float(series.median()),
            "p10": float(series.quantile(0.10)),
            "p90": float(series.quantile(0.90)),
            "percentile": float((series <= current).mean()),
        }
    return pd.DataFrame(rows).T


def risk_model_calibration(
    holdings: pd.DataFrame,
    close_wide: pd.DataFrame,
    nav: pd.Series,
    *,
    window: int = 252,
    min_names: int = 5,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Volatilidad prevista frente a realizada en cada periodo entre rebalanceos.

    Para cada fecha de rebalanceo `e`: covarianza Ledoit-Wolf de los nombres
    comprados con los `window` dias HASTA `e`, pesos realizados de ese dia,
    volatilidad prevista del periodo = sigma diaria x raiz de sus sesiones. El
    realizado es el retorno del NAV hasta el siguiente rebalanceo.

    El resultado incluye costes y caja, que la prevision no ve: un sesgo
    pequeno hacia arriba es esperable y no es un fallo del modelo.
    """
    if holdings.empty:
        return pd.DataFrame(), {}
    holdings = holdings.copy()
    holdings["date"] = pd.to_datetime(holdings["date"])
    dates = sorted(holdings["date"].unique())
    nav = nav.dropna().astype(float)
    rows = []
    for i, date in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else nav.index[-1]
        block = holdings[holdings["date"] == date].set_index("ticker")
        weights = block["weight"].astype(float)
        visible = close_wide.loc[close_wide.index <= date].reindex(columns=weights.index)
        returns = visible.tail(window + 1).pct_change(fill_method=None).iloc[1:]
        # Solo nombres con historia suficiente: la prevision de una OPI de dos
        # meses es ruido y contaminaria el estadistico.
        usable = returns.columns[returns.notna().mean() >= 0.8]
        if len(usable) < min_names or len(returns) < 60:
            continue
        cov, _ = ledoit_wolf(returns[usable].fillna(0.0))
        w = weights.reindex(usable).to_numpy()
        # Los nombres sin historia se suponen como los demas: la prevision se
        # escala al peso total. Dejarlos fuera la infravaloraria tanto como su peso.
        coverage = float(w.sum() / weights.sum()) if weights.sum() else 0.0
        var_daily = float(w @ cov.to_numpy() @ w) / max(coverage, 1e-9) ** 2
        segment = nav.loc[(nav.index >= date) & (nav.index <= end)]
        sessions = len(segment) - 1
        if sessions < 5:
            continue
        predicted = math.sqrt(max(var_daily, 0.0) * sessions)
        realized = float(segment.iloc[-1] / segment.iloc[0] - 1.0)
        daily = segment.pct_change().dropna()
        rows.append({
            "date": pd.Timestamp(date),
            "sessions": sessions,
            "predicted_vol_annual": math.sqrt(max(var_daily, 0.0) * TRADING_DAYS),
            "realized_vol_annual": float(daily.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(daily) > 2 else np.nan,
            "predicted_period_vol": predicted,
            "realized_return": realized,
            "z": realized / predicted if predicted > 0 else np.nan,
            "weight_modelled": coverage,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table, {}
    z = table["z"].dropna()
    # Se desmedia z: el retorno medio positivo de la estrategia no es error de
    # prevision de volatilidad.
    bias = float(z.std(ddof=1))
    half_width = 1.96 * math.sqrt(1.0 / (2.0 * len(z)))
    summary = {
        "n_periods": int(len(z)),
        "bias_statistic": bias,
        "ci_low": 1.0 - half_width,
        "ci_high": 1.0 + half_width,
        "mean_predicted_vol": float(table["predicted_vol_annual"].mean()),
        "mean_realized_vol": float(table["realized_vol_annual"].mean()),
        "share_beyond_2sd": float((z.sub(z.mean()).abs() > 2.0).mean()),
        "verdict": ("understates risk" if bias > 1.0 + half_width
                    else "overstates risk" if bias < 1.0 - half_width
                    else "well calibrated"),
    }
    return table, summary
