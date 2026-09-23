"""Decaimiento de la senal: la medida que decide la frecuencia de rebalanceo.

Dos mundos sinteticos con la misma fuerza de senal y vida util opuesta:

- PERSISTENTE: el score de cada empresa es estable en el tiempo y predice su
  retorno todos los meses. Rebalancear menos no pierde nada.
- EFIMERA: el score cambia cada mes y solo predice el mes siguiente. Un mes de
  retraso basta para que no quede nada.

Si la medida no distingue estos dos mundos, no sirve para decidir nada en el
real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig import validation as val

N_NAMES = 60
N_MONTHS = 72


def _mundo(persistente: bool, seed: int = 5):
    """Panel mensual + cierres diarios donde el score predice el mes siguiente."""
    rng = np.random.default_rng(seed)
    month_ends = pd.date_range("2018-01-31", periods=N_MONTHS, freq="ME")
    tickers = [f"T{i:02d}" for i in range(N_NAMES)]

    fijo = rng.normal(size=N_NAMES)
    scores = np.zeros((N_MONTHS, N_NAMES))
    for t in range(N_MONTHS):
        scores[t] = fijo if persistente else rng.normal(size=N_NAMES)

    # Retorno del mes t+1 = senal del mes t + ruido. Ruido grande, como en real.
    prices = np.zeros((N_MONTHS, N_NAMES))
    prices[0] = 100.0
    for t in range(1, N_MONTHS):
        ret = 0.01 * scores[t - 1] + rng.normal(0.0, 0.04, N_NAMES)
        prices[t] = prices[t - 1] * (1.0 + ret)

    # Cierres diarios planos dentro de cada mes: el cierre de fin de mes es el
    # que manda, que es lo unico que mira `forward_returns`.
    sessions = pd.bdate_range(month_ends[0] - pd.Timedelta(days=27), month_ends[-1])
    close = pd.DataFrame(index=sessions, columns=tickers, dtype=float)
    for t, end in enumerate(month_ends):
        start = month_ends[t - 1] if t else sessions[0] - pd.Timedelta(days=1)
        mask = (sessions > start) & (sessions <= end)
        close.loc[mask] = prices[t]

    panel = pd.DataFrame(
        [
            {"date": month_ends[t], "ticker": tickers[i], "score_composite": scores[t, i]}
            for t in range(N_MONTHS)
            for i in range(N_NAMES)
        ]
    )
    return panel, close


def test_la_senal_persistente_aguanta_el_retraso():
    panel, close = _mundo(persistente=True)
    decay = val.ic_by_lag(panel, close, lags_m=(0, 1, 2, 3))
    assert decay.loc[0, "ic_mean"] > 0.1
    # Tres meses de retraso conservan casi toda la senal.
    assert decay.loc[3, "vs_fresh"] > 0.8


def test_la_senal_efimera_muere_con_un_mes_de_retraso():
    panel, close = _mundo(persistente=False)
    decay = val.ic_by_lag(panel, close, lags_m=(0, 1, 2, 3))
    assert decay.loc[0, "ic_mean"] > 0.1
    # Un mes despues ya no queda nada distinguible de cero.
    assert abs(decay.loc[1, "ic_mean"]) < 0.05
    assert decay.loc[1, "p_value"] > 0.05


def test_el_coste_de_rebalancear_trimestral_distingue_los_dos_mundos():
    panel_p, close_p = _mundo(persistente=True)
    panel_e, close_e = _mundo(persistente=False)
    trimestral_p = val.rebalance_signal_cost(val.ic_by_lag(panel_p, close_p), 3)
    trimestral_e = val.rebalance_signal_cost(val.ic_by_lag(panel_e, close_e), 3)

    # Persistente: el trimestral conserva practicamente toda la senal.
    assert trimestral_p > 0.85
    # Efimera: conserva un tercio -- solo el mes en que la senal esta fresca.
    assert trimestral_e == pytest.approx(1 / 3, abs=0.15)


def test_rebalanceo_mensual_conserva_toda_la_senal_por_definicion():
    panel, close = _mundo(persistente=False)
    decay = val.ic_by_lag(panel, close, lags_m=(0, 1, 2))
    assert val.rebalance_signal_cost(decay, 1) == pytest.approx(1.0)


def test_ic_por_horizonte_usa_rezagos_suficientes_para_el_solape():
    """Con horizonte 6, los retornos se solapan cinco meses entre fechas."""
    panel, close = _mundo(persistente=True)
    tabla = val.ic_by_horizon(panel, close, horizons=(1, 6))
    assert set(tabla.index) == {1, 6}
    # Senal persistente: el IC acumulado a 6 meses no se desploma.
    assert tabla.loc[6, "ic_mean"] > 0.5 * tabla.loc[1, "ic_mean"]
    assert tabla.loc[6, "months"] < tabla.loc[1, "months"]


def test_sin_los_retrasos_necesarios_el_coste_no_se_inventa():
    decay = pd.DataFrame({"ic_mean": [0.03, 0.02]}, index=pd.Index([0, 1], name="lag_m"))
    # Trimestral necesita los retrasos 0, 1 y 2; falta el 2.
    assert np.isnan(val.rebalance_signal_cost(decay, 3))
