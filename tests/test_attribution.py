"""Atribucion y las dos reglas que salieron de ella.

La atribucion existe para distinguir un proceso de una posicion afortunada. Los
tests fijan esa distincion con carteras inventadas donde la respuesta se sabe de
antemano.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig import attribution as attr
from sfc_tfsig.factors.composite import build_scores


def _caso(retornos: dict[str, float], nav: float = 100_000.0, pesos: dict | None = None):
    """Una sola fecha de rebalanceo con los retornos dados por nombre."""
    fecha, fin = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")
    tickers = list(retornos)
    pesos = pesos or {t: 1.0 / len(tickers) for t in tickers}

    holdings = pd.DataFrame({
        "date": [fecha] * len(tickers),
        "ticker": tickers,
        "price": [100.0] * len(tickers),
        "shares": [pesos[t] * nav / 100.0 for t in tickers],
        "weight": [pesos[t] for t in tickers],
        "sector": ["Tech"] * len(tickers),
    })
    close = pd.DataFrame(
        {t: [100.0, 100.0 * (1 + retornos[t])] for t in tickers},
        index=[fecha, fin],
    )
    rebalances = pd.DataFrame({"execution_date": [fecha], "nav": [nav]})
    return holdings, close, rebalances


def test_las_contribuciones_suman_el_retorno_bruto_de_la_cartera():
    holdings, close, reb = _caso({"AAA": 0.20, "BBB": -0.10, "CCC": 0.00})
    c = attr.position_contributions(holdings, close, reb)
    # Equiponderada: (0.20 - 0.10 + 0.00) / 3 = +3.33%
    assert c["contribution"].sum() == pytest.approx(0.0333, abs=1e-4)


def test_el_peso_manda_sobre_el_retorno():
    """Un +10% con peso grande aporta mas que un +50% con peso diminuto."""
    holdings, close, reb = _caso(
        {"GRANDE": 0.10, "PEQUENA": 0.50},
        pesos={"GRANDE": 0.9, "PEQUENA": 0.1},
    )
    c = attr.position_contributions(holdings, close, reb).set_index("ticker")
    assert c.loc["GRANDE", "contribution"] > c.loc["PEQUENA", "contribution"]


def test_sin_precio_final_la_posicion_aporta_cero_y_no_desaparece():
    fecha, fin = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")
    holdings = pd.DataFrame({
        "date": [fecha, fecha], "ticker": ["AAA", "SINPRECIO"],
        "price": [100.0, 100.0], "shares": [500.0, 500.0],
        "weight": [0.5, 0.5], "sector": ["Tech", "Tech"],
    })
    close = pd.DataFrame({"AAA": [100.0, 110.0]}, index=[fecha, fin])
    reb = pd.DataFrame({"execution_date": [fecha], "nav": [100_000.0]})

    c = attr.position_contributions(holdings, close, reb).set_index("ticker")
    assert len(c) == 2
    assert c.loc["SINPRECIO", "contribution"] == pytest.approx(0.0)


def test_la_concentracion_distingue_proceso_de_suerte():
    """Un ano hecho por un nombre y otro hecho por veinte, con la misma suma."""
    suerte = {f"T{i:02d}": (2.0 if i == 0 else 0.0) for i in range(20)}
    proceso = {f"T{i:02d}": 0.10 for i in range(20)}

    c_suerte = attr.position_contributions(*_caso(suerte))
    c_proceso = attr.position_contributions(*_caso(proceso))

    res_suerte = attr.concentration_summary(c_suerte)
    res_proceso = attr.concentration_summary(c_proceso)

    assert res_suerte["total_gross_contribution"] == pytest.approx(
        res_proceso["total_gross_contribution"]
    )
    assert res_suerte["top5_share"] == pytest.approx(1.0)
    assert res_proceso["top5_share"] == pytest.approx(0.25)
    assert res_suerte["names_for_half"] == 1
    assert res_proceso["names_for_half"] == 10


def test_quitar_los_cinco_mejores_revela_si_queda_algo():
    ganadores = {f"T{i:02d}": (1.0 if i < 5 else 0.001) for i in range(20)}
    c = attr.position_contributions(*_caso(ganadores))
    res = attr.concentration_summary(c)
    assert res["total_without_top5"] < 0.05 * res["total_gross_contribution"]


# ---------------------------------------------------------------------------
#  La regla que salio de la atribucion
# ---------------------------------------------------------------------------


def _panel_con_vehiculo(n=30):
    """Panel donde un nombre no tiene NINGUN fundamental, como un fideicomiso."""
    filas = []
    for i in range(n):
        vehiculo = i == 0
        filas.append({
            "date": pd.Timestamp("2024-01-31"),
            "ticker": "VEHICULO" if vehiculo else f"T{i:02d}",
            "sector": "Financials",
            # El vehiculo no tiene ingresos ni patrimonio: sus metricas
            # fundamentales quedan en NaN, igual que GBTC en el panel real.
            "earnings_yield": np.nan if vehiculo else 0.01 + i * 0.001,
            "fcf_yield": np.nan if vehiculo else 0.01 + i * 0.001,
            "roic": np.nan if vehiculo else 0.05 + i * 0.002,
            "debt_to_equity": np.nan if vehiculo else 1.0,
            # Pero SI tiene precio, asi que puntua altisimo en momentum.
            "momentum": 5.0 if vehiculo else 0.01 * i,
            "volatility": 0.10 if vehiculo else 0.30,
        })
    return pd.DataFrame(filas)


def test_sin_fundamentales_no_hay_score_compuesto(make_cfg):
    cfg = make_cfg(**{"factors.processing.require_fundamental_score": True})
    out = build_scores(_panel_con_vehiculo(), cfg).set_index("ticker")

    # Puntua alto en los factores de precio...
    assert out.loc["VEHICULO", "score_momentum"] > 0
    # ...pero no recibe compuesto, asi que no puede entrar en cartera.
    assert pd.isna(out.loc["VEHICULO", "score_composite"])
    assert out.loc["T15", "score_composite"] == pytest.approx(
        out.loc["T15", "score_composite"]
    )


def test_sin_la_regla_el_vehiculo_entra_por_momentum(make_cfg):
    """Fija el comportamiento anterior, para que se vea que la regla hace falta."""
    cfg = make_cfg(**{"factors.processing.require_fundamental_score": False})
    out = build_scores(_panel_con_vehiculo(), cfg).set_index("ticker")
    assert not pd.isna(out.loc["VEHICULO", "score_composite"])
    # Y no solo entra: con momentum 5.0 y la menor volatilidad, lidera.
    assert out["score_composite"].idxmax() == "VEHICULO"


def test_la_regla_no_estorba_a_un_modelo_de_solo_precio(make_cfg):
    """Con valor y calidad apagados, no hay fundamentales que exigir."""
    cfg = make_cfg(**{
        "factors.processing.require_fundamental_score": True,
        "factors.weights": {"value": 0.0, "quality": 0.0, "momentum": 1.0, "lowvol": 0.0},
    })
    out = build_scores(_panel_con_vehiculo(), cfg).set_index("ticker")
    assert not pd.isna(out.loc["VEHICULO", "score_composite"])
