"""Modelo factorial: recupera una estructura de factores inventada y la reparte bien."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig.risk import factor_model as fm

N = 300


@pytest.fixture
def mercado():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=380)
    tickers = [f"T{i}" for i in range(N)]
    sectors = np.array(["Technology", "Financials", "Energy"])[rng.integers(0, 3, N)]
    momentum = rng.normal(0, 1, N)
    market = rng.normal(0.0003, 0.01, len(dates))
    mom_factor = rng.normal(0, 0.004, len(dates))
    rets = market[:, None] + mom_factor[:, None] * momentum[None, :] + rng.normal(0, 0.015, (len(dates), N))
    close = pd.DataFrame((1 + rets).cumprod(axis=0) * 50.0, index=dates, columns=tickers)
    month_ends = pd.date_range("2023-01-31", dates[-1], freq="ME")
    panel = pd.concat([
        pd.DataFrame({"date": d, "ticker": tickers, "sector": sectors, "score_momentum": momentum,
                      "score_value": rng.normal(0, 1, N), "market_cap": np.exp(rng.normal(22, 1, N))})
        for d in month_ends
    ])
    return {"close": close, "panel": panel, "sectors": pd.Series(sectors, index=tickers),
            "momentum": pd.Series(momentum, index=tickers), "dates": dates}


def _cartera(m, names):
    w = pd.Series(1.0 / len(names), index=names)
    return w, m["sectors"].reindex(names)


def test_una_cartera_de_ganadores_carga_su_riesgo_en_momentum(mercado):
    w, s = _cartera(mercado, list(mercado["momentum"].nlargest(30).index))
    res = fm.factor_risk(w, s, mercado["close"], mercado["panel"], mercado["dates"][-1])
    assert res.exposures["momentum"] > 1.5
    assert res.decomposition.loc["momentum", "variance_share"] > 0.2


def test_una_cartera_neutra_en_momentum_no_lo_carga(mercado):
    w, s = _cartera(mercado, list(mercado["momentum"].abs().nsmallest(30).index))
    res = fm.factor_risk(w, s, mercado["close"], mercado["panel"], mercado["dates"][-1])
    assert abs(res.decomposition.loc["momentum", "variance_share"]) < 0.02


def test_la_descomposicion_suma_uno_y_la_volatilidad(mercado):
    w, s = _cartera(mercado, list(mercado["momentum"].nlargest(30).index))
    res = fm.factor_risk(w, s, mercado["close"], mercado["panel"], mercado["dates"][-1])
    assert res.decomposition["variance_share"].sum() == pytest.approx(1.0)
    assert res.decomposition["vol_contribution"].sum() == pytest.approx(res.total_vol)
    assert res.total_vol ** 2 == pytest.approx(res.factor_vol ** 2 + res.specific_vol ** 2)


def test_la_volatilidad_del_factor_se_recupera(mercado):
    returns, _, r2, _ = fm.estimate_factor_model(mercado["close"], mercado["panel"], mercado["dates"][-1])
    assert returns["momentum"].std() * np.sqrt(252) == pytest.approx(0.004 * np.sqrt(252), rel=0.2)
    assert 0.0 < r2.mean() < 1.0


def test_las_exposiciones_de_cada_dia_son_de_un_panel_anterior(mercado):
    """Un panel fechado el mismo dia o despues no puede explicar el retorno de ese dia."""
    _, _, _, by_date = fm.estimate_factor_model(mercado["close"], mercado["panel"], mercado["dates"][-1])
    assert all(d < mercado["dates"][-1] for d in by_date)


def test_un_nombre_fuera_del_panel_se_modela_con_su_sector(mercado):
    w = pd.Series({"T0": 0.5, "FANTASMA": 0.5})
    s = pd.Series({"T0": mercado["sectors"]["T0"], "FANTASMA": "Energy"})
    close = mercado["close"].assign(FANTASMA=mercado["close"]["T0"])
    res = fm.factor_risk(w, s, close, mercado["panel"], mercado["dates"][-1])
    assert any("FANTASMA" in warning for warning in res.warnings)
    assert np.isfinite(res.total_vol)


def test_sin_capitalizacion_la_regresion_pondera_igual(mercado):
    panel = mercado["panel"].drop(columns="market_cap")
    w, s = _cartera(mercado, list(mercado["momentum"].nlargest(30).index))
    res = fm.factor_risk(w, s, mercado["close"], panel, mercado["dates"][-1])
    assert "size" not in res.exposures.index
    assert res.exposures["momentum"] > 1.5


def test_panel_sin_fechas_anteriores_da_error_con_mensaje(mercado):
    early = mercado["dates"][0] - pd.Timedelta(days=30)
    with pytest.raises(ValueError, match="anteriores"):
        fm.estimate_factor_model(mercado["close"], mercado["panel"], early)


def test_con_pocos_nombres_no_hay_regresion(mercado):
    panel = mercado["panel"][mercado["panel"]["ticker"].isin([f"T{i}" for i in range(20)])]
    with pytest.raises(ValueError, match="suficientes nombres"):
        fm.estimate_factor_model(mercado["close"], panel, mercado["dates"][-1])


def test_el_panel_del_mismo_dia_no_explica_el_retorno_de_ese_dia():
    """Trampa a proposito: el score de cada fin de mes predice EXACTAMENTE el
    retorno de ese mismo dia. Si el modelo usara el panel del dia (mirar al
    futuro), el factor capturaria ese retorno y el R2 de esos dias seria casi
    1. Con el panel anterior, que no sabe nada de ese dia, ronda cero."""
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2023-01-02", periods=300)
    tickers = [f"T{i}" for i in range(120)]
    month_ends = pd.DatetimeIndex(sorted({d for d in dates if d == dates[dates.month == d.month][-1]}))
    rets = pd.DataFrame(rng.normal(0, 0.01, (len(dates), len(tickers))), index=dates, columns=tickers)
    panels = []
    for d in month_ends:
        score = rng.normal(0, 1, len(tickers))
        rets.loc[d] = 0.03 * score            # el retorno del dia ES el score del dia
        panels.append(pd.DataFrame({"date": d, "ticker": tickers, "sector": "Energy",
                                    "score_momentum": score}))
    close = (1 + rets).cumprod() * 50.0
    panel = pd.concat(panels)
    factor_returns, _, r2, _ = fm.estimate_factor_model(close, panel, dates[-1], window=280)
    trap_days = [d for d in month_ends if d in r2.index]
    assert len(trap_days) >= 5
    assert r2.loc[trap_days].max() < 0.3
    assert factor_returns.loc[trap_days, "momentum"].abs().max() < 0.01
