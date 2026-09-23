"""Partes puras del ensamblaje del panel: calendario, dilucion y retorno futuro."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig import panel as panel_mod
from sfc_tfsig.data import prices as prices_mod


def _sessions(start="2019-06-03", periods=900):
    return pd.bdate_range(start=start, periods=periods)


def test_fechas_de_rebalanceo_mensuales_caen_en_fin_de_mes(cfg):
    sessions = _sessions()
    fechas = panel_mod.rebalance_dates(cfg, sessions)
    assert len(fechas) > 10
    assert all(f.is_month_end for f in fechas)
    assert fechas[0] >= pd.Timestamp(cfg.get("calendar.start"))


def test_rebalanceo_trimestral_da_una_cuarta_parte_de_fechas(make_cfg):
    sessions = _sessions()
    mensual = panel_mod.rebalance_dates(make_cfg(), sessions)
    trimestral = panel_mod.rebalance_dates(make_cfg(**{"calendar.rebalance": "Q"}), sessions)
    assert len(trimestral) == pytest.approx(len(mensual) / 3, abs=2)


def test_precio_as_of_toma_el_ultimo_cierre_anterior():
    sessions = pd.to_datetime(["2023-01-30", "2023-01-31", "2023-02-01"])
    close = pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=sessions)
    precio = panel_mod._as_of_price_frame(close, pd.Timestamp("2023-01-31"))
    assert precio["AAA"] == 11.0
    # Un domingo: se usa el viernes anterior, no el lunes siguiente.
    precio_domingo = panel_mod._as_of_price_frame(close, pd.Timestamp("2023-01-30 23:59"))
    assert precio_domingo["AAA"] == 10.0


def test_dilucion_compara_contra_el_mismo_mes_del_ano_anterior():
    fechas = pd.date_range("2022-01-31", periods=14, freq="ME")
    panel = pd.DataFrame({
        "date": fechas,
        "ticker": ["AAA"] * 14,
        "shares": np.linspace(100.0, 113.0, 14),
    })
    out = panel_mod._add_shares_growth(panel)
    fila = out[out["date"] == pd.Timestamp("2023-01-31")].iloc[0]
    # 112 acciones frente a 100 doce meses antes.
    assert fila["shares_growth"] == pytest.approx(0.12, abs=1e-9)


def test_sin_historia_de_un_ano_la_dilucion_queda_vacia():
    fechas = pd.date_range("2022-01-31", periods=6, freq="ME")
    panel = pd.DataFrame({"date": fechas, "ticker": ["AAA"] * 6, "shares": [100.0] * 6})
    out = panel_mod._add_shares_growth(panel)
    assert out["shares_growth"].isna().all()


def test_el_retorno_futuro_es_del_mes_siguiente_no_del_actual():
    sessions = pd.bdate_range("2023-01-02", periods=120)
    close = pd.DataFrame({"AAA": np.linspace(100.0, 160.0, len(sessions))}, index=sessions)
    mensual = prices_mod.month_end(close)

    panel = pd.DataFrame({
        "date": mensual.index[:3],
        "ticker": ["AAA"] * 3,
        "score_composite": [1.0, 1.0, 1.0],
    })
    out = panel_mod.add_forward_returns(panel, close)

    primera = out.iloc[0]
    esperado = mensual["AAA"].iloc[1] / mensual["AAA"].iloc[0] - 1.0
    assert primera["forward_return"] == pytest.approx(esperado)
    # El ultimo mes del panel no tiene mes siguiente dentro de la muestra.
    assert not pd.isna(out.iloc[1]["forward_return"])


def test_momentum_salta_el_ultimo_mes():
    """El mes mas reciente NO entra en la ventana de momentum."""
    sessions = pd.bdate_range("2022-01-03", "2023-08-31")
    precios = pd.Series(100.0, index=sessions)
    # Subida fuerte SOLO dentro del ultimo mes natural de la serie. Acotarlo por
    # mes natural y no por numero de sesiones importa: 21 sesiones se reparten
    # entre dos meses y entonces el salto ya esta dentro de la ventana, que es
    # justo lo que el test quiere descartar.
    ultimo_mes = sessions[sessions.to_period("M") == sessions[-1].to_period("M")]
    precios.loc[ultimo_mes] = 200.0
    close = pd.DataFrame({"AAA": precios})

    mom = prices_mod.momentum(close, lookback_m=12, skip_m=1)
    ultimo = mom["AAA"].dropna().iloc[-1]
    # Con skip=1, el salto del ultimo mes queda fuera: el momentum sigue en cero.
    assert ultimo == pytest.approx(0.0, abs=1e-9)


def test_volatilidad_realizada_sube_con_la_dispersion():
    sessions = pd.bdate_range("2022-01-03", periods=400)
    rng = np.random.default_rng(11)
    tranquila = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, len(sessions))))
    agitada = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(sessions))))
    close = pd.DataFrame({"CALMA": tranquila, "RUIDO": agitada}, index=sessions)

    vol = prices_mod.realized_volatility(close, window_d=252).dropna()
    assert vol["RUIDO"].iloc[-1] > vol["CALMA"].iloc[-1] * 3


def test_cobertura_ordena_de_peor_a_mejor():
    panel = pd.DataFrame({
        "roic": [1.0, 2.0, np.nan, np.nan],
        "earnings_yield": [0.1, 0.2, 0.3, 0.4],
    })
    tabla = panel_mod.coverage_report(panel, ["roic", "earnings_yield"])
    assert tabla.iloc[0]["metric"] == "roic"
    assert tabla.iloc[0]["coverage"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
#  Historia minima de precios
# ---------------------------------------------------------------------------


def test_historia_en_meses_se_mide_desde_el_primer_cierre():
    primero = pd.Series({
        "VIEJA": pd.Timestamp("2010-01-04"),
        "OPI": pd.Timestamp("2026-05-13"),
        "FUTURA": pd.Timestamp("2026-10-01"),
    })
    meses = panel_mod.history_months_at(primero, pd.Timestamp("2026-08-31"))
    assert meses["VIEJA"] > 190
    assert meses["OPI"] == pytest.approx(3.6, abs=0.1)
    # Todavia no cotizaba: negativo, y el filtro lo excluye.
    assert meses["FUTURA"] < 0


def test_una_opi_de_tres_meses_no_entra_al_universo(cfg):
    """El bug: `min_history_months = 24` estaba declarado y no se aplicaba."""
    from sfc_tfsig import universe as universe_mod

    candidatas = pd.DataFrame({"ticker": ["VIEJA", "OPI"], "sector": ["Tech", "Tech"]})
    elegibles = universe_mod.apply_liquidity_filters(
        candidatas, cfg,
        price=pd.Series({"VIEJA": 50.0, "OPI": 50.0}),
        dollar_volume=pd.Series({"VIEJA": 1e8, "OPI": 1e8}),
        market_cap=pd.Series({"VIEJA": 1e10, "OPI": 1e10}),
        history_months=pd.Series({"VIEJA": 200.0, "OPI": 3.6}),
    )
    assert elegibles["ticker"].tolist() == ["VIEJA"]


def test_el_filtro_de_historia_es_obligatorio_no_opcional(cfg):
    """Sin valor por defecto a proposito: olvidarlo tiene que fallar, no pasar."""
    import inspect
    from sfc_tfsig import universe as universe_mod

    parametro = inspect.signature(universe_mod.apply_liquidity_filters).parameters["history_months"]
    assert parametro.default is inspect.Parameter.empty
