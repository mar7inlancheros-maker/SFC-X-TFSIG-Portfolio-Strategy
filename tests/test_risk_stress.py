"""Pruebas de estres con precios y choques cuyo resultado se calcula a mano."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig.risk import stress


def _precios() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    frame = pd.DataFrame(index=dates)
    frame["SPY"] = 100.0
    frame["AAA"] = 50.0
    frame["NUEVA"] = np.nan
    # Episodio: del 2020-02-03 al 2020-02-28 el mercado cae un 20% y AAA un 30%.
    crash = frame.index >= "2020-02-10"
    frame.loc[crash, "SPY"] = 80.0
    frame.loc[crash, "AAA"] = 35.0
    # NUEVA empieza a cotizar despues del inicio del episodio.
    frame.loc[frame.index >= "2020-02-20", "NUEVA"] = 10.0
    return frame


def test_el_escenario_historico_usa_precios_reales_y_beta_para_el_resto():
    weights = pd.Series({"AAA": 0.5, "NUEVA": 0.4})
    betas = pd.Series({"AAA": 1.2, "NUEVA": 1.5})
    scenario = {"name": "caida", "start": "2020-02-03", "end": "2020-02-28"}

    summary, detail = stress.historical_stress(weights, betas, _precios(), "SPY", [scenario])
    row = summary.iloc[0]
    assert row["benchmark"] == pytest.approx(-0.20)
    # AAA real (-30%) con 50%; NUEVA no cotizaba -> beta 1.5 x -20% = -30%, con 40%.
    assert row["portfolio"] == pytest.approx(0.5 * -0.30 + 0.4 * -0.30)
    assert row["weight_proxied"] == pytest.approx(0.4)
    assert bool(detail.set_index("ticker").loc["NUEVA", "proxied"])
    # Relativo frente a la parte invertida (90%) del benchmark.
    assert row["relative"] == pytest.approx(row["portfolio"] - 0.9 * -0.20)


def test_un_episodio_sin_benchmark_se_marca_y_no_se_inventa():
    weights = pd.Series({"AAA": 1.0})
    scenario = {"name": "antiguo", "start": "2008-09-12", "end": "2008-11-20"}
    summary, _ = stress.historical_stress(weights, pd.Series({"AAA": 1.0}), _precios(),
                                          "SPY", [scenario])
    assert summary.iloc[0]["status"] != "ok"
    assert "portfolio" not in summary or pd.isna(summary.iloc[0].get("portfolio"))


def test_un_precio_viejo_no_cuenta_como_precio_del_dia():
    frame = pd.DataFrame({"X": [10.0, np.nan, np.nan]},
                         index=pd.to_datetime(["2020-01-01", "2020-01-20", "2020-02-20"]))
    assert np.isnan(stress.price_on_or_before(frame, "2020-02-20")["X"])


def test_la_estrategia_entonces_sale_del_nav_si_cubre_el_episodio():
    prices = _precios()
    nav = prices["SPY"] * 1000.0
    summary, _ = stress.historical_stress(
        pd.Series({"AAA": 1.0}), pd.Series({"AAA": 1.0}), prices, "SPY",
        [{"name": "caida", "start": "2020-02-03", "end": "2020-02-28"}], strategy_nav=nav,
    )
    assert summary.iloc[0]["strategy_then"] == pytest.approx(-0.20)


def test_el_escenario_hipotetico_suma_beta_sector_y_factor():
    weights = pd.Series({"BANCO": 0.5, "TECH": 0.5})
    sectors = pd.Series({"BANCO": "Financials", "TECH": "Technology"})
    betas = pd.Series({"BANCO": 1.0, "TECH": 1.5})
    scores = pd.DataFrame({"score_momentum": [0.0, 2.0]}, index=["BANCO", "TECH"])
    scenario = {"name": "x", "market": -0.20, "sectors": {"Financials": -0.10},
                "factors": {"momentum": -0.05}}

    summary, detail = stress.hypothetical_stress(weights, sectors, betas, [scenario],
                                                 factor_scores=scores)
    shocks = detail.set_index("ticker")["shock"]
    assert shocks["BANCO"] == pytest.approx(-0.20 - 0.10)
    assert shocks["TECH"] == pytest.approx(1.5 * -0.20 + 2.0 * -0.05)
    row = summary.iloc[0]
    assert row["portfolio"] == pytest.approx(0.5 * -0.30 + 0.5 * -0.40)
    assert row["from_beta"] + row["from_sectors"] + row["from_factors"] == pytest.approx(row["portfolio"])


def test_ningun_choque_baja_de_menos_cien():
    summary, detail = stress.hypothetical_stress(
        pd.Series({"X": 1.0}), pd.Series({"X": "Technology"}), pd.Series({"X": 3.0}),
        [{"name": "fin del mundo", "market": -0.60}],
    )
    assert detail["shock"].min() == pytest.approx(-1.0)
    assert summary.iloc[0]["portfolio"] == pytest.approx(-1.0)


def test_factor_sin_scores_se_avisa_y_cuenta_cero():
    summary, _ = stress.hypothetical_stress(
        pd.Series({"X": 1.0}), pd.Series({"X": "Technology"}), pd.Series({"X": 1.0}),
        [{"name": "mom", "market": 0.0, "factors": {"momentum": -0.1}}],
    )
    assert summary.iloc[0]["portfolio"] == pytest.approx(0.0)
    assert summary.iloc[0]["missing_factors"] == "momentum"


def test_prueba_inversa_divide_por_la_beta_invertida():
    table = stress.reverse_stress(1.25, 0.8, [0.10, 0.30])
    assert table.loc[0, "market_move"] == pytest.approx(-0.10 / 1.0)
    assert table.loc[1, "market_move"] == pytest.approx(-0.30)
    # Con beta 0.2, perder un 30% exigiria un mercado a -150%: imposible.
    assert not stress.reverse_stress(0.2, 1.0, [0.30]).loc[0, "plausible"]


def test_peores_ventanas_encuentran_la_caida():
    nav = pd.Series(100.0, index=pd.bdate_range("2021-01-01", periods=60))
    nav.iloc[30:] = 70.0
    worst = stress.worst_windows(nav, None, [1, 5]).set_index("window_d")
    assert worst.loc[1, "strategy"] == pytest.approx(-0.30)
    assert str(worst.loc[1, "end"]) == str(nav.index[30].date())


# ---------------------------------------------------------------------------
#  Historia reconstruida, VaR estresado y choques de covarianza
# ---------------------------------------------------------------------------


def test_la_historia_reconstruida_usa_beta_donde_no_hay_precio():
    prices = _precios()
    weights = pd.Series({"AAA": 0.5, "NUEVA": 0.5})
    backcast, proxied = stress.book_backcast(weights, pd.Series({"AAA": 1.0, "NUEVA": 2.0}), prices, "SPY")
    crash_day = prices.index[prices.index.get_loc(pd.Timestamp("2020-02-10"))]
    # Ese dia: AAA -30% real, NUEVA sin precio -> 2 x -20% = -40%.
    assert backcast.loc[crash_day] == pytest.approx(0.5 * -0.30 + 0.5 * -0.40)
    assert proxied.loc[crash_day] == pytest.approx(0.5)
    assert proxied.iloc[-1] == pytest.approx(0.0)


def test_el_var_estresado_encuentra_la_peor_ventana():
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.005, 600)
    storm = rng.normal(0, 0.04, 100)
    r = pd.Series(np.concatenate([calm, storm, calm]), index=pd.bdate_range("2010-01-01", periods=1300))
    out = stress.stressed_var(r, 0.99, 250)
    assert out["stressed_var"] > 3 * out["current_var"]
    storm_start, storm_end = r.index[600], r.index[699]
    assert out["window_start"] <= storm_end and out["window_end"] >= storm_start


def test_choques_de_covarianza():
    cov = pd.DataFrame([[1e-4, 0.0], [0.0, 1e-4]], index=["A", "B"], columns=["A", "B"])
    w = pd.Series({"A": 0.5, "B": 0.5})
    table = stress.covariance_shocks(w, cov, [
        {"name": "base"}, {"name": "vol x2", "vol_mult": 2.0},
        {"name": "correlacion 1", "corr_blend": 1.0},
    ]).set_index("scenario")
    assert table.loc["vol x2", "vs_base"] == pytest.approx(2.0)
    # Dos nombres incorrelados al 50%: con correlacion 1 la vol sube raiz(2).
    assert table.loc["correlacion 1", "vs_base"] == pytest.approx(np.sqrt(2.0))


def test_estres_historico_sin_columna_del_benchmark():
    summary, detail = stress.historical_stress(
        pd.Series({"AAA": 1.0}), pd.Series({"AAA": 1.0}), _precios().drop(columns="SPY"), "SPY",
        [{"name": "caida", "start": "2020-02-03", "end": "2020-02-28"}],
    )
    assert summary.iloc[0]["status"] == "no prices for SPY"
    assert detail.empty


def test_la_historia_reconstruida_exige_el_benchmark():
    with pytest.raises(ValueError, match="SPY"):
        stress.book_backcast(pd.Series({"AAA": 1.0}), pd.Series({"AAA": 1.0}),
                             _precios().drop(columns="SPY"), "SPY")


def test_var_estresado_con_poca_historia_devuelve_vacio():
    assert stress.stressed_var(pd.Series(np.zeros(100)), 0.99, 250) == {}


def test_un_nombre_sin_beta_ni_precio_se_mueve_como_el_mercado():
    summary, _ = stress.historical_stress(
        pd.Series({"NUEVA": 1.0}), pd.Series(dtype=float), _precios(), "SPY",
        [{"name": "caida", "start": "2020-02-03", "end": "2020-02-28"}],
    )
    assert summary.iloc[0]["portfolio"] == pytest.approx(-0.20)


def test_la_correlacion_de_crisis_conserva_una_matriz_valida():
    rng = np.random.default_rng(0)
    cov = np.cov(rng.normal(size=(300, 6)), rowvar=False)
    shocked = stress._shocked_cov(cov, 2.0, 0.5)
    assert np.linalg.eigvalsh(shocked).min() > -1e-12
    np.testing.assert_allclose(np.diag(shocked), 4.0 * np.diag(cov))


def test_el_peso_aproximado_se_pondera_por_el_peso_de_cada_nombre():
    prices = _precios()
    weights = pd.Series({"AAA": 0.8, "NUEVA": 0.2})
    _, proxied = stress.book_backcast(weights, pd.Series({"AAA": 1.0, "NUEVA": 1.0}), prices, "SPY")
    assert proxied.iloc[0] == pytest.approx(0.2)


def test_en_un_escenario_hipotetico_un_nombre_sin_beta_tiene_beta_uno():
    _summary, detail = stress.hypothetical_stress(
        pd.Series({"X": 1.0}), pd.Series({"X": "Energy"}), pd.Series(dtype=float),
        [{"name": "mercado -20%", "market": -0.20}],
    )
    assert detail["shock"].iloc[0] == pytest.approx(-0.20)


def test_la_peor_ventana_empieza_h_sesiones_antes_de_terminar():
    nav = pd.Series(100.0, index=pd.bdate_range("2021-01-01", periods=60))
    nav.iloc[30:35] = [95.0, 90.0, 85.0, 80.0, 75.0]
    nav.iloc[35:] = 75.0
    worst = stress.worst_windows(nav, None, [5]).set_index("window_d")
    assert worst.loc[5, "strategy"] == pytest.approx(-0.25)
    assert str(worst.loc[5, "start"]) == str(nav.index[29].date())
    assert str(worst.loc[5, "end"]) == str(nav.index[34].date())
