"""Motor long/short: construccion, neutralidad, backtest, estres y Monte Carlo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine.backtest import engine as bt
from quant_engine.data.cleaning import daily_rf
from quant_engine.portfolio import analytics as pa
from quant_engine.portfolio import construction as C
from quant_engine.portfolio import neutral as N
from quant_engine.portfolio.constraints import ConstructionParams, cap_leg, check, exposures
from quant_engine.risk import montecarlo, stress

NAMES = list("ABCDEFGHIJ")
SIGNS = pd.Series([1.0] * 5 + [-1.0] * 5, index=NAMES)
PARAMS = ConstructionParams(gross=2.0, net=0.0, max_position=0.4)


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(10)
    r = pd.DataFrame(rng.normal(0.0003, 0.015, (800, 10)) + rng.normal(0, 0.01, (800, 1)), columns=NAMES)
    betas = pd.Series(np.linspace(0.6, 1.5, 10), index=NAMES)
    return r, r.cov(), r.std() * np.sqrt(252), betas


# ---------------------------------------------------------------------------
#  Construccion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", C.METHODS)
def test_todos_los_metodos_respetan_signos_bruta_y_topes(market, method):
    r, cov, vol, betas = market
    mu = C.shrink_means(r)
    mu[SIGNS > 0] += 0.0005  # los largos esperan mas: max_sharpe tiene solucion
    res = C.build(method, SIGNS, cov=cov, vol=vol, params=PARAMS, mu=mu, betas=betas)
    assert res.status == "ok", res.note
    assert check(res.weights, SIGNS, PARAMS) == []


def test_paridad_de_riesgo_iguala_contribuciones_con_cortos(market):
    _, cov, vol, _ = market
    w = C.build("risk_parity", SIGNS, cov=cov, vol=vol, params=PARAMS).weights
    rc = pa.ex_ante(w, cov)["risk_contribution"]
    assert rc.sum() == pytest.approx(1.0)
    assert rc.std() < 1e-3


def test_minima_varianza_no_supera_a_equiponderado(market):
    _, cov, vol, _ = market
    ew = C.build("equal_weight", SIGNS, cov=cov, vol=vol, params=PARAMS).weights
    mv = C.build("min_variance", SIGNS, cov=cov, vol=vol, params=PARAMS).weights
    assert pa.ex_ante(mv, cov)["vol_annual"] <= pa.ex_ante(ew, cov)["vol_annual"] + 1e-9


def test_max_sharpe_sin_solucion_se_declara_no_se_fuerza(market):
    """Si los cortos esperan mas que los largos, no hay Sharpe positivo con esos signos."""
    r, cov, vol, _ = market
    mu = pd.Series(0.0, index=NAMES)
    mu[SIGNS > 0], mu[SIGNS < 0] = -0.001, 0.001
    res = C.build("max_sharpe", SIGNS, cov=cov, vol=vol, params=PARAMS, mu=mu)
    assert res.status == "failed"
    assert "sin solucion" in res.note


def test_tope_imposible_lanza_error_no_incumple_en_silencio():
    with pytest.raises(ValueError, match="no cabe"):
        cap_leg(pd.Series(1.0, index=list("AB")), 1.0, 0.3)


def test_tope_redistribuye_sin_perder_exposicion():
    leg = cap_leg(pd.Series([10.0, 1.0, 1.0, 1.0]), 1.0, 0.4)
    assert leg.sum() == pytest.approx(1.0)
    assert leg.max() <= 0.4 + 1e-12


def test_james_stein_contrae_hacia_la_media():
    rng = np.random.default_rng(11)
    r = pd.DataFrame(rng.normal(0, 0.02, (60, 10)), columns=NAMES)
    shrunk = C.shrink_means(r)
    assert shrunk.std() < r.mean().std()


# ---------------------------------------------------------------------------
#  Neutralidad
# ---------------------------------------------------------------------------


def test_beta_neutral_anula_la_beta_y_mueve_la_neta(market):
    _, _, _, betas = market
    ew = C.equal_weight(SIGNS, PARAMS)
    bn, note = N.beta_neutral_legs(ew, betas, PARAMS)
    assert N.portfolio_beta(bn, betas) == pytest.approx(0.0, abs=1e-9)
    assert exposures(bn)["gross"] == pytest.approx(2.0)
    assert exposures(bn)["net"] != pytest.approx(0.0)


@pytest.mark.parametrize("method", ["min_variance", "equal_weight"])
def test_beta_neutral_como_parametro_de_construccion(market, method):
    _, cov, vol, betas = market
    params = ConstructionParams(gross=2.0, net=0.0, max_position=0.5, beta_neutral=True)
    res = C.build(method, SIGNS, cov=cov, vol=vol, params=params, betas=betas)
    assert res.status == "ok"
    assert N.portfolio_beta(res.weights, betas) == pytest.approx(0.0, abs=1e-6)


def test_sector_neutral_reduce_la_exposicion_y_declara_lo_imposible():
    sectors = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Energy", "E": "Energy",
               "F": "Tech", "G": "Energy", "H": "Energy", "I": "Health", "J": "Health"}
    ew = C.equal_weight(SIGNS, PARAMS)
    before = N.sector_exposure(ew, sectors).abs().sum()
    w, note, one_sided = N.sector_neutral(SIGNS, sectors, PARAMS)
    after = N.sector_exposure(w, sectors).abs().sum()
    assert after < before
    assert one_sided == ["Health"]


# ---------------------------------------------------------------------------
#  Backtest
# ---------------------------------------------------------------------------


def _flat(n=300, names=("A", "B", "BENCH")):
    idx = pd.bdate_range("2021-01-01", periods=n)
    return pd.DataFrame(0.0, index=idx, columns=list(names))


def _cfg(**kw):
    base = dict(method="equal_weight", rebalance="monthly",
                params=ConstructionParams(gross=2.0, net=0.0, max_position=1.0),
                capital=1_000_000.0, transaction_cost=0.0, borrow_cost=0.0, risk_free_rate=0.0,
                estimation_window_d=60, min_estimation_obs=40)
    base.update(kw)
    return bt.BacktestConfig(**base)


SIGNS_AB = pd.Series({"A": 1.0, "B": -1.0})


def test_precios_planos_neutral_en_dolares_rinde_rf_menos_prestamo():
    """100/100 sin movimiento: la caja (con el producto del corto) rinde rf; el prestamo cuesta."""
    r = _flat()
    cfg = _cfg(risk_free_rate=0.05, borrow_cost=0.01)
    res = bt.run(r, "BENCH", bt.constant_schedule(SIGNS_AB), cfg, start=r.index[80], mode="A")
    days = len(res.nav) - 1
    expected = (1 + daily_rf(0.05) - daily_rf(0.01)) ** days
    assert res.nav.iloc[-1] / res.nav.iloc[0] == pytest.approx(expected, rel=1e-4)


def test_un_corto_pierde_cuando_su_precio_sube():
    r = _flat()
    start = r.index[80]
    r.loc[r.index > start + pd.Timedelta(days=40), "B"] = 0.0
    jump_day = r.index[r.index > start + pd.Timedelta(days=40)][0]
    r.loc[jump_day, "B"] = 0.10  # el corto sube un 10%
    res = bt.run(r, "BENCH", bt.constant_schedule(SIGNS_AB), _cfg(), start=start, mode="A")
    before = res.nav.loc[:jump_day].iloc[-2]
    after = res.nav.loc[jump_day]
    # Corto del 100% del capital: pierde el 10% del capital.
    assert after / before - 1 == pytest.approx(-0.10, rel=1e-6)


def test_la_ejecucion_es_al_dia_siguiente_de_la_senal():
    """Un salto el dia siguiente a la senal NO se captura: se compra despues."""
    r = _flat()
    start = r.index[80]
    signal = bt.rebalance_dates(r.index, "monthly", start)[0]
    exec_day = r.index[r.index > signal][0]
    r.loc[exec_day, "A"] = 0.20
    res = bt.run(r, "BENCH", bt.constant_schedule(SIGNS_AB), _cfg(), start=start, mode="A")
    # El NAV al cierre de la ejecucion no refleja el +20% de A.
    assert res.nav.loc[exec_day] == pytest.approx(1_000_000.0)


def test_los_costes_se_cobran_sobre_el_nocional_operado():
    r = _flat()
    res = bt.run(r, "BENCH", bt.constant_schedule(SIGNS_AB), _cfg(transaction_cost=0.001),
                 start=r.index[80], mode="A")
    first = res.rebalances.iloc[0]
    # Primera cartera: se compran 1M y se venden en corto 1M -> 2M operados.
    assert first["cost"] == pytest.approx(2_000_000 * 0.001)


def test_senales_historicas_desde_csv(tmp_path):
    path = tmp_path / "signals.csv"
    pd.DataFrame({"date": ["2021-01-01", "2021-01-01", "2021-06-01"],
                  "ticker": ["A", "B", "A"], "signal": ["LONG", "SHORT", "FLAT"]}).to_csv(path, index=False)
    schedule = bt.csv_schedule(path)
    assert schedule(pd.Timestamp("2021-03-01")).to_dict() == {"A": 1.0, "B": -1.0}
    assert schedule(pd.Timestamp("2021-07-01")).to_dict() == {"B": -1.0}
    assert schedule(pd.Timestamp("2020-12-01")) is None


def test_senal_desconocida_en_el_csv_falla(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2021-01-01"], "ticker": ["A"], "signal": ["MAYBE"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no reconocidas"):
        bt.csv_schedule(path)


# ---------------------------------------------------------------------------
#  Estres y Monte Carlo
# ---------------------------------------------------------------------------


def test_estres_historico_un_corto_gana_lo_que_cae_su_activo():
    idx = pd.bdate_range("2022-01-03", "2022-10-12")
    r = pd.DataFrame({"L": 0.0, "S": 0.0, "SPY": 0.0}, index=idx)
    r.iloc[1, r.columns.get_loc("S")] = -0.40
    r.iloc[1, r.columns.get_loc("SPY")] = -0.24
    table = stress.historical_scenarios(pd.Series({"L": 1.0, "S": -1.0}), r, "SPY")
    row = table[table["scenario"] == "2022 rate shock"].iloc[0]
    assert row["short_leg_pnl"] == pytest.approx(0.40)
    assert row["portfolio"] == pytest.approx(row["long_leg_pnl"] + row["short_leg_pnl"])


def test_ruptura_de_cobertura_sube_el_riesgo_de_una_cartera_cubierta():
    cov = pd.DataFrame([[0.0004, 0.00036], [0.00036, 0.0004]], index=list("LS"), columns=list("LS"))
    w = pd.Series({"L": 1.0, "S": -1.0})
    t = stress.covariance_shocks(w, cov, pd.Series({"L": 1.0, "S": -1.0}), [2.0]).set_index("shock")
    assert t.loc["hedge breakdown (long-short corr = 0)", "vol_annual"] > t.loc["base", "vol_annual"]


def test_monte_carlo_es_reproducible_con_semilla():
    r = pd.Series(np.random.default_rng(12).normal(0.0005, 0.01, 500))
    a = montecarlo.stationary_bootstrap(r, 200, 50, 10, seed=7)
    b = montecarlo.stationary_bootstrap(r, 200, 50, 10, seed=7)
    assert np.array_equal(a, b)


def test_deriva_neutra_centra_la_simulacion_en_rf():
    r = pd.Series(np.random.default_rng(13).normal(0.002, 0.01, 1000))  # deriva historica alta
    rf_d = daily_rf(0.04)
    neutral = r - r.mean() + rf_d
    paths = montecarlo.stationary_bootstrap(neutral, 4000, 252, 10, seed=1)
    out = montecarlo.summarize(paths, [0.1])
    # Riqueza esperada ~ 1 + rf, no la deriva historica (~1,65).
    assert out["expected_terminal_wealth"] == pytest.approx(1.04, abs=0.03)


def test_normal_neutral_rinde_rf_sobre_el_capital_no_bruta_por_rf():
    """Con rf en cada posicion con signo, una 100/100 rendia bruta x rf = 8%."""
    w = pd.Series({"L": 1.0, "S": -1.0})
    cov = pd.DataFrame(np.eye(2) * 1e-8, index=list("LS"), columns=list("LS"))
    paths = montecarlo.parametric_normal(w, pd.Series(0.0, index=list("LS")), cov, 2000, 252, seed=3,
                                         drift=daily_rf(0.04))
    out = montecarlo.summarize(paths, [0.1])
    assert out["expected_terminal_wealth"] == pytest.approx(1.04, abs=0.002)
