"""El motor de backtest, con precios inventados y resultados calculables a mano.

Si estos tests pasan, el motor contabiliza bien. Si fallan, cualquier numero que
salga del modelo es decorativo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig.backtest import benchmark_nav, run_backtest
from sfc_tfsig import metrics


def _sessions(start="2020-01-01", periods=400):
    return pd.bdate_range(start=start, periods=periods)


def _flat_prices(tickers, sessions, price=100.0):
    return pd.DataFrame(price, index=sessions, columns=tickers)


def _panel(dates, tickers, scores=None, sector="Tech", price=100.0):
    rows = []
    for date in dates:
        for i, ticker in enumerate(tickers):
            rows.append({
                "date": date,
                "ticker": ticker,
                "sector": sector if isinstance(sector, str) else sector[i],
                "price": price,
                "volatility": 0.2,
                "score_composite": (scores or {}).get(ticker, float(len(tickers) - i)),
            })
    return pd.DataFrame(rows)


def test_sin_movimiento_de_precios_el_nav_solo_baja_por_costes(make_cfg):
    """Precios planos: el unico cambio posible del NAV es el coste de operar."""
    cfg = make_cfg(**{"portfolio.n_positions": 10, "portfolio.buffer_rank": 10,
                      "portfolio.max_sector_w": 1.0, "portfolio.max_weight": 0.2,
                      "portfolio.cash_buffer": 0.0})
    sessions = _sessions(periods=200)
    tickers = [f"T{i:02d}" for i in range(10)]
    close = _flat_prices(tickers, sessions)
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    panel = _panel(dates, tickers)

    result = run_backtest(panel, close, cfg, progress=False)

    capital = float(cfg.get("backtest.initial_capital"))
    # Primera compra: se invierte el 100% del capital. La rotacion sale 0.5
    # porque la convencion de la industria divide entre dos -- solo se opera una
    # pata, no hay ventas. El COSTE si es sobre el nocional completo: 20 bps.
    primera = result.rebalances.iloc[0]
    assert primera["turnover"] == pytest.approx(0.5, abs=1e-6)
    assert primera["cost"] == pytest.approx(capital * 0.0020, rel=1e-6)

    # Despues, con precios planos, la cartera ya esta en su objetivo y lo unico
    # que se reajusta es el hueco que dejo el coste del mes anterior: una
    # rotacion residual del orden de 0.1%, no cero. Cero solo saldria si el
    # motor rebalanceara a pesos fijos ignorando el NAV, que es justo el atajo
    # que infla los backtests.
    residual = result.rebalances["turnover"].iloc[1:]
    assert (residual < 0.002).all()
    assert result.nav.iloc[-1] == pytest.approx(capital * (1 - 0.0020), rel=1e-3)
    assert result.nav.iloc[-1] < capital


def test_el_retardo_de_ejecucion_impide_capturar_el_salto_del_dia_de_la_senal(make_cfg):
    """El precio salta el dia de la senal: con lag 1 se compra DESPUES del salto.

    Este es el test que separa un backtest honesto de uno que se ve precioso.
    """
    cfg = make_cfg(**{"portfolio.n_positions": 10, "portfolio.buffer_rank": 10,
                      "portfolio.max_sector_w": 1.0, "portfolio.max_weight": 0.2,
                      "portfolio.cash_buffer": 0.0, "costs.commission_bps": 0.0,
                      "costs.spread_bps": 0.0, "costs.slippage_bps": 0.0})
    sessions = _sessions(periods=200)
    tickers = [f"T{i:02d}" for i in range(10)]
    close = _flat_prices(tickers, sessions)

    signal_date = pd.Timestamp("2020-01-31")
    salto = sessions[sessions > signal_date][0]
    # El dia siguiente a la senal, todo sube un 10% y se queda ahi.
    close.loc[close.index >= salto] = 110.0

    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    panel = _panel(dates, tickers)
    result = run_backtest(panel, close, cfg, progress=False)

    capital = float(cfg.get("backtest.initial_capital"))
    # Se compra a 110, no a 100: el salto NO se captura.
    assert result.trades.iloc[0]["price"] == pytest.approx(110.0)
    assert result.nav.iloc[-1] == pytest.approx(capital, rel=1e-6)


def test_una_subida_posterior_si_se_captura(make_cfg):
    cfg = make_cfg(**{"portfolio.n_positions": 10, "portfolio.buffer_rank": 10,
                      "portfolio.max_sector_w": 1.0, "portfolio.max_weight": 0.2,
                      "portfolio.cash_buffer": 0.0, "costs.commission_bps": 0.0,
                      "costs.spread_bps": 0.0, "costs.slippage_bps": 0.0})
    sessions = _sessions(periods=200)
    tickers = [f"T{i:02d}" for i in range(10)]
    close = _flat_prices(tickers, sessions)
    close.loc[close.index >= pd.Timestamp("2020-03-15")] = 120.0

    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    result = run_backtest(_panel(dates, tickers), close, cfg, progress=False)

    capital = float(cfg.get("backtest.initial_capital"))
    assert result.nav.iloc[-1] == pytest.approx(capital * 1.20, rel=1e-6)


def test_los_pesos_derivan_entre_rebalanceos(make_cfg):
    """Sin rebalanceo continuo: quien sube, pesa mas hasta el proximo ajuste."""
    cfg = make_cfg(**{"portfolio.n_positions": 10, "portfolio.buffer_rank": 10,
                      "portfolio.max_sector_w": 1.0, "portfolio.max_weight": 0.6,
                      "portfolio.cash_buffer": 0.0, "portfolio.min_weight": 0.0,
                      "costs.commission_bps": 0.0, "costs.spread_bps": 0.0,
                      "costs.slippage_bps": 0.0})
    sessions = _sessions(periods=200)
    tickers = ["AAA"] + [f"T{i:02d}" for i in range(9)]
    close = _flat_prices(tickers, sessions)
    close.loc[close.index >= pd.Timestamp("2020-02-15"), "AAA"] = 200.0

    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    panel = _panel(dates, tickers)
    result = run_backtest(panel, close, cfg, progress=False)

    # En el segundo rebalanceo, AAA ya vale el doble: hubo rotacion para volver
    # al objetivo, y por tanto una venta parcial de AAA.
    segundo = result.rebalances.iloc[1]
    assert segundo["turnover"] > 0.02
    ventas = result.trades[(result.trades["side"] == "SELL")]
    assert "AAA" in ventas["ticker"].tolist()


def test_costes_mas_altos_producen_peor_resultado(make_cfg):
    sessions = _sessions(periods=300)
    tickers = [f"T{i:02d}" for i in range(20)]
    rng = np.random.default_rng(7)
    paths = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, size=(len(sessions), 20)), axis=0))
    close = pd.DataFrame(paths, index=sessions, columns=tickers)

    dates = pd.date_range("2020-01-31", periods=10, freq="ME")
    # Scores que cambian cada mes: fuerza rotacion real.
    rows = []
    for j, date in enumerate(dates):
        for i, ticker in enumerate(tickers):
            rows.append({"date": date, "ticker": ticker, "sector": "Tech", "price": 100.0,
                         "volatility": 0.2, "score_composite": float((i + j) % 20)})
    panel = pd.DataFrame(rows)

    base = {"portfolio.n_positions": 10, "portfolio.buffer_rank": 10,
            "portfolio.max_sector_w": 1.0, "portfolio.max_weight": 0.2,
            "portfolio.cash_buffer": 0.0}
    barato = make_cfg(**base, **{"costs.commission_bps": 0.0, "costs.spread_bps": 0.0,
                                 "costs.slippage_bps": 0.0})
    caro = make_cfg(**base, **{"costs.commission_bps": 50.0, "costs.spread_bps": 50.0,
                               "costs.slippage_bps": 50.0})

    sin_costes = run_backtest(panel, close, barato, progress=False)
    con_costes = run_backtest(panel, close, caro, progress=False)

    assert con_costes.nav.iloc[-1] < sin_costes.nav.iloc[-1]
    assert con_costes.total_costs > sin_costes.total_costs


def test_el_fingerprint_de_la_config_viaja_con_el_resultado(make_cfg):
    cfg = make_cfg(**{"portfolio.n_positions": 10, "portfolio.buffer_rank": 10,
                      "portfolio.max_sector_w": 1.0})
    sessions = _sessions(periods=100)
    tickers = [f"T{i:02d}" for i in range(10)]
    close = _flat_prices(tickers, sessions)
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    result = run_backtest(_panel(dates, tickers), close, cfg, progress=False)
    assert result.config_fingerprint == cfg.fingerprint


# ---------------------------------------------------------------------------
#  Metricas
# ---------------------------------------------------------------------------


def test_cagr_de_una_serie_conocida():
    index = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    nav = pd.Series(np.linspace(100, 200, len(index)), index=index)
    # Duplicar en ~5 anos: 2^(1/5) - 1 = 14.9%
    assert metrics.cagr(nav) == pytest.approx(0.1487, abs=0.005)


def test_max_drawdown_de_una_caida_conocida():
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    nav = pd.Series([100.0, 120.0, 60.0, 80.0, 130.0], index=index)
    assert metrics.max_drawdown(nav) == pytest.approx(-0.5)
    detalle = metrics.drawdown_detail(nav)
    assert detalle["peak"] == index[1]
    assert detalle["trough"] == index[2]
    assert detalle["recovery"] == index[4]


def test_sharpe_descuenta_la_tasa_libre_de_riesgo():
    index = pd.bdate_range("2020-01-01", periods=252)
    nav = pd.Series(100 * (1.0003 ** np.arange(len(index))), index=index)
    assert metrics.sharpe(nav, risk_free=0.0) > metrics.sharpe(nav, risk_free=0.05)


def test_benchmark_nav_se_alinea_y_escala():
    sessions = _sessions(periods=50)
    close = _flat_prices(["SPY"], sessions)
    close.loc[close.index[25]:] = 110.0
    nav_index = sessions[10:40]
    bench = benchmark_nav(close, "SPY", nav_index, 100_000.0)
    assert bench.iloc[0] == pytest.approx(100_000.0)
    assert bench.iloc[-1] == pytest.approx(110_000.0)


def test_la_rotacion_anualizada_usa_los_rebalanceos_reales_no_doce(make_cfg):
    """Trimestral: la rotacion anual no puede salir multiplicada por 12."""
    from sfc_tfsig import report as report_mod
    from sfc_tfsig.metrics import evaluate

    cfg = make_cfg(**{"portfolio.n_positions": 10, "portfolio.buffer_rank": 10,
                      "portfolio.max_sector_w": 1.0, "portfolio.max_weight": 0.2,
                      "portfolio.cash_buffer": 0.0})
    sessions = _sessions(start="2020-01-01", periods=800)
    tickers = [f"T{i:02d}" for i in range(20)]
    close = _flat_prices(tickers, sessions)
    trimestres = pd.date_range("2020-03-31", periods=10, freq="QE")
    rows = []
    for j, date in enumerate(trimestres):
        for i, ticker in enumerate(tickers):
            rows.append({"date": date, "ticker": ticker, "sector": "Tech", "price": 100.0,
                         "volatility": 0.2, "score_composite": float((i + 5 * j) % 20)})
    result = run_backtest(pd.DataFrame(rows), close, cfg, progress=False)
    perf = evaluate(result.nav)

    texto = report_mod.costs_section(result, perf)
    esperado = result.rebalances["turnover"].sum() / perf.years
    assert f"{esperado * 100:.2f}%" in texto
    # Y no la version inflada:
    assert f"{result.average_turnover * 12 * 100:.2f}%" not in texto
