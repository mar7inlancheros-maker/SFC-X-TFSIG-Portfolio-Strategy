"""Cadena completa sobre datos sinteticos: panel -> scores -> cartera ->
backtest -> metricas -> reporte -> ordenes.

Los tests de cada modulo comprueban que las piezas funcionan. Este comprueba que
encajan: que las columnas que produce uno son las que espera el siguiente, que
los nombres no se desincronizan, y que el reporte se genera sin tocar la red.

El mercado sintetico esta construido con una relacion real entre fundamentales y
retorno futuro, asi que ademas sirve de comprobacion de cordura: si el motor
esta bien conectado, el modelo deberia batir al indice equiponderado en un
mundo donde la senal SI existe. Si no lo bate ahi, no lo va a batir en el real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfc_tfsig import panel as panel_mod, report as report_mod, validation as validation_mod
from sfc_tfsig.backtest import benchmark_nav, run_backtest
from sfc_tfsig.factors.composite import build_scores
from sfc_tfsig.metrics import evaluate
from sfc_tfsig.orders import build_orders, render_orders
from sfc_tfsig.portfolio import build_portfolio

N_NAMES = 80
SECTORES = ["Tech", "Financials", "Energy", "HealthCare", "Industrials",
            "ConsumerStaples", "Utilities", "Materials"]


@pytest.fixture(scope="module")
def mercado():
    """Mercado sintetico de 80 acciones y 5 anos, con senal real y ruido.

    La calidad de cada empresa (fija) y su valoracion (variable) influyen en la
    deriva de su precio. El ruido es diez veces mayor que la senal, como en el
    mercado de verdad: si el motor solo funciona con senal limpia, no funciona.
    """
    rng = np.random.default_rng(20260920)
    sessions = pd.bdate_range("2019-01-01", "2024-12-31")
    tickers = [f"T{i:02d}" for i in range(N_NAMES)]

    # Calidad intrinseca: constante por empresa, conocida por el modelo via ROIC.
    calidad = rng.normal(size=N_NAMES)
    deriva_diaria = 0.0002 + 0.00025 * calidad

    precios = np.zeros((len(sessions), N_NAMES))
    precios[0] = rng.uniform(20, 200, N_NAMES)
    shocks = rng.normal(0.0, 0.018, size=(len(sessions), N_NAMES))
    for t in range(1, len(sessions)):
        precios[t] = precios[t - 1] * np.exp(deriva_diaria + shocks[t])

    close = pd.DataFrame(precios, index=sessions, columns=tickers)
    # Un benchmark equiponderado del propio universo: la referencia honesta.
    close["INDICE"] = close[tickers].div(close[tickers].iloc[0]).mean(axis=1) * 100.0

    fechas = pd.date_range("2019-12-31", "2024-11-30", freq="ME")
    fechas = pd.DatetimeIndex([f for f in fechas if (sessions <= f).any()])

    filas = []
    for fecha in fechas:
        precio_actual = close.loc[close.index <= fecha, tickers].iloc[-1]
        for i, ticker in enumerate(tickers):
            # Beneficio proporcional a la calidad; el precio fluctua con el
            # ruido, asi que el rendimiento por beneficio varia en el tiempo.
            beneficio = 10.0 * (1.0 + 0.5 * calidad[i])
            filas.append({
                "date": fecha,
                "ticker": ticker,
                "sector": SECTORES[i % len(SECTORES)],
                "country": "US" if i % 5 else "CA",
                "price": float(precio_actual[ticker]),
                "market_cap": float(precio_actual[ticker]) * 1_000_000.0,
                "net_income": beneficio * 1_000_000.0,
                "revenue": 100.0 * 1_000_000.0,
                "equity": 80.0 * 1_000_000.0,
                "assets": 150.0 * 1_000_000.0,
                "ocf": beneficio * 1.1 * 1_000_000.0,
                "capex": 3.0 * 1_000_000.0,
                "operating_income": beneficio * 1.3 * 1_000_000.0,
                "cash": 10.0 * 1_000_000.0,
                "debt_long": 30.0 * 1_000_000.0,
                "current_assets": 60.0 * 1_000_000.0,
                "current_liabilities": 30.0 * 1_000_000.0,
                "interest_expense": 1.5 * 1_000_000.0,
                "tax_expense": beneficio * 0.25 * 1_000_000.0,
                "pretax_income": beneficio * 1_000_000.0,
                "cogs": 60.0 * 1_000_000.0,
                "shares": 1_000_000.0,
                "shares_growth": 0.0,
            })

    crudo = pd.DataFrame(filas)
    crudo["momentum"] = np.nan
    crudo["volatility"] = np.nan

    momentum = __import__("sfc_tfsig.data.prices", fromlist=["x"]).momentum(close[tickers])
    volatilidad = __import__("sfc_tfsig.data.prices", fromlist=["x"]).realized_volatility(close[tickers])
    for nombre, tabla in (("momentum", momentum), ("volatility", volatilidad)):
        stacked = tabla.stack(future_stack=True).rename(nombre).reset_index()
        stacked.columns = ["date", "ticker", nombre]
        stacked["_m"] = stacked["date"].dt.to_period("M")
        crudo["_m"] = crudo["date"].dt.to_period("M")
        crudo = crudo.drop(columns=[nombre]).merge(
            stacked[["ticker", "_m", nombre]], on=["ticker", "_m"], how="left"
        ).drop(columns="_m")

    from sfc_tfsig.financials import compute_all
    return compute_all(crudo), close


@pytest.fixture(scope="module")
def cfg_integracion():
    from conftest import BASE_CONFIG
    from sfc_tfsig.config import config_from_dict
    return config_from_dict(BASE_CONFIG).replace(**{
        "calendar.start": "2019-12-31",
        "portfolio.n_positions": 20,
        "portfolio.buffer_rank": 30,
        "portfolio.max_weight": 0.10,
        "portfolio.max_sector_w": 0.30,
        "portfolio.cash_buffer": 0.02,
        "backtest.benchmark": "INDICE",
        "validation.train_years": 3,
    })


def test_la_cadena_completa_corre_sin_red(mercado, cfg_integracion):
    crudo, close = mercado
    cfg = cfg_integracion

    puntuado = build_scores(crudo, cfg)
    assert "score_composite" in puntuado.columns
    assert puntuado["score_composite"].notna().mean() > 0.9

    resultado = run_backtest(puntuado, close, cfg, progress=False)
    assert len(resultado.nav) > 1000
    assert not resultado.holdings.empty
    assert not resultado.trades.empty

    indice = benchmark_nav(close, "INDICE", resultado.nav.index,
                           float(cfg.get("backtest.initial_capital")))
    perf = evaluate(resultado.nav, indice)
    assert not pd.isna(perf.cagr)
    assert not pd.isna(perf.sharpe)
    assert not pd.isna(perf.information_ratio)


def test_en_un_mundo_con_senal_el_modelo_bate_al_indice(mercado, cfg_integracion):
    """Comprobacion de cordura del cableado, no promesa de rendimiento.

    El mercado sintetico esta construido para que la calidad prediga el retorno.
    Si el motor esta bien conectado, tiene que encontrarlo. Que lo encuentre
    AQUI no dice nada sobre si existe senal en el mercado real -- eso lo decide
    la validacion fuera de muestra sobre datos de verdad.
    """
    crudo, close = mercado
    cfg = cfg_integracion
    puntuado = build_scores(crudo, cfg)
    resultado = run_backtest(puntuado, close, cfg, progress=False)
    indice = benchmark_nav(close, "INDICE", resultado.nav.index,
                           float(cfg.get("backtest.initial_capital")))
    perf = evaluate(resultado.nav, indice)
    assert perf.excess_cagr > 0.0


def test_las_restricciones_de_riesgo_se_respetan_en_todas_las_fechas(mercado, cfg_integracion):
    crudo, close = mercado
    cfg = cfg_integracion
    puntuado = build_scores(crudo, cfg)
    resultado = run_backtest(puntuado, close, cfg, progress=False)

    por_fecha_sector = resultado.holdings.groupby(["date", "sector"])["weight"].sum()
    # Margen sobre el tope: entre rebalanceos los pesos derivan con los precios,
    # pero EN el rebalanceo el tope se cumple.
    assert por_fecha_sector.max() <= float(cfg.get("portfolio.max_sector_w")) + 0.02
    assert resultado.holdings["weight"].max() <= float(cfg.get("portfolio.max_weight")) + 0.01


def test_la_validacion_produce_las_tablas_del_reporte(mercado, cfg_integracion):
    crudo, close = mercado
    cfg = cfg_integracion
    puntuado = build_scores(crudo, cfg)
    con_futuro = panel_mod.add_forward_returns(puntuado, close)

    informe = validation_mod.full_report(con_futuro, cfg)
    assert informe["ic_summary"]["n"] > 24
    assert not informe["quantiles"].empty
    assert not informe["walk_forward"].empty
    assert (informe["walk_forward"]["test_months"] > 0).all()


def test_el_reporte_se_genera_entero_y_lleva_las_limitaciones(mercado, cfg_integracion):
    crudo, close = mercado
    cfg = cfg_integracion
    puntuado = build_scores(crudo, cfg)
    resultado = run_backtest(puntuado, close, cfg, progress=False)
    indice = benchmark_nav(close, "INDICE", resultado.nav.index,
                           float(cfg.get("backtest.initial_capital")))
    perf = evaluate(resultado.nav, indice)
    con_futuro = panel_mod.add_forward_returns(puntuado, close)
    informe = validation_mod.full_report(con_futuro, cfg)

    texto = report_mod.build_report(resultado, perf, cfg, benchmark_nav=indice,
                                    validation=informe)

    for seccion in ("## Rendimiento", "## Costes y rotacion", "## Cartera al",
                    "## Validacion estadistica", "## Limitaciones declaradas"):
        assert seccion in texto
    assert cfg.fingerprint in texto
    assert "Sesgo de supervivencia" in texto
    # El reporte va a fichero redirigido: tiene que ser ASCII puro.
    texto.encode("cp1252")


def test_las_ordenes_del_ultimo_mes_cuadran_con_la_cartera(mercado, cfg_integracion):
    crudo, close = mercado
    cfg = cfg_integracion
    puntuado = build_scores(crudo, cfg)

    ultima = puntuado["date"].max()
    seccion = puntuado[puntuado["date"] == ultima]
    objetivo = build_portfolio(seccion, cfg)
    ordenes = build_orders(objetivo, pd.Series(dtype="float64"), 200_000.0, cfg)

    assert len(objetivo) == int(cfg.get("portfolio.n_positions"))
    assert set(ordenes["side"]) == {"BUY"}
    invertido = ordenes["notional"].sum()
    esperado = 200_000.0 * (1.0 - float(cfg.get("portfolio.cash_buffer")))
    assert invertido == pytest.approx(esperado, rel=0.02)

    texto = render_orders(ordenes, objetivo, 200_000.0, cfg, ultima)
    assert "Rebalanceo propuesto" in texto
    texto.encode("cp1252")
