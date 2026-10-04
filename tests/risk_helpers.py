"""Datos sinteticos compartidos por los tests de riesgo.

Funciones, no fixtures: cada fichero de test decide el alcance de su fixture.
Se importan como `from risk_helpers import ...`, igual que `conftest.BASE_CONFIG`.
"""

from __future__ import annotations

import tomllib

import numpy as np
import pandas as pd

from sfc_tfsig.config import config_from_dict, risk_config_from_dict
from sfc_tfsig.paths import RISK_CONFIG

SECTORS = ["Technology", "Financials", "Energy", "HealthCare", "Industrials", "Utilities"]


def light_risk_config(**limit_overrides):
    """La politica del repo con menos trayectorias: los tests miden la tuberia,
    no la precision de los percentiles."""
    with RISK_CONFIG.open("rb") as fh:
        data = tomllib.load(fh)
    data["montecarlo"]["bootstrap"]["n_paths"] = 500
    data["montecarlo"]["parametric"]["n_sims"] = 5000
    data["var"]["garch"]["n_paths"] = 2000
    data["limits"].update(limit_overrides)
    return risk_config_from_dict(data)


def synthetic_book(n_names: int = 12, n_days: int = 1400, seed: int = 0) -> dict:
    """Mercado con un factor comun de colas gordas y una cartera equiponderada."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_days)
    market = rng.standard_t(4, n_days) * 0.007
    tickers = [f"T{i}" for i in range(n_names)]
    rets = pd.DataFrame(
        {t: 0.0004 + (0.7 + 0.05 * i) * market + rng.normal(0, 0.012, n_days)
         for i, t in enumerate(tickers)},
        index=dates,
    )
    rets["SPY"] = market + 0.0003
    close = (1 + rets).cumprod() * 50.0
    third = n_names // 3
    sectors = pd.Series(["Technology"] * third + ["Financials"] * third
                        + ["Energy"] * (n_names - 2 * third), index=tickers)
    return {
        "close": close,
        "volume": pd.DataFrame(2e6, index=dates, columns=close.columns),
        "weights": pd.Series(0.98 / n_names, index=tickers),
        "sectors": sectors,
        "nav": (1 + rets[tickers].mean(axis=1)).cumprod() * 100_000.0,
        "bench": close["SPY"] / close["SPY"].iloc[0] * 100_000.0,
        "scores": pd.DataFrame({"score_momentum": rng.normal(1, 0.5, n_names),
                                "score_quality": rng.normal(0.3, 0.5, n_names)}, index=tickers),
    }


def synthetic_strategy(base_config: dict, n_names: int = 60, seed: int = 7):
    """Panel puntuado, precios y volumen para correr la estrategia de verdad.

    Suficientes nombres (60) para que el modelo factorial tenga seccion cruzada
    y cuatro anos para que el GARCH y la calibracion tengan historia.
    """
    from sfc_tfsig.data import prices as prices_mod
    from sfc_tfsig.factors.composite import build_scores
    from sfc_tfsig.financials import compute_all

    cfg = config_from_dict(base_config).replace(**{"calendar.start": "2020-01-31"})
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2019-01-01", "2023-12-29")
    tickers = [f"N{i:02d}" for i in range(n_names)]
    quality = rng.normal(size=n_names)
    market = rng.normal(0.0003, 0.01, len(sessions))
    shocks = rng.normal(0, 0.015, (len(sessions), n_names))
    log_prices = np.cumsum(market[:, None] + 0.0002 * quality[None, :] + shocks, axis=0)
    close = pd.DataFrame(np.exp(log_prices) * 50.0, index=sessions, columns=tickers)
    close["SPY"] = np.exp(np.cumsum(market)) * 100.0

    rows = []
    for date in pd.date_range("2019-12-31", "2023-11-30", freq="ME"):
        price = close.loc[close.index <= date, tickers].iloc[-1]
        for i, ticker in enumerate(tickers):
            income = 10.0 * (1.0 + 0.5 * quality[i]) * 1e6
            rows.append({
                "date": date, "ticker": ticker, "sector": SECTORS[i % len(SECTORS)],
                "country": "US", "price": float(price[ticker]),
                "market_cap": float(price[ticker]) * 1e6 * (1 + i % 7),
                "net_income": income, "revenue": 1e8, "equity": 8e7, "assets": 1.5e8,
                "ocf": income * 1.1, "capex": 3e6, "operating_income": income * 1.3,
                "cash": 1e7, "debt_long": 3e7, "current_assets": 6e7, "current_liabilities": 3e7,
                "interest_expense": 1.5e6, "tax_expense": income * 0.25, "pretax_income": income,
                "cogs": 6e7, "shares": 1e6, "shares_growth": 0.0,
            })
    raw = pd.DataFrame(rows)
    for name, table in (("momentum", prices_mod.momentum(close[tickers])),
                        ("volatility", prices_mod.realized_volatility(close[tickers]))):
        stacked = table.stack(future_stack=True).rename(name).reset_index()
        stacked.columns = ["date", "ticker", name]
        stacked["_m"] = stacked["date"].dt.to_period("M")
        raw["_m"] = raw["date"].dt.to_period("M")
        raw = raw.merge(stacked[["ticker", "_m", name]], on=["ticker", "_m"], how="left").drop(columns="_m")
    scored = build_scores(compute_all(raw), cfg)
    volume = pd.DataFrame(5e6, index=close.index, columns=close.columns)
    return cfg, scored, close, volume
