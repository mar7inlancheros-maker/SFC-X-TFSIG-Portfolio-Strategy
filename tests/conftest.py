"""Configuracion comun de los tests: `src/` importable y una config de juguete."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sfc_tfsig.config import config_from_dict  # noqa: E402


BASE_CONFIG = {
    "meta": {"name": "test", "version": "0", "base_currency": "USD"},
    "universe": {
        "exchanges": ["NYSE", "Nasdaq"],
        "include_tsx_only": False,
        "min_price": 5.0,
        "min_dollar_volume": 1_000_000,
        "min_market_cap": 100_000_000,
        "min_history_months": 24,
        "max_names": 500,
        "exclude_sic": [6726],
        "max_fundamental_staleness_d": 550,
    },
    "calendar": {"start": "2020-01-31", "end": "", "rebalance": "M", "execution_lag_d": 1},
    "factors": {
        "weights": {"value": 0.25, "quality": 0.25, "momentum": 0.25, "lowvol": 0.25},
        "processing": {
            "winsorize_pct": 0.02,
            "sector_neutral": True,
            "min_sector_names": 5,
            "min_coverage": 0.6,
            "require_fundamental_score": False,
        },
        "value": {"metrics": ["earnings_yield", "fcf_yield"]},
        "quality": {"metrics": ["roic", "debt_to_equity"]},
        "momentum": {"lookback_m": 12, "skip_m": 1},
        "lowvol": {"lookback_d": 252},
    },
    "portfolio": {
        "n_positions": 10,
        "weighting": "equal",
        "max_weight": 0.20,
        "min_weight": 0.01,
        "max_sector_w": 0.40,
        "buffer_rank": 15,
        "cash_buffer": 0.0,
    },
    "costs": {"commission_bps": 5.0, "spread_bps": 10.0, "slippage_bps": 5.0},
    "backtest": {"initial_capital": 100_000.0, "benchmark": "SPY", "risk_free_rate": 0.0},
    "validation": {"train_years": 3, "test_years": 1, "n_quantiles": 5, "newey_west_lag": 6},
    "reporting": {"cadence": "Q", "charts": False, "decimals": 2},
}


@pytest.fixture
def cfg():
    return config_from_dict(BASE_CONFIG)


@pytest.fixture
def make_cfg():
    """Config de juguete con parametros sustituidos: `make_cfg(**{"portfolio.n_positions": 5})`."""
    def _make(**overrides):
        return config_from_dict(BASE_CONFIG).replace(**overrides)
    return _make
