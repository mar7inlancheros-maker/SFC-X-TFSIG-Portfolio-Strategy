"""Configuracion del motor: YAML de defaults + lo que el usuario escribe en la terminal.

Separado de `sfc_tfsig.config` a proposito. El modelo multifactor y este motor
responden preguntas distintas -- "que 30 acciones comprar" frente a "se sostiene
esta lista de research" -- y compartir configuracion haria que tocar un default
de aqui cambiara el fingerprint del modelo de alla.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from sfc_tfsig.paths import CONFIG_DIR, ROOT

DEFAULT_YAML = CONFIG_DIR / "quant_engine.yaml"

CONSTRUCTIONS = ("equal_weight", "inverse_vol", "risk_parity", "min_variance", "max_sharpe")
REBALANCES = ("monthly", "quarterly")
LOOKBACK_YEARS = {"1y": 1, "2y": 2, "3y": 3, "5y": 5, "10y": 10}

TRADING_DAYS = 252


class SettingsError(ValueError):
    """Un parametro que el motor no puede ejecutar."""


@dataclass(frozen=True)
class EngineSettings:
    longs: tuple[str, ...]
    shorts: tuple[str, ...]
    benchmark: str
    lookback: str
    risk_free_rate: float
    rebalance: str
    construction: str
    initial_capital: float
    transaction_cost: float
    borrow_cost: float
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    # -- derivados ---------------------------------------------------------
    @property
    def tickers(self) -> tuple[str, ...]:
        return self.longs + self.shorts

    @property
    def lookback_years(self) -> int:
        return LOOKBACK_YEARS[self.lookback]

    @property
    def signs(self) -> dict[str, int]:
        """+1 largo, -1 corto. El signo lo decide el research, nunca el motor."""
        out = {t: 1 for t in self.longs}
        out.update({t: -1 for t in self.shorts})
        return out

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def fingerprint(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "raw"}
        payload["raw"] = self.raw
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def with_changes(self, **changes: Any) -> "EngineSettings":
        updated = replace(self, **changes)
        validate(updated)
        return updated


def parse_tickers(text: str) -> tuple[str, ...]:
    """'aapl, msft ,NVDA' -> ('AAPL', 'MSFT', 'NVDA'), sin duplicados, en orden."""
    items = [t.strip().upper() for t in str(text).replace(";", ",").split(",")]
    return tuple(dict.fromkeys(t for t in items if t))


def load_yaml(path: Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_YAML
    if not path.exists():
        raise SettingsError(f"no existe {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def validate(s: EngineSettings) -> None:
    if not s.longs or not s.shorts:
        raise SettingsError("hace falta al menos un ticker LONG y uno SHORT")
    overlap = set(s.longs) & set(s.shorts)
    if overlap:
        raise SettingsError(f"tickers en LONG y SHORT a la vez: {sorted(overlap)}")
    if s.benchmark in s.tickers:
        raise SettingsError(f"el benchmark {s.benchmark} no puede estar en la cartera")
    if s.lookback not in LOOKBACK_YEARS:
        raise SettingsError(f"periodo '{s.lookback}' no valido. Opciones: {list(LOOKBACK_YEARS)}")
    if s.rebalance not in REBALANCES:
        raise SettingsError(f"rebalanceo '{s.rebalance}' no valido. Opciones: {list(REBALANCES)}")
    if s.construction not in CONSTRUCTIONS:
        raise SettingsError(f"construccion '{s.construction}' no valida. Opciones: {list(CONSTRUCTIONS)}")
    if s.initial_capital <= 0:
        raise SettingsError("el capital inicial debe ser positivo")
    if not 0 <= s.transaction_cost < 0.05:
        raise SettingsError("coste de transaccion fuera de [0, 5%): revisa las unidades (0.0005 = 0.05%)")
    if not -0.05 < s.risk_free_rate < 0.25:
        raise SettingsError("tasa libre de riesgo fuera de rango: revisa las unidades (0.04 = 4%)")
    if not 0 <= s.borrow_cost < 0.5:
        raise SettingsError("coste de prestamo fuera de rango: revisa las unidades (0.0025 = 0.25%)")

    gross = float(s.get("portfolio.gross_exposure", 2.0))
    cap = float(s.get("portfolio.max_position", 0.2))
    # Con signos fijados, cada lado necesita sitio para su mitad de la bruta.
    for side, names in (("LONG", s.longs), ("SHORT", s.shorts)):
        if len(names) * cap < gross / 2 - 1e-9:
            raise SettingsError(
                f"{side}: {len(names)} nombres x tope {cap:.0%} = {len(names) * cap:.0%} "
                f"no alcanza para la mitad de la exposicion bruta ({gross / 2:.0%}). "
                "Sube portfolio.max_position en el YAML o anade nombres."
            )


def default_values(yaml_path: Path | None = None) -> dict[str, Any]:
    """Defaults del YAML, sin validar: para mostrarlos antes de pedir tickers."""
    d = load_yaml(yaml_path).get("defaults", {})
    return {
        "benchmark": str(d.get("benchmark", "SPY")).upper(),
        "lookback": str(d.get("lookback", "5y")),
        "risk_free_rate": float(d.get("risk_free_rate", 0.04)),
        "rebalance": str(d.get("rebalance", "monthly")),
        "construction": str(d.get("construction", "risk_parity")),
        "initial_capital": float(d.get("initial_capital", 1_000_000)),
        "transaction_cost": float(d.get("transaction_cost", 0.0005)),
        "borrow_cost": float(d.get("borrow_cost", 0.0025)),
    }


def build_settings(
    longs: str | tuple[str, ...],
    shorts: str | tuple[str, ...],
    *,
    yaml_path: Path | None = None,
    **overrides: Any,
) -> EngineSettings:
    raw = load_yaml(yaml_path)
    values = default_values(yaml_path)
    for key, value in overrides.items():
        if value is None or value == "":
            continue
        values[key] = value
    values["benchmark"] = str(values["benchmark"]).upper()

    settings = EngineSettings(
        longs=parse_tickers(longs) if isinstance(longs, str) else tuple(longs),
        shorts=parse_tickers(shorts) if isinstance(shorts, str) else tuple(shorts),
        raw=raw,
        **values,
    )
    validate(settings)
    return settings


def output_dir(settings: EngineSettings) -> Path:
    return ROOT / str(settings.get("output.directory", "output/quant_engine"))
