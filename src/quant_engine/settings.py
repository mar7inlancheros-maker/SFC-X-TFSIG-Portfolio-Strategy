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

# Una canasta solo LONG invierte el 100% del capital en la pata larga. No se
# toma `gross_exposure` del YAML: ese 200% es la suma de las dos patas de un
# libro long/short, y aplicado a una sola pata seria apalancamiento 2x.
LONG_ONLY_GROSS = 1.0
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
    positions: tuple[tuple[str, int], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    # -- derivados ---------------------------------------------------------
    @property
    def tickers(self) -> tuple[str, ...]:
        return self.longs + self.shorts

    @property
    def long_only(self) -> bool:
        """Canasta sin SHORT: research solo propone compras."""
        return not self.shorts

    @property
    def exposure(self) -> tuple[float, float]:
        """(bruta, neta) objetivo. Solo LONG: 100% largo; long/short: la del YAML."""
        if self.long_only:
            return LONG_ONLY_GROSS, LONG_ONLY_GROSS
        return float(self.get("portfolio.gross_exposure", 2.0)), float(self.get("portfolio.net_exposure", 0.0))

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


def parse_positions(text: str) -> tuple[tuple[str, int], ...]:
    """'AAPL:500, tsla:-200' -> (('AAPL', 500), ('TSLA', -200)). Negativo = corto.

    Acciones enteras, no pesos: el plan compara acciones con acciones. Un
    ticker repetido es un error, no una suma: casi siempre es un tecleo.
    """
    out: dict[str, int] = {}
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        ticker, sep, qty = item.replace("=", ":").partition(":")
        ticker = ticker.strip().upper()
        if not sep or not ticker:
            raise SettingsError(f"posicion '{item}' sin formato TICKER:ACCIONES (ej. AAPL:500, TSLA:-200)")
        try:
            shares = int(qty.strip().replace("+", "").replace("_", ""))
        except ValueError:
            raise SettingsError(f"'{qty.strip()}' no es un numero entero de acciones en '{item}'") from None
        if ticker in out:
            raise SettingsError(f"{ticker} aparece dos veces en las posiciones: deja una sola entrada")
        if shares:
            out[ticker] = shares
    return tuple(out.items())


def format_positions(positions: tuple[tuple[str, int], ...]) -> str:
    return ",".join(f"{t}:{q}" for t, q in positions)


def load_yaml(path: Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_YAML
    if not path.exists():
        raise SettingsError(f"no existe {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def validate(s: EngineSettings) -> None:
    if not s.longs:
        raise SettingsError("hace falta al menos un ticker LONG (sin SHORT, la canasta es solo LONG)")
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

    cap = float(s.get("portfolio.max_position", 0.2))
    # Con signos fijados, cada lado necesita sitio para su pata: la mitad de la
    # bruta en un long/short, el 100% en una canasta solo LONG.
    for side, names, leg in _legs(s):
        if len(names) * cap < leg - 1e-9:
            raise SettingsError(
                f"{side}: {len(names)} nombres x tope {cap:.0%} = {len(names) * cap:.0%} "
                f"no alcanza para la pata ({leg:.0%}). "
                "Sube portfolio.max_position en el YAML o anade nombres."
            )


def _legs(s: EngineSettings) -> list[tuple[str, tuple[str, ...], float]]:
    gross, net = s.exposure
    if s.long_only:
        return [("LONG", s.longs, gross)]
    return [("LONG", s.longs, (gross + net) / 2), ("SHORT", s.shorts, (gross - net) / 2)]


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

    if isinstance(values.get("positions"), str):
        values["positions"] = parse_positions(values["positions"])
    longs_t = parse_tickers(longs) if isinstance(longs, str) else tuple(longs)
    shorts_t = parse_tickers(shorts) if isinstance(shorts, str) else tuple(shorts)

    # Con pocos nombres por pata el tope no alcanza a invertir la pata: 2 nombres
    # x 35% = 70% < 100%. En vez de rechazar la lista, el tope sube al minimo
    # posible (pata / nombres) y queda anotado. Con esos nombres todos los
    # metodos de construccion convergen en equiponderado, y el reporte lo avisa.
    port = dict(raw.get("portfolio", {}) or {})
    gross = float(port.get("gross_exposure", 2.0))
    cap = float(port.get("max_position", 0.2))
    # Solo LONG: una pata que invierte el 100%; long/short: cada pata, la mitad.
    fewest = min(len(longs_t), len(shorts_t)) if shorts_t else len(longs_t)
    leg = gross / 2 if shorts_t else LONG_ONLY_GROSS
    if fewest and fewest * cap < leg - 1e-9:
        needed = leg / fewest
        raw = {**raw, "portfolio": {**port, "max_position": needed},
               "_notes": [*raw.get("_notes", []),
                          f"tope por nombre subido de {cap:.0%} a {needed:.0%}: solo {fewest} "
                          f"nombre(s) en una pata"]}

    settings = EngineSettings(longs=longs_t, shorts=shorts_t, raw=raw, **values)
    validate(settings)
    return settings


def output_dir(settings: EngineSettings) -> Path:
    return ROOT / str(settings.get("output.directory", "output/quant_engine"))
