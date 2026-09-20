"""Carga, valida y huella digital de `config/strategy.toml`.

Tres responsabilidades, en este orden de importancia:

1. **Validar.** Un typo en el TOML no puede convertirse en una cartera rara que
   nadie sabe explicar en el comite. Se comprueba antes de tocar datos: tipos,
   rangos, y las relaciones entre parametros que el codigo da por ciertas mas
   abajo (por ejemplo `buffer_rank >= n_positions`, o que los pesos de factores
   no sumen cero).
2. **Fingerprint.** sha256 del contenido normalizado. Va estampado en cada
   backtest y cada reporte: dos resultados con el mismo fingerprint son
   comparables, con fingerprint distinto no lo son, y eso zanja la discusion de
   "pero a mi me daba otra cosa".
3. **Acceso.** Lectura por ruta con puntos, `cfg.get("portfolio.n_positions")`,
   con error claro si la clave no existe.

La configuracion es inmutable una vez cargada. Si un modulo necesita variar un
parametro (barridos de sensibilidad, walk-forward), usa `cfg.replace(...)`, que
devuelve una copia nueva con su propio fingerprint.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .paths import DEFAULT_CONFIG

_MISSING = object()


class ConfigError(ValueError):
    """El TOML existe pero dice algo que el motor no puede ejecutar."""


@dataclass(frozen=True)
class Config:
    """Configuracion validada e inmutable."""

    data: Mapping[str, Any]
    source: Path | None
    fingerprint: str

    # -- acceso ------------------------------------------------------------
    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """`cfg.get("factors.weights.value")`."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is _MISSING:
                    raise ConfigError(f"falta la clave '{dotted}' en la configuracion")
                return default
            node = node[part]
        return node

    def section(self, name: str) -> Mapping[str, Any]:
        node = self.get(name)
        if not isinstance(node, Mapping):
            raise ConfigError(f"'{name}' no es una seccion")
        return node

    def __getitem__(self, dotted: str) -> Any:
        return self.get(dotted)

    # -- derivados ---------------------------------------------------------
    def replace(self, **overrides: Any) -> "Config":
        """Copia con parametros sustituidos, por ruta con puntos.

        `cfg.replace(**{"portfolio__n_positions": 50})` no; se usa un dict:
        `cfg.replace(**{"portfolio.n_positions": 50})`. El fingerprint cambia,
        que es justo el punto: un barrido de sensibilidad no debe poder
        confundirse con la configuracion oficial.
        """
        data = copy.deepcopy(dict(self.data))
        for dotted, value in overrides.items():
            parts = dotted.split(".")
            node = data
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        validate(data)
        return Config(data=data, source=self.source, fingerprint=_fingerprint(data))

    @property
    def factor_weights(self) -> dict[str, float]:
        """Pesos normalizados a 1, descartando los apagados (peso 0)."""
        raw = {k: float(v) for k, v in self.get("factors.weights").items()}
        active = {k: v for k, v in raw.items() if v > 0}
        total = sum(active.values())
        return {k: v / total for k, v in active.items()}

    @property
    def total_cost_bps(self) -> float:
        """Coste de un lado (compra o venta) en puntos basicos."""
        c = self.section("costs")
        return (
            float(c["commission_bps"])
            + float(c["spread_bps"])
            + float(c["slippage_bps"])
        )


# ---------------------------------------------------------------------------
#  Validacion
# ---------------------------------------------------------------------------

_REQUIRED_SECTIONS = (
    "meta",
    "universe",
    "calendar",
    "factors",
    "portfolio",
    "costs",
    "backtest",
    "validation",
    "reporting",
)

_KNOWN_FACTORS = ("value", "quality", "momentum", "lowvol")
_WEIGHTING_SCHEMES = ("equal", "score_tilt", "inverse_vol")


def _need(node: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in node:
        raise ConfigError(f"falta '{key}' en [{where}]")
    return node[key]


def _positive(value: Any, name: str) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"'{name}' debe ser numerico, llego {value!r}") from None
    if num <= 0:
        raise ConfigError(f"'{name}' debe ser > 0, llego {num}")
    return num


def _fraction(value: Any, name: str, *, allow_zero: bool = True) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"'{name}' debe ser numerico, llego {value!r}") from None
    lo = 0.0 if allow_zero else 1e-12
    if not lo <= num <= 1.0:
        raise ConfigError(f"'{name}' debe estar en [0, 1], llego {num}")
    return num


def validate(data: Mapping[str, Any]) -> None:
    """Lanza `ConfigError` con un mensaje accionable, o no hace nada."""
    for section in _REQUIRED_SECTIONS:
        if section not in data:
            raise ConfigError(f"falta la seccion [{section}] en la configuracion")

    # -- universo ---------------------------------------------------------
    uni = data["universe"]
    exchanges = _need(uni, "exchanges", "universe")
    if not isinstance(exchanges, list) or not exchanges:
        raise ConfigError("[universe].exchanges debe ser una lista no vacia")
    _positive(_need(uni, "min_price", "universe"), "universe.min_price")
    _positive(_need(uni, "min_dollar_volume", "universe"), "universe.min_dollar_volume")
    _positive(_need(uni, "min_market_cap", "universe"), "universe.min_market_cap")
    if int(_need(uni, "min_history_months", "universe")) < 13:
        raise ConfigError(
            "[universe].min_history_months < 13: no alcanza para momentum 12-1, "
            "el factor saldria vacio en vez de fallar"
        )
    if int(_need(uni, "max_names", "universe")) < 50:
        raise ConfigError("[universe].max_names < 50: el panel no da para seccion cruzada")

    # -- calendario -------------------------------------------------------
    cal = data["calendar"]
    if _need(cal, "rebalance", "calendar") not in ("M", "Q"):
        raise ConfigError("[calendar].rebalance debe ser 'M' o 'Q'")
    if int(_need(cal, "execution_lag_d", "calendar")) < 0:
        raise ConfigError(
            "[calendar].execution_lag_d negativo: ejecutar ANTES de la senal es "
            "look-ahead puro"
        )

    # -- factores ---------------------------------------------------------
    fac = data["factors"]
    weights = _need(fac, "weights", "factors")
    unknown = set(weights) - set(_KNOWN_FACTORS)
    if unknown:
        raise ConfigError(
            f"factores desconocidos en [factors.weights]: {sorted(unknown)}. "
            f"Conocidos: {list(_KNOWN_FACTORS)}"
        )
    for name, weight in weights.items():
        if float(weight) < 0:
            raise ConfigError(
                f"[factors.weights].{name} negativo ({weight}): para invertir un "
                "factor se cambia la metrica, no el peso"
            )
    if sum(float(w) for w in weights.values()) <= 0:
        raise ConfigError("todos los pesos de factores son cero: no hay modelo")

    proc = _need(fac, "processing", "factors")
    _fraction(_need(proc, "winsorize_pct", "factors.processing"), "winsorize_pct")
    if float(proc["winsorize_pct"]) >= 0.5:
        raise ConfigError("winsorize_pct >= 0.5 recortaria toda la distribucion")
    _fraction(_need(proc, "min_coverage", "factors.processing"), "min_coverage")
    if int(_need(proc, "min_sector_names", "factors.processing")) < 2:
        raise ConfigError("min_sector_names < 2: no se puede estandarizar con un solo nombre")

    mom = _need(fac, "momentum", "factors")
    if int(mom["lookback_m"]) <= int(mom["skip_m"]):
        raise ConfigError("[factors.momentum]: lookback_m debe ser mayor que skip_m")
    if int(_need(fac["lowvol"], "lookback_d", "factors.lowvol")) < 60:
        raise ConfigError("[factors.lowvol].lookback_d < 60: la vol seria ruido")

    # -- cartera ----------------------------------------------------------
    pf = data["portfolio"]
    n = int(_need(pf, "n_positions", "portfolio"))
    if n < 10:
        raise ConfigError(
            f"[portfolio].n_positions = {n}: con menos de 10 nombres el riesgo "
            "idiosincratico domina y el backtest deja de medir los factores"
        )
    scheme = _need(pf, "weighting", "portfolio")
    if scheme not in _WEIGHTING_SCHEMES:
        raise ConfigError(f"[portfolio].weighting debe ser uno de {list(_WEIGHTING_SCHEMES)}")
    max_w = _fraction(_need(pf, "max_weight", "portfolio"), "max_weight", allow_zero=False)
    min_w = _fraction(_need(pf, "min_weight", "portfolio"), "min_weight")
    if min_w >= max_w:
        raise ConfigError("[portfolio]: min_weight debe ser menor que max_weight")
    if max_w * n < 1.0:
        raise ConfigError(
            f"[portfolio]: max_weight ({max_w}) x n_positions ({n}) = {max_w * n:.2f} < 1. "
            "Los topes impiden invertir el 100% del capital"
        )
    if int(_need(pf, "buffer_rank", "portfolio")) < n:
        raise ConfigError("[portfolio].buffer_rank debe ser >= n_positions")
    _fraction(_need(pf, "max_sector_w", "portfolio"), "max_sector_w", allow_zero=False)
    _fraction(_need(pf, "cash_buffer", "portfolio"), "cash_buffer")

    # -- costes -----------------------------------------------------------
    for key in ("commission_bps", "spread_bps", "slippage_bps"):
        value = float(_need(data["costs"], key, "costs"))
        if value < 0:
            raise ConfigError(f"[costs].{key} negativo: un coste no paga dividendos")

    # -- backtest ---------------------------------------------------------
    bt = data["backtest"]
    _positive(_need(bt, "initial_capital", "backtest"), "backtest.initial_capital")
    if not str(_need(bt, "benchmark", "backtest")).strip():
        raise ConfigError("[backtest].benchmark vacio: sin referencia no hay alfa que medir")

    # -- validacion -------------------------------------------------------
    val = data["validation"]
    if _positive(_need(val, "train_years", "validation"), "train_years") < 2:
        raise ConfigError("[validation].train_years < 2: ventana demasiado corta para calibrar")
    _positive(_need(val, "test_years", "validation"), "test_years")
    if int(_need(val, "n_quantiles", "validation")) < 3:
        raise ConfigError("[validation].n_quantiles < 3: no hay spread que mirar")
    if int(_need(val, "newey_west_lag", "validation")) < 0:
        raise ConfigError("[validation].newey_west_lag no puede ser negativo")


# ---------------------------------------------------------------------------
#  Carga
# ---------------------------------------------------------------------------


def _fingerprint(data: Mapping[str, Any]) -> str:
    """sha256 del contenido normalizado (claves ordenadas, sin espacios)."""
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path | None = None) -> Config:
    """Lee el TOML, lo valida y devuelve una `Config` inmutable."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise ConfigError(
            f"no existe {cfg_path}. El repo trae 'config/strategy.toml'; "
            "si lo renombraste, pasa la ruta con --config"
        )
    with cfg_path.open("rb") as fh:
        data = tomllib.load(fh)
    validate(data)
    return Config(data=data, source=cfg_path, fingerprint=_fingerprint(data))


def config_from_dict(data: Mapping[str, Any]) -> Config:
    """Config en memoria, para tests y barridos. Valida igual que la del disco."""
    payload = copy.deepcopy(dict(data))
    validate(payload)
    return Config(data=payload, source=None, fingerprint=_fingerprint(payload))
