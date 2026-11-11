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
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .paths import DEFAULT_CONFIG, RISK_CONFIG

_MISSING = object()


class ConfigError(ValueError):
    """El TOML existe pero dice algo que el motor no puede ejecutar."""


@dataclass(frozen=True)
class Config:
    """Configuracion validada e inmutable."""

    data: Mapping[str, Any]
    source: Path | None
    fingerprint: str
    # "strategy" (config/strategy.toml) o "risk" (config/risk.toml). Decide que
    # validador corre en `replace`: la politica de riesgo no tiene [portfolio]
    # ni [factors], y validarla como estrategia fallaria siempre.
    kind: str = "strategy"

    # -- acceso ------------------------------------------------------------
    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """`cfg.get("factors.weights.value")`."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is _MISSING:
                    raise ConfigError(f"missing key '{dotted}' in the configuration")
                return default
            node = node[part]
        return node

    def section(self, name: str) -> Mapping[str, Any]:
        node = self.get(name)
        if not isinstance(node, Mapping):
            raise ConfigError(f"'{name}' is not a section")
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
        (validate_risk if self.kind == "risk" else validate)(data)
        return Config(data=data, source=self.source, fingerprint=_fingerprint(data),
                      kind=self.kind)

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
        raise ConfigError(f"missing '{key}' in [{where}]")
    return node[key]


def _positive(value: Any, name: str) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"'{name}' must be numeric, got {value!r}") from None
    if num <= 0:
        raise ConfigError(f"'{name}' must be > 0, got {num}")
    return num


def _fraction(value: Any, name: str, *, allow_zero: bool = True) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"'{name}' must be numeric, got {value!r}") from None
    lo = 0.0 if allow_zero else 1e-12
    if not lo <= num <= 1.0:
        raise ConfigError(f"'{name}' must be in [0, 1], got {num}")
    return num


def validate(data: Mapping[str, Any]) -> None:
    """Lanza `ConfigError` con un mensaje accionable, o no hace nada."""
    for section in _REQUIRED_SECTIONS:
        if section not in data:
            raise ConfigError(f"missing section [{section}] in the configuration")

    # -- universo ---------------------------------------------------------
    uni = data["universe"]
    exchanges = _need(uni, "exchanges", "universe")
    if not isinstance(exchanges, list) or not exchanges:
        raise ConfigError("[universe].exchanges must be a non-empty list")
    _positive(_need(uni, "min_price", "universe"), "universe.min_price")
    _positive(_need(uni, "min_dollar_volume", "universe"), "universe.min_dollar_volume")
    _positive(_need(uni, "min_market_cap", "universe"), "universe.min_market_cap")
    if int(_need(uni, "min_history_months", "universe")) < 13:
        raise ConfigError(
            "[universe].min_history_months < 13: not enough for momentum 12-1, "
            "the factor would come out empty instead of failing"
        )
    if int(_need(uni, "max_names", "universe")) < 50:
        raise ConfigError("[universe].max_names < 50: the panel is too small for a cross-section")

    # -- calendario -------------------------------------------------------
    cal = data["calendar"]
    if _need(cal, "rebalance", "calendar") not in ("M", "Q"):
        raise ConfigError("[calendar].rebalance must be 'M' or 'Q'")
    if int(_need(cal, "execution_lag_d", "calendar")) < 0:
        raise ConfigError(
            "[calendar].execution_lag_d negative: executing BEFORE the signal is "
            "pure look-ahead"
        )

    # -- factores ---------------------------------------------------------
    fac = data["factors"]
    weights = _need(fac, "weights", "factors")
    unknown = set(weights) - set(_KNOWN_FACTORS)
    if unknown:
        raise ConfigError(
            f"unknown factors in [factors.weights]: {sorted(unknown)}. "
            f"Known: {list(_KNOWN_FACTORS)}"
        )
    for name, weight in weights.items():
        if float(weight) < 0:
            raise ConfigError(
                f"[factors.weights].{name} negative ({weight}): to invert a "
                "factor, change the metric, not the weight"
            )
    if sum(float(w) for w in weights.values()) <= 0:
        raise ConfigError("all factor weights are zero: there is no model")

    proc = _need(fac, "processing", "factors")
    _fraction(_need(proc, "winsorize_pct", "factors.processing"), "winsorize_pct")
    if float(proc["winsorize_pct"]) >= 0.5:
        raise ConfigError("winsorize_pct >= 0.5 would clip the whole distribution")
    _fraction(_need(proc, "min_coverage", "factors.processing"), "min_coverage")
    if int(_need(proc, "min_sector_names", "factors.processing")) < 2:
        raise ConfigError("min_sector_names < 2: a single name cannot be standardized")

    mom = _need(fac, "momentum", "factors")
    if int(mom["lookback_m"]) <= int(mom["skip_m"]):
        raise ConfigError("[factors.momentum]: lookback_m must be greater than skip_m")
    if int(_need(fac["lowvol"], "lookback_d", "factors.lowvol")) < 60:
        raise ConfigError("[factors.lowvol].lookback_d < 60: the vol would be noise")

    # -- cartera ----------------------------------------------------------
    pf = data["portfolio"]
    n = int(_need(pf, "n_positions", "portfolio"))
    if n < 10:
        raise ConfigError(
            f"[portfolio].n_positions = {n}: with fewer than 10 names idiosyncratic "
            "risk dominates and the backtest stops measuring the factors"
        )
    scheme = _need(pf, "weighting", "portfolio")
    if scheme not in _WEIGHTING_SCHEMES:
        raise ConfigError(f"[portfolio].weighting must be one of {list(_WEIGHTING_SCHEMES)}")
    max_w = _fraction(_need(pf, "max_weight", "portfolio"), "max_weight", allow_zero=False)
    min_w = _fraction(_need(pf, "min_weight", "portfolio"), "min_weight")
    if min_w >= max_w:
        raise ConfigError("[portfolio]: min_weight must be less than max_weight")
    if max_w * n < 1.0:
        raise ConfigError(
            f"[portfolio]: max_weight ({max_w}) x n_positions ({n}) = {max_w * n:.2f} < 1. "
            "The caps prevent investing 100% of capital"
        )
    if int(_need(pf, "buffer_rank", "portfolio")) < n:
        raise ConfigError("[portfolio].buffer_rank must be >= n_positions")
    _fraction(_need(pf, "max_sector_w", "portfolio"), "max_sector_w", allow_zero=False)
    _fraction(_need(pf, "cash_buffer", "portfolio"), "cash_buffer")

    # -- costes -----------------------------------------------------------
    for key in ("commission_bps", "spread_bps", "slippage_bps"):
        value = float(_need(data["costs"], key, "costs"))
        if value < 0:
            raise ConfigError(f"[costs].{key} negative: a cost does not pay dividends")

    # -- backtest ---------------------------------------------------------
    bt = data["backtest"]
    _positive(_need(bt, "initial_capital", "backtest"), "backtest.initial_capital")
    if not str(_need(bt, "benchmark", "backtest")).strip():
        raise ConfigError("[backtest].benchmark empty: without a reference there is no alpha to measure")

    # -- validacion -------------------------------------------------------
    val = data["validation"]
    if _positive(_need(val, "train_years", "validation"), "train_years") < 2:
        raise ConfigError("[validation].train_years < 2: window too short to calibrate")
    _positive(_need(val, "test_years", "validation"), "test_years")
    if int(_need(val, "n_quantiles", "validation")) < 3:
        raise ConfigError("[validation].n_quantiles < 3: there is no spread to look at")
    if int(_need(val, "newey_west_lag", "validation")) < 0:
        raise ConfigError("[validation].newey_west_lag cannot be negative")


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
            f"{cfg_path} does not exist. The repo ships 'config/strategy.toml'; "
            "if you renamed it, pass the path with --config"
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


# ---------------------------------------------------------------------------
#  Politica de riesgo (config/risk.toml)
#
#  Fichero aparte porque no decide que se compra: niveles de confianza,
#  escenarios y limites de vigilancia. Tiene su propio fingerprint. Si estuviera
#  dentro de strategy.toml, mover un limite de VaR invalidaria la cache del
#  panel y haria incomparables dos backtests identicos.
# ---------------------------------------------------------------------------

_RISK_SECTIONS = ("var", "montecarlo", "exposure", "factor_model", "history", "liquidity",
                  "stress", "limits")


def _confidence(value: Any, name: str) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"'{name}' must be numeric, got {value!r}") from None
    if not 0.5 < num < 1.0:
        raise ConfigError(
            f"'{name}' = {num}: a confidence level is between 0.5 and 1 (0.99, not 99)"
        )
    return num


def _positive_ints(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"'{name}' must be a non-empty list of positive integers")
    out = []
    for item in value:
        if not isinstance(item, int) or item <= 0:
            raise ConfigError(f"'{name}' contains {item!r}: positive integers only")
        out.append(int(item))
    return out


def _fractions(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"'{name}' must be a non-empty list of fractions")
    return [_fraction(v, name, allow_zero=False) for v in value]


def _iso_date(value: Any, name: str) -> str:
    text = str(value)
    try:
        date.fromisoformat(text)
    except ValueError:
        raise ConfigError(f"'{name}' = {value!r}: it must be a YYYY-MM-DD date") from None
    return text


def validate_risk(data: Mapping[str, Any]) -> None:
    """Valida `config/risk.toml`. Lanza `ConfigError` con un mensaje accionable.

    Los nombres de sector de los escenarios y las claves de [limits] se
    comprueban en `risk.analysis.validate_policy`, que conoce los sectores del
    universo y las medidas de riesgo: aqui solo se comprueban tipos y rangos.
    """
    for section in _RISK_SECTIONS:
        if section not in data:
            raise ConfigError(f"missing section [{section}] in the risk policy")

    var = data["var"]
    confidences = _need(var, "confidence", "var")
    if not isinstance(confidences, list) or not confidences:
        raise ConfigError("[var].confidence must be a list, for example [0.95, 0.99]")
    for c in confidences:
        _confidence(c, "var.confidence")
    _positive_ints(_need(var, "horizons_d", "var"), "var.horizons_d")
    lam = float(_need(var, "ewma_lambda", "var"))
    if not 0.0 < lam < 1.0:
        raise ConfigError(f"[var].ewma_lambda = {lam}: it must be in (0, 1); 0.94 is RiskMetrics")
    bt = _need(var, "backtest", "var")
    _confidence(_need(bt, "confidence", "var.backtest"), "var.backtest.confidence")
    if int(_need(bt, "window_d", "var.backtest")) < 60:
        raise ConfigError("[var.backtest].window_d < 60: the tail quantile would be noise")
    if int(_need(bt, "recent_d", "var.backtest")) < 50:
        raise ConfigError("[var.backtest].recent_d < 50: the traffic light needs at least 50 sessions")
    garch = _need(var, "garch", "var")
    if int(_need(garch, "n_paths", "var.garch")) < 1000:
        raise ConfigError("[var.garch].n_paths < 1000: the 99% ES would come from 10 paths")
    _positive_ints(_need(garch, "horizons_d", "var.garch"), "var.garch.horizons_d")

    mc = data["montecarlo"]
    _need(mc, "seed", "montecarlo")
    boot = _need(mc, "bootstrap", "montecarlo")
    if int(_need(boot, "n_paths", "montecarlo.bootstrap")) < 500:
        raise ConfigError(
            "[montecarlo.bootstrap].n_paths < 500: the tail percentiles do not stabilize"
        )
    years = _need(boot, "horizons_y", "montecarlo.bootstrap")
    if not isinstance(years, list) or not years or any(float(y) <= 0 for y in years):
        raise ConfigError("[montecarlo.bootstrap].horizons_y must be a list of positive years")
    if int(_need(boot, "mean_block_d", "montecarlo.bootstrap")) < 1:
        raise ConfigError("[montecarlo.bootstrap].mean_block_d must be >= 1")
    haircut = float(_need(boot, "return_haircut", "montecarlo.bootstrap"))
    if not 0.0 <= haircut <= 0.20:
        raise ConfigError(
            f"[montecarlo.bootstrap].return_haircut = {haircut}: it is an ANNUAL haircut "
            "between 0 and 0.20 (0.02 is two points of CAGR)"
        )
    _fractions(_need(boot, "drawdown_thresholds", "montecarlo.bootstrap"),
               "montecarlo.bootstrap.drawdown_thresholds")
    par = _need(mc, "parametric", "montecarlo")
    if int(_need(par, "n_sims", "montecarlo.parametric")) < 1000:
        raise ConfigError(
            "[montecarlo.parametric].n_sims < 1000: the 99% ES would come from 10 scenarios"
        )
    _positive_ints(_need(par, "horizons_d", "montecarlo.parametric"),
                   "montecarlo.parametric.horizons_d")
    if float(_need(par, "t_dof", "montecarlo.parametric")) <= 2:
        raise ConfigError(
            "[montecarlo.parametric].t_dof <= 2: the Student t has no finite variance "
            "and cannot be matched to the estimated covariance"
        )

    exp = data["exposure"]
    if int(_need(exp, "cov_window_d", "exposure")) < 60:
        raise ConfigError(
            "[exposure].cov_window_d < 60: a covariance of 30 names with barely more data than names"
        )
    _fraction(_need(exp, "min_coverage", "exposure"), "exposure.min_coverage")
    if int(_need(exp, "adv_window_d", "exposure")) < 5:
        raise ConfigError("[exposure].adv_window_d < 5: the median volume would come from one odd day")
    _fraction(_need(exp, "participation", "exposure"), "exposure.participation", allow_zero=False)

    if int(_need(data["factor_model"], "window_d", "factor_model")) < 120:
        raise ConfigError(
            "[factor_model].window_d < 120: the factor covariance and the specific "
            "variance need at least half a year of daily regressions"
        )

    hist = data["history"]
    if int(_need(hist, "rolling_window_d", "history")) < 20:
        raise ConfigError("[history].rolling_window_d < 20: rolling risk would be daily noise")
    if int(_need(hist, "calibration_window_d", "history")) < 60:
        raise ConfigError("[history].calibration_window_d < 60: a forecast with less than a quarter")

    liq = data["liquidity"]
    _positive(_need(liq, "impact_coef", "liquidity"), "liquidity.impact_coef")
    _fraction(_need(liq, "stress_adv_mult", "liquidity"), "liquidity.stress_adv_mult", allow_zero=False)
    for key in ("stress_vol_mult", "stress_cost_mult"):
        if float(_need(liq, key, "liquidity")) < 1.0:
            raise ConfigError(
                f"[liquidity].{key} < 1: a crisis scenario cannot make exiting cheaper"
            )
    navs = _need(liq, "capacity_navs", "liquidity")
    if not isinstance(navs, list) or not navs or any(float(v) <= 0 for v in navs):
        raise ConfigError("[liquidity].capacity_navs must be a list of positive NAVs")
    _fraction(_need(liq, "capacity_budget", "liquidity"), "liquidity.capacity_budget", allow_zero=False)

    st = data["stress"]
    _iso_date(_need(st, "history_start", "stress"), "stress.history_start")
    sv = _need(st, "stressed_var", "stress")
    _confidence(_need(sv, "confidence", "stress.stressed_var"), "stress.stressed_var.confidence")
    if int(_need(sv, "window_d", "stress.stressed_var")) < 60:
        raise ConfigError("[stress.stressed_var].window_d < 60: the tail quantile would be noise")
    for i, sc in enumerate(st.get("covariance", []), start=1):
        where = f"stress.covariance #{i}"
        _need(sc, "name", where)
        if float(sc.get("vol_mult", 1.0)) <= 0:
            raise ConfigError(f"[{where}] '{sc['name']}': vol_mult must be positive")
        _fraction(sc.get("corr_blend", 0.0), f"{where}.corr_blend")
    for i, sc in enumerate(st.get("historical", []), start=1):
        where = f"stress.historical #{i}"
        for key in ("name", "start", "end"):
            _need(sc, key, where)
        start = _iso_date(sc["start"], f"{where}.start")
        end = _iso_date(sc["end"], f"{where}.end")
        if start >= end:
            raise ConfigError(f"[{where}] '{sc['name']}': start must be before end")
    for i, sc in enumerate(st.get("hypothetical", []), start=1):
        where = f"stress.hypothetical #{i}"
        _need(sc, "name", where)
        market = float(_need(sc, "market", where))
        if market <= -1.0:
            raise ConfigError(f"[{where}] '{sc['name']}': market <= -100% is not a scenario")
        unknown = set(sc.get("factors", {})) - set(_KNOWN_FACTORS)
        if unknown:
            raise ConfigError(
                f"[{where}] '{sc['name']}': unknown factors {sorted(unknown)}. "
                f"Known: {list(_KNOWN_FACTORS)}"
            )
    _fractions(_need(_need(st, "reverse", "stress"), "loss_thresholds", "stress.reverse"),
               "stress.reverse.loss_thresholds")

    limits = data["limits"]
    warn = float(_need(limits, "warning_fraction", "limits"))
    if not 0.0 < warn < 1.0:
        raise ConfigError(f"[limits].warning_fraction = {warn}: it must be in (0, 1)")
    for key, value in limits.items():
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"[limits].{key} must be numeric, got {value!r}") from None
        if num <= 0:
            raise ConfigError(f"[limits].{key} = {num}: a limit must be positive")


def load_risk_config(path: str | Path | None = None) -> Config:
    """Lee `config/risk.toml`, lo valida y devuelve una `Config` de tipo riesgo."""
    cfg_path = Path(path) if path is not None else RISK_CONFIG
    if not cfg_path.exists():
        raise ConfigError(
            f"{cfg_path} does not exist. The repo ships 'config/risk.toml'; if you "
            "renamed it, pass the path with --riesgo-config"
        )
    with cfg_path.open("rb") as fh:
        data = tomllib.load(fh)
    validate_risk(data)
    return Config(data=data, source=cfg_path, fingerprint=_fingerprint(data), kind="risk")


def risk_config_from_dict(data: Mapping[str, Any]) -> Config:
    """Politica de riesgo en memoria, para tests. Valida igual que la del disco."""
    payload = copy.deepcopy(dict(data))
    validate_risk(payload)
    return Config(data=payload, source=None, fingerprint=_fingerprint(payload), kind="risk")
