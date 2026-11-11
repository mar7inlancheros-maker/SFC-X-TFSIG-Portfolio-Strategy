"""Limites de vigilancia: la politica de `config/risk.toml` frente a la cartera.

Los limites NO restringen la cartera. El modelo compra lo que compra; esto dice
si lo que compro cabe en la politica de riesgo del comite. Un limite BREACHED se
escala: el comite decide si actua (reducir, cubrir, cambiar un parametro de
strategy.toml) o lo acepta por escrito.

**`LIMIT_SPECS` es la tabla de medidas.** Cada limite del TOML dice con que
medida se compara y en que direccion. Un limite sin entrada aqui lanza
`ConfigError` antes de calcular nada: un numero que nadie sabe con que comparar
no protege de nada, y ignorarlo en silencio seria peor que no tenerlo.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config, ConfigError


@dataclass(frozen=True)
class LimitSpec:
    label: str
    measure: str       # clave del diccionario de medidas de `analysis`
    direction: str     # "max": la medida no debe superarlo | "min": no debe bajar de el
    unit: str          # "pct" | "num" | "days"
    action: str        # que hacer si se excede, en una frase


LIMIT_SPECS: dict[str, LimitSpec] = {
    "max_ex_ante_vol": LimitSpec(
        "Annual ex-ante volatility", "ex_ante_vol", "max", "pct",
        "check whether the rise comes from a market regime or from specific names"),
    "max_var_99_1d": LimitSpec(
        "1-day 99% VaR (Student t)", "var_99_1d", "max", "pct",
        "compare with the ES and decide whether to cut exposure or accept it"),
    "max_beta": LimitSpec(
        "Beta to the benchmark", "beta", "max", "num",
        "the portfolio bets on the market more than on the factors; review cyclical sectors"),
    "max_tracking_error": LimitSpec(
        "Ex-ante tracking error", "tracking_error", "max", "pct",
        "the deviation from the benchmark exceeds the mandate; review concentration"),
    "max_current_drawdown": LimitSpec(
        "Current drawdown from the peak", "current_drawdown", "max", "pct",
        "call a model review: parameters are not changed to climb out of the hole"),
    "max_name_risk_share": LimitSpec(
        "Largest contribution of one name to risk", "max_name_risk_share", "max", "pct",
        "the weight cap is not enough for that name; consider a risk cap"),
    "max_sector_risk_share": LimitSpec(
        "Largest contribution of one sector to risk", "max_sector_risk_share", "max", "pct",
        "the sector weight cap does not contain the risk; review max_sector_w"),
    "max_liquidation_days": LimitSpec(
        "Days to liquidate the least liquid position", "max_liquidation_days", "max", "days",
        "reduce that position or raise the universe liquidity filter"),
    "min_effective_names": LimitSpec(
        "Effective number of names", "effective_names", "min", "num",
        "the portfolio is more concentrated than its number of positions says"),
    "max_stress_loss": LimitSpec(
        "Worst loss in hypothetical scenarios", "max_stress_loss", "max", "pct",
        "identify the scenario and the names that explain it"),
    "max_stressed_var_99_1d": LimitSpec(
        "Stressed 1-day 99% VaR (worst historical window)", "stressed_var_99_1d", "max", "pct",
        "today's portfolio would not survive its worst year; review beta and concentration"),
    "max_liquidation_cost": LimitSpec(
        "Cost of liquidating the whole book in a crisis", "liquidation_cost_stress", "max", "pct",
        "reduce the lowest-volume positions or the capital under management"),
}


def validate_limit_keys(risk_cfg: Config) -> None:
    """Lanza `ConfigError` si [limits] trae una clave sin medida conocida."""
    unknown = sorted(k for k in risk_cfg.section("limits") if k != "warning_fraction"
                     and k not in LIMIT_SPECS)
    if unknown:
        raise ConfigError(
            f"[limits] has keys with no known measure: {unknown}. Known: "
            f"{sorted(LIMIT_SPECS)}. If it is a new limit, add its LimitSpec in "
            "risk/limits.py with the measure that feeds it"
        )


def _status(value: float, limit: float, direction: str, warn: float) -> str:
    if value is None or not np.isfinite(value):
        return "NO DATA"
    if direction == "max":
        if value > limit:
            return "BREACHED"
        return "WARNING" if value >= warn * limit else "OK"
    if value < limit:
        return "BREACHED"
    return "WARNING" if value < limit / warn else "OK"


def check_limits(measures: Mapping[str, float], risk_cfg: Config) -> pd.DataFrame:
    """Una fila por limite: medida, limite, uso y estado.

    Una medida ausente de `measures` (el drawdown en una comprobacion previa a
    operar, que no tiene historia) sale como NO DATA, no como OK.
    """
    validate_limit_keys(risk_cfg)
    limits = risk_cfg.section("limits")
    warn = float(limits["warning_fraction"])
    rows = []
    for key, limit in limits.items():
        if key == "warning_fraction":
            continue
        spec = LIMIT_SPECS[key]
        value = measures.get(spec.measure, float("nan"))
        value = float(value) if value is not None else float("nan")
        limit = float(limit)
        if spec.direction == "max":
            usage = value / limit if limit else float("nan")
        else:
            usage = limit / value if value else float("nan")
        status = _status(value, limit, spec.direction, warn)
        rows.append({
            "limit": key,
            "label": spec.label,
            "value": value,
            "threshold": limit,
            "direction": spec.direction,
            "unit": spec.unit,
            "usage": usage,
            "status": status,
            "action": spec.action if status in ("BREACHED", "WARNING") else "",
        })
    return pd.DataFrame(rows)


def overall_status(checks: pd.DataFrame) -> str:
    """El peor estado de la tabla.

    Un limite NO DATA no cuenta como OK: si ninguno tiene dato, el estado es
    NO DATA. Si algunos lo tienen (antes de operar no hay drawdown), manda el
    peor de los que si.
    """
    if checks.empty or (checks["status"] == "NO DATA").all():
        return "NO DATA"
    for status in ("BREACHED", "WARNING"):
        if (checks["status"] == status).any():
            return status
    return "OK"
