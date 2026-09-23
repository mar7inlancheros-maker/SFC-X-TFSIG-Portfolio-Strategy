"""Limites de vigilancia: la politica de `config/risk.toml` frente a la cartera.

Los limites NO restringen la cartera. El modelo compra lo que compra; esto dice
si lo que compro cabe en la politica de riesgo del comite. Un limite EXCEDIDO se
escala: el comite decide si actua (reducir, cubrir, cambiar un parametro de
strategy.toml) o lo acepta por escrito.

**`LIMIT_SPECS` es la tabla de medidas.** Cada limite del TOML dice con que
medida se compara y en que direccion. Un limite sin entrada aqui lanza
`ConfigError` antes de calcular nada: un numero que nadie sabe con que comparar
no protege de nada, y ignorarlo en silencio seria peor que no tenerlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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
        "Volatilidad ex-ante anual", "ex_ante_vol", "max", "pct",
        "revisar si la subida viene de un regimen de mercado o de nombres concretos"),
    "max_var_99_1d": LimitSpec(
        "VaR 99% 1 dia (t de Student)", "var_99_1d", "max", "pct",
        "comparar con el ES y decidir si se reduce exposicion o se acepta"),
    "max_beta": LimitSpec(
        "Beta frente al benchmark", "beta", "max", "num",
        "la cartera apuesta al mercado mas que a los factores; revisar sectores ciclicos"),
    "max_tracking_error": LimitSpec(
        "Tracking error ex-ante", "tracking_error", "max", "pct",
        "la desviacion frente al benchmark supera el mandato; revisar concentracion"),
    "max_current_drawdown": LimitSpec(
        "Drawdown actual desde maximos", "current_drawdown", "max", "pct",
        "convocar revision del modelo: no se cambian parametros para salir del agujero"),
    "max_name_risk_share": LimitSpec(
        "Mayor contribucion de un nombre al riesgo", "max_name_risk_share", "max", "pct",
        "el tope de peso no basta para ese nombre; valorar un tope de riesgo"),
    "max_sector_risk_share": LimitSpec(
        "Mayor contribucion de un sector al riesgo", "max_sector_risk_share", "max", "pct",
        "el tope sectorial en peso no contiene el riesgo; revisar max_sector_w"),
    "max_liquidation_days": LimitSpec(
        "Dias para liquidar la posicion menos liquida", "max_liquidation_days", "max", "days",
        "reducir esa posicion o subir el filtro de liquidez del universo"),
    "min_effective_names": LimitSpec(
        "Numero efectivo de nombres", "effective_names", "min", "num",
        "la cartera esta mas concentrada de lo que dice su numero de posiciones"),
    "max_stress_loss": LimitSpec(
        "Peor perdida en escenarios hipoteticos", "max_stress_loss", "max", "pct",
        "identificar el escenario y los nombres que lo explican"),
}


def validate_limit_keys(risk_cfg: Config) -> None:
    """Lanza `ConfigError` si [limits] trae una clave sin medida conocida."""
    unknown = sorted(k for k in risk_cfg.section("limits") if k != "warning_fraction"
                     and k not in LIMIT_SPECS)
    if unknown:
        raise ConfigError(
            f"[limits] trae claves sin medida conocida: {unknown}. Conocidas: "
            f"{sorted(LIMIT_SPECS)}. Si es un limite nuevo, anade su LimitSpec en "
            "risk/limits.py con la medida que lo alimenta"
        )


def _status(value: float, limit: float, direction: str, warn: float) -> str:
    if value is None or not np.isfinite(value):
        return "SIN DATO"
    if direction == "max":
        if value > limit:
            return "EXCEDIDO"
        return "ALERTA" if value >= warn * limit else "OK"
    if value < limit:
        return "EXCEDIDO"
    return "ALERTA" if value < limit / warn else "OK"


def check_limits(measures: Mapping[str, float], risk_cfg: Config) -> pd.DataFrame:
    """Una fila por limite: medida, limite, uso y estado.

    Una medida ausente de `measures` (el drawdown en una comprobacion previa a
    operar, que no tiene historia) sale como SIN DATO, no como OK.
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
            "action": spec.action if status in ("EXCEDIDO", "ALERTA") else "",
        })
    return pd.DataFrame(rows)


def overall_status(checks: pd.DataFrame) -> str:
    """El peor estado de la tabla.

    Un limite SIN DATO no cuenta como OK: si ninguno tiene dato, el estado es
    SIN DATO. Si algunos lo tienen (antes de operar no hay drawdown), manda el
    peor de los que si.
    """
    if checks.empty or (checks["status"] == "SIN DATO").all():
        return "SIN DATO"
    for status in ("EXCEDIDO", "ALERTA"):
        if (checks["status"] == status).any():
            return status
    return "OK"
