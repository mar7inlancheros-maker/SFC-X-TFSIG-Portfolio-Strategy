"""Del modelo a la mesa: la lista de ordenes del rebalanceo.

El backtest dice que habria pasado. Esto dice que hay que hacer el lunes.

**Posiciones actuales.** Se leen de un CSV que mantiene el equipo
(`data/positions.csv`, columnas `ticker,shares`). Es deliberado que sea un
fichero y no un estado interno del modelo: la cartera real puede diferir de la
teorica -- una orden no se llena, alguien ejecuta a otro precio, entra capital
nuevo -- y la fuente de verdad tiene que ser lo que hay en la cuenta, no lo que
el modelo cree que hay.

**Lo que NO decide este modulo.** No manda ordenes a ningun broker. Genera la
propuesta que el comite aprueba o rechaza, y la deja en CSV para ejecutarla a
mano o cargarla en la plataforma de paper trading. La aprobacion humana es parte
del proceso de gobierno, no un obstaculo tecnico que haya que automatizar.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import Config
from .paths import DATA_DIR, REPORT_DIR

POSITIONS_FILE = DATA_DIR / "positions.csv"


def load_positions(path: Path | None = None) -> pd.Series:
    """Posiciones actuales: ticker -> numero de acciones. Vacio si no hay fichero."""
    path = path or POSITIONS_FILE
    if not path.exists():
        return pd.Series(dtype="float64")
    df = pd.read_csv(path)
    missing = {"ticker", "shares"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} necesita las columnas ticker,shares (faltan {missing})")
    series = df.set_index(df["ticker"].astype(str).str.upper())["shares"].astype(float)
    return series[series != 0]


def save_positions(shares: pd.Series, path: Path | None = None) -> Path:
    """Guarda las posiciones resultantes tras ejecutar."""
    path = path or POSITIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = shares.rename("shares").rename_axis("ticker").reset_index()
    frame.to_csv(path, index=False)
    return path


def build_orders(
    target: pd.DataFrame,
    current_shares: pd.Series,
    nav: float,
    cfg: Config,
    *,
    min_order_notional: float = 100.0,
) -> pd.DataFrame:
    """Cartera objetivo + posiciones actuales -> ordenes.

    `target` viene de `portfolio.build_portfolio` y trae `ticker`, `weight` y
    `price`. `nav` es el valor total de la cuenta HOY, incluida la caja.

    Las ordenes por debajo de `min_order_notional` se descartan: mover 40
    dolares para corregir un peso del 0.02% paga comision y no cambia nada.
    """
    if target.empty:
        return pd.DataFrame(columns=["ticker", "side", "shares", "price", "notional",
                                     "current_weight", "target_weight"])

    prices = target.set_index("ticker")["price"].astype(float)
    target_weights = target.set_index("ticker")["weight"].astype(float)

    universe = prices.index.union(current_shares.index)
    prices = prices.reindex(universe)
    held = current_shares.reindex(universe).fillna(0.0)

    # Una posicion que ya no esta en el objetivo se vende entera; su precio sale
    # de la cartera actual si el modelo ya no la cubre.
    missing_price = prices.isna() & (held != 0)
    if missing_price.any():
        raise ValueError(
            "faltan precios para posiciones que hay que vender: "
            f"{sorted(universe[missing_price])}. Anade el precio manualmente o "
            "amplia el universo antes de generar ordenes."
        )

    current_value = held * prices.fillna(0.0)
    current_weight = current_value / nav if nav else current_value * 0.0

    target_value = target_weights.reindex(universe).fillna(0.0) * nav
    target_shares = (target_value / prices).fillna(0.0)

    delta = target_shares - held
    notional = (delta.abs() * prices).fillna(0.0)

    orders = pd.DataFrame(
        {
            "ticker": universe,
            "side": ["BUY" if d > 0 else "SELL" for d in delta],
            "shares": delta.abs().round(4).to_numpy(),
            "price": prices.to_numpy(),
            "notional": notional.round(2).to_numpy(),
            "current_weight": current_weight.reindex(universe).fillna(0.0).to_numpy(),
            "target_weight": target_weights.reindex(universe).fillna(0.0).to_numpy(),
        }
    )
    orders = orders[(orders["shares"] > 1e-6) & (orders["notional"] >= min_order_notional)]

    estimated_cost = orders["notional"].sum() * cfg.total_cost_bps / 10_000.0
    orders.attrs["estimated_cost"] = float(estimated_cost)
    orders.attrs["turnover"] = float(orders["notional"].sum() / nav / 2.0) if nav else float("nan")

    # Vender antes de comprar: en una cuenta sin margen, el efectivo de las
    # ventas es lo que financia las compras del mismo dia.
    return orders.sort_values(["side", "notional"], ascending=[True, False]).reset_index(drop=True)


def render_orders(orders: pd.DataFrame, target: pd.DataFrame, nav: float,
                  cfg: Config, as_of: pd.Timestamp) -> str:
    """Hoja de ordenes legible, para llevar al comite."""
    lines = [
        f"# Rebalanceo propuesto -- {pd.Timestamp(as_of).date()}",
        "",
        f"- Configuracion: fingerprint `{cfg.fingerprint}`",
        f"- Valor de la cuenta: {nav:,.2f} {cfg.get('meta.base_currency')}",
        f"- Posiciones objetivo: {len(target)}",
        f"- Ordenes: {len(orders)}",
        f"- Rotacion: {orders.attrs.get('turnover', float('nan')) * 100:.1f}%",
        f"- Coste estimado: {orders.attrs.get('estimated_cost', 0.0):,.2f} "
        f"({cfg.total_cost_bps:.0f} bps por lado)",
        "",
        "> Las ventas van primero: financian las compras del mismo dia en una",
        "> cuenta sin margen.",
        "",
    ]

    if orders.empty:
        lines.append("_Sin ordenes: la cartera ya esta en su objetivo._")
        return "\n".join(lines)

    lines += ["| # | Lado | Ticker | Acciones | Precio | Nocional | Peso actual | Peso objetivo |",
              "|---|---|---|---|---|---|---|---|"]
    for i, (_, row) in enumerate(orders.iterrows(), start=1):
        lines.append(
            f"| {i} | {row['side']} | {row['ticker']} | {row['shares']:,.2f} | "
            f"{row['price']:,.2f} | {row['notional']:,.2f} | "
            f"{row['current_weight'] * 100:.2f}% | {row['target_weight'] * 100:.2f}% |"
        )

    lines += ["", "## Cartera objetivo completa", "",
              "| Ticker | Sector | Peso | Score |", "|---|---|---|---|"]
    for _, row in target.sort_values("weight", ascending=False).iterrows():
        score = row.get("score_composite", float("nan"))
        lines.append(
            f"| {row['ticker']} | {row.get('sector', 'n/d')} | "
            f"{row['weight'] * 100:.2f}% | "
            f"{'' if pd.isna(score) else f'{score:.2f}'} |"
        )

    return "\n".join(lines) + "\n"


def save_orders(orders: pd.DataFrame, text: str, as_of: pd.Timestamp) -> dict[str, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(as_of).strftime("%Y%m%d")
    generated = datetime.now().strftime("%H%M")
    csv_path = REPORT_DIR / f"ordenes_{stamp}_{generated}.csv"
    md_path = REPORT_DIR / f"ordenes_{stamp}_{generated}.md"
    orders.to_csv(csv_path, index=False)
    md_path.write_text(text, encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path}
