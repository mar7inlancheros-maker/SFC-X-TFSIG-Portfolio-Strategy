"""Robustez: se sostiene el resultado si cambian las decisiones arbitrarias?

Se vuelve a correr el backtest del modo A cambiando UNA cosa cada vez:

- ventana de evaluacion (1, 3, 5 anos);
- frecuencia de rebalanceo;
- coste de transaccion;
- metodo de construccion;
- estimador de volatilidad (muestral frente a EWMA).

Y, aparte, la estabilidad del RANKING del Quant Score si los pesos de sus
componentes fueran otros razonables.

**Como leerlo.** Lo que se busca no es la mejor fila, es la dispersion. Un
Sharpe que va de 0,9 a 1,1 entre variantes es una propiedad de la cartera; uno
que va de -0,2 a 1,4 depende de una decision que alguien tomo, y cualquier
numero concreto de esa tabla es anecdotico. **No se elige la mejor variante:**
hacerlo es sobreajustar con pasos extra.
"""

from __future__ import annotations

import pandas as pd

from .backtest import engine as bt
from .backtest import metrics as btm


def run_robustness(result) -> dict[str, object]:
    from .analysis import backtest_config  # evita import circular

    s = result.settings
    clean = result.clean
    signs = pd.Series({**{t: 1.0 for t in result.longs}, **{t: -1.0 for t in result.shorts}})
    schedule = bt.constant_schedule(signs)
    rob = s.get("robustness", {}) or {}
    rows = []

    def record(dimension: str, variant: str, cfg: bt.BacktestConfig, start: pd.Timestamp) -> None:
        res = bt.run(clean.returns, clean.benchmark, schedule, cfg, start=start, mode="A")
        bench = (1.0 + clean.returns[clean.benchmark].loc[res.nav.index].fillna(0.0)).cumprod()
        m = btm.summary(res, bench)
        rows.append({"dimension": dimension, "variant": variant,
                     "sharpe": m.get("sharpe"), "cagr": m.get("cagr"),
                     "max_drawdown": m.get("max_drawdown"), "turnover_annual": m.get("turnover_annual"),
                     "cost_drag_annual": m.get("cost_drag_annual")})

    base = backtest_config(s)
    for lb in rob.get("lookbacks", ["1y", "3y", "5y"]):
        years = int(str(lb).rstrip("y"))
        start = result.as_of - pd.DateOffset(years=years)
        if start < clean.returns.index[0] + pd.DateOffset(years=1):
            rows.append({"dimension": "lookback", "variant": lb, "sharpe": None,
                         "note": "fuera de los datos descargados"})
            continue
        record("lookback", lb, base, start)

    for freq in rob.get("rebalances", ["monthly", "quarterly"]):
        record("rebalance", freq, backtest_config(s, rebalance=freq), result.eval_start)

    for tc in rob.get("transaction_costs", [0.0005, 0.0015, 0.003]):
        record("transaction_cost", f"{float(tc):.2%}", backtest_config(s, transaction_cost=float(tc)), result.eval_start)

    for method in rob.get("constructions", ["equal_weight", "inverse_vol", "risk_parity", "min_variance"]):
        record("construction", method, backtest_config(s, method=method), result.eval_start)

    for est in ("sample", "ewma"):
        record("vol_estimator", est, backtest_config(s, vol_estimator=est), result.eval_start)

    table = pd.DataFrame(rows)
    sharpe = pd.to_numeric(table.get("sharpe"), errors="coerce")
    return {
        "table": table,
        "sharpe_min": float(sharpe.min()) if sharpe.notna().any() else float("nan"),
        "sharpe_max": float(sharpe.max()) if sharpe.notna().any() else float("nan"),
        "sharpe_dispersion": float(sharpe.max() - sharpe.min()) if sharpe.notna().any() else float("nan"),
        "rank_stability": result.rank_stability,
    }
