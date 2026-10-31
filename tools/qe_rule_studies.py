#!/usr/bin/env python
"""Estudios fuera de muestra que decidieron reglas del plan long/short.

    python tools/qe_rule_studies.py time-stop      # stop por tiempo: largos vs cortos
    python tools/qe_rule_studies.py persistence    # el E(R) de la regla por nombre, persiste?
    python tools/qe_rule_studies.py persistence --end 2026-10-30 --years 6

Corren sobre `scoring.reference_peers` del YAML (50 large caps) con la regla
de `trade_plan.simulate_trades`. Descargan precios de Yahoo (con cache). Cada
estudio separa las entradas por fecha en dos mitades: lo que se elige con la
primera se mide en la segunda. Si una regla del plan cambia (stop, objetivos,
salida parcial), volver a correrlos antes de fiarse de las cifras de CLAUDE.md.

Resultados del 30-oct-2026, 6 anos:
- time-stop: cortos 80% de nombres mejor, +0,05R, t 3,7; largos 40%, -0,03R,
  t -1,8. Por eso es regla solo en los cortos.
- persistence: dentro del lado, correlacion de rangos E(R)_1 vs E(R)_2 de
  -0,05 (cortos) y -0,07 (largos). Por eso la conviccion no usa el backtest
  del nombre.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quant_engine import trade_plan as tp  # noqa: E402
from quant_engine.data import loader  # noqa: E402
from quant_engine.settings import DEFAULT_YAML, load_yaml  # noqa: E402
from sfc_tfsig.console import enable_utf8_stdout  # noqa: E402
from sfc_tfsig.validation import newey_west_se  # noqa: E402

COST, BORROW = 0.0005, 0.0025


def _series(f: pd.DataFrame):
    c = f["Close"].dropna()
    h, lo = f["High"].reindex(c.index).fillna(c), f["Low"].reindex(c.index).fillna(c)
    return c, h, lo, tp.atr(h, lo, c)


def _sim(c, h, lo, a, side, p, time_stop=None):
    return tp.simulate_trades(c, h, lo, a, side, p, cost=COST, borrow=BORROW if side < 0 else 0.0,
                              time_stop=time_stop)


def time_stop(market, p) -> None:
    rows = []
    for t, f in market.items():
        c, h, lo, a = _series(f)
        for side in (1, -1):
            sims = _sim(c, h, lo, a, side, p)
            hw = tp.holding_window(sims.iloc[: len(sims) // 2], p)
            if not hw.get("review_day"):
                continue
            o = tp.time_stop_out_of_sample(sims, _sim(c, h, lo, a, side, p, int(hw["review_day"])), p)
            if o:
                rows.append({"ticker": t, "side": side, **o})
    df = pd.DataFrame(rows)
    for label, sub in (("longs", df[df.side > 0]), ("shorts", df[df.side < 0])):
        imp = sub["improvement"]
        print(f"{label:7} n {len(imp):3}  improved {(imp > 0).mean():.0%}  mean {imp.mean():+.3f}R  "
              f"t {imp.mean() / newey_west_se(imp, 0):+.2f}")


def _rank_corr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    r = float(np.corrcoef(x.rank(), y.rank())[0, 1])
    return r, r * math.sqrt((len(x) - 2) / max(1e-12, 1 - r * r))


def persistence(market, p) -> None:
    rows = []
    for t, f in market.items():
        c, h, lo, a = _series(f)
        for side in (1, -1):
            sims = _sim(c, h, lo, a, side, p)
            if len(sims) < 200:
                continue
            cut = sims.index[len(sims) // 2]
            if side < 0 and p.short_time_stop:
                # La regla de los cortos, con el dia de revision de la primera mitad.
                hw = tp.holding_window(sims.loc[sims.index < cut], p)
                if hw.get("review_day"):
                    sims = _sim(c, h, lo, a, side, p, int(hw["review_day"]))
            s1, s2 = tp.trade_stats(sims.loc[sims.index < cut]), tp.trade_stats(sims.loc[sims.index >= cut])
            rows.append({"ticker": t, "side": side, "er1": s1["exp_r"], "t1": s1["t_stat"], "er2": s2["exp_r"]})
    df = pd.DataFrame(rows).dropna(subset=["er1", "er2"])
    for label, sub in (("pooled", df), ("longs", df[df.side > 0]), ("shorts", df[df.side < 0])):
        bad, rest = sub[sub.t1 <= -2], sub[sub.t1 > -2]
        r, rt = _rank_corr(sub.er1, sub.er2)
        print(f"{label:7} n {len(sub):3}  rank corr E(R)1 vs E(R)2 {r:+.2f} (t {rt:+.2f})  "
              f"t1<=-2: n {len(bad)} next E(R) {bad.er2.mean():+.3f}  rest: next E(R) {rest.er2.mean():+.3f}")
    print("pooled mixes the sides: a positive correlation there can be only the market drift against shorts.")


def main(argv: list[str] | None = None) -> int:
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("study", choices=["time-stop", "persistence"])
    ap.add_argument("--end", default=None, help="ultima fecha (AAAA-MM-DD); por defecto, hoy")
    ap.add_argument("--years", type=int, default=6)
    args = ap.parse_args(argv)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    peers = [str(t).upper() for t in load_yaml(DEFAULT_YAML)["scoring"]["reference_peers"]]
    market = loader.load_ohlcv(peers, end - pd.DateOffset(years=args.years), end).ohlcv
    p = tp.PlanParams()
    {"time-stop": time_stop, "persistence": persistence}[args.study](market, p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
