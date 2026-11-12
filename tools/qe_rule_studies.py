#!/usr/bin/env python
"""Estudios fuera de muestra que decidieron reglas del plan long/short.

    python tools/qe_rule_studies.py time-stop      # stop por tiempo: largos vs cortos
    python tools/qe_rule_studies.py persistence    # el E(R) de la regla por nombre, persiste?
    python tools/qe_rule_studies.py persistence --end 2026-10-30 --years 6
    python tools/qe_rule_studies.py multifactor    # la regla gana cuando sigue al multifactor?
    python tools/qe_rule_studies.py quant-score    # el Quant Score (solo precios) predice a escala?
    python tools/qe_rule_studies.py short-rule     # otro stop / objetivos mejoraria los cortos?
    python tools/qe_rule_studies.py long-rule      # y los largos?
    python tools/qe_rule_studies.py rule-holdout   # las reglas nuevas aguantan en OTROS nombres?
    python tools/qe_rule_studies.py timing         # los consejos tecnicos de entrada mueven el E(R)?
    python tools/qe_rule_studies.py levels         # y los de soportes y resistencias frente al stop y el TP1?

Corren sobre `scoring.reference_peers` del YAML (50 large caps) con la regla
de `trade_plan.simulate_trades`. Descargan precios de Yahoo (con cache). Cada
estudio separa las entradas por fecha en dos mitades: lo que se elige con la
primera se mide en la segunda. Si una regla del plan cambia (stop, objetivos,
salida parcial), volver a correrlos antes de fiarse de las cifras de CLAUDE.md.

Resultados del 30-oct-2026, 6 anos, con la regla de cada lado (largos: stop
3 ATR, objetivos 2R y 4R; cortos: 2 ATR, 1R y 2R):
- time-stop: cortos 68% de nombres mejor, +0,04R, t 4,3; largos 38%, -0,04R,
  t -2,5. Por eso es regla solo en los cortos.
- short-rule / long-rule: +0,082R (cortos) y +0,084R (largos) frente a 2,5 ATR y
  1,5R / 3R en la segunda mitad. rule-holdout, en 871 nombres que no las
  eligieron: +0,057R (t 1,9) y +0,055R (t 1,8), no confirmado a t >= 2.
- timing (870 nombres sin los 50 pares, 36 meses): ninguna de las seis
  afirmaciones de la ficha movio el E(R) (la mayor, t 1,3; Holm 1,00);
  comprar extendido, +0,07R mejor. Por eso la ficha cita la cifra y no
  aconseja esperar.
- levels (mismas entradas): ninguna de las seis afirmaciones de niveles se
  sostiene; la resistencia antes del TP1 salta en el 87% de los largos. El
  unico p suelto bajo va contra la ficha (cortos, +0,047R, t 2,5, Holm 0,07).
- persistence: dentro del lado, correlacion de rangos E(R)_1 vs E(R)_2 de
  +0,10 (cortos, t 0,7) y -0,11 (largos). Por eso la conviccion no usa el backtest
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
from quant_engine.factors import technical  # noqa: E402
from quant_engine.inference import statistics as stats_mod  # noqa: E402
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
            hw = tp.holding_window(sims.iloc[: len(sims) // 2], tp.for_side(p, side))
            if not hw.get("review_day"):
                continue
            o = tp.time_stop_out_of_sample(sims, _sim(c, h, lo, a, side, p, int(hw["review_day"])),
                                           tp.for_side(p, side))
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
                hw = tp.holding_window(sims.loc[sims.index < cut], tp.for_side(p, side))
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


def multifactor(end: pd.Timestamp, months: int, top: int, p) -> None:
    """La regla del plan abierta tras cada cierre de mes del panel, en los `top`
    nombres mas liquidos, en largo y en corto, agrupada por lo que dice el
    multifactor (`multifactor.agreement`, la misma funcion que usa el plan).

    Criterio fijado antes de mirar: el multifactor merece su escalon de
    conviccion si, en cada lado, respaldar menos neutral da t >= 2 y
    contradecir menos neutral da t <= -2 (Newey-West sobre la serie mensual,
    retardo 3: las operaciones de 63 sesiones se solapan unos tres meses).
    """
    import dataclasses  # noqa: PLC0415

    from quant_engine.data import multifactor as mf  # noqa: PLC0415
    from sfc_tfsig import cli as mf_cli  # noqa: PLC0415
    from sfc_tfsig.config import load_config  # noqa: PLC0415
    from sfc_tfsig.data import cache  # noqa: PLC0415

    panel = cache.read_frame(mf_cli._panel_cache_name(load_config(None)))  # noqa: SLF001
    panel = panel.dropna(subset=["score_composite"])
    panel["percentile"] = panel.groupby("date")["score_composite"].rank(pct=True)
    last_entry = end - pd.offsets.BDay(p.horizon_d + 5)
    dates = sorted(d for d in panel["date"].unique() if d <= last_entry)[-months:]
    picks = (panel[panel["date"].isin(dates)].sort_values("dollar_volume", ascending=False)
             .groupby("date").head(top))
    names = sorted(picks["ticker"].unique())
    print(f"{len(dates)} month-ends {pd.Timestamp(dates[0]):%Y-%m} to {pd.Timestamp(dates[-1]):%Y-%m}, "
          f"{len(names)} names; downloading OHLC...", flush=True)
    market = loader.load_ohlcv(names, pd.Timestamp(dates[0]) - pd.DateOffset(months=3), end).ohlcv
    one = dataclasses.replace(p, history_d=1)
    rows = []
    for t, grp in picks.groupby("ticker"):
        if t not in market:
            continue
        c, h, lo, a = _series(market[t])
        for _, row in grp.iterrows():
            i = int(c.index.searchsorted(row["date"], side="right"))   # primera sesion DESPUES del cierre de mes
            if i < 20 or i + p.horizon_d >= len(c):
                continue
            cut = slice(0, i + p.horizon_d + 1)
            for side in (1, -1):
                sims = tp.simulate_trades(c.iloc[cut], h.iloc[cut], lo.iloc[cut], a.iloc[cut], side, one,
                                          cost=COST, borrow=BORROW if side < 0 else 0.0)
                if len(sims):
                    rows.append({"date": row["date"], "ticker": t, "side": side, "r": float(sims["r_mult"].iloc[-1]),
                                 "agree": mf.agreement(side, row["percentile"])})
    df = pd.DataFrame(rows)
    print(f"{len(df):,} trades")
    for side, label in ((1, "longs"), (-1, "shorts")):
        sub = df[df.side == side]
        monthly = sub.groupby(["date", "agree"])["r"].mean().unstack()
        parts = []
        for k, name in ((1, "backs"), (0, "neutral"), (-1, "contradicts")):
            m = monthly.get(k, pd.Series(dtype=float)).dropna()
            t = m.mean() / newey_west_se(m, 3) if len(m) > 3 else float("nan")
            parts.append(f"{name} {m.mean():+.3f}R (t {t:+.2f}, n {int((sub.agree == k).sum())})")
        diffs = []
        for k, name in ((1, "backs-neutral"), (-1, "contradicts-neutral")):
            d = (monthly.get(k) - monthly.get(0)).dropna() if k in monthly and 0 in monthly else pd.Series(dtype=float)
            t = d.mean() / newey_west_se(d, 3) if len(d) > 3 else float("nan")
            diffs.append(f"{name} {d.mean():+.3f}R t {t:+.2f}")
        print(f"{label:7} " + " | ".join(parts))
        print(f"{'':7} " + " | ".join(diffs))


def quant_score(end: pd.Timestamp, months: int, top: int, p) -> None:
    """El Quant Score del motor, solo con precios (como corre sin WRDS), en los
    `top` nombres mas liquidos de cada cierre de mes del panel del multifactor.

    Las mismas funciones que la validacion del analisis (`analysis.features_at`,
    `score_validation.validate`), con ~600 nombres por fecha en vez de 54. Los
    nombres de cada fecha son los liquidos ESE mes (sin sesgo de supervivencia).
    Criterio fijado antes de mirar: predice si la t del IC a 21 sesiones es >= 2,
    el mismo umbral que exige el plan para un FLIP (`flip_min_ic_t`).
    """
    from quant_engine import analysis  # noqa: PLC0415
    from quant_engine.data import cleaning  # noqa: PLC0415
    from quant_engine.settings import build_settings  # noqa: PLC0415
    from quant_engine.signals import score_validation as sv  # noqa: PLC0415
    from sfc_tfsig import cli as mf_cli  # noqa: PLC0415
    from sfc_tfsig.config import load_config  # noqa: PLC0415
    from sfc_tfsig.data import cache  # noqa: PLC0415

    panel = cache.read_frame(mf_cli._panel_cache_name(load_config(None)))  # noqa: SLF001
    last_entry = end - pd.offsets.BDay(p.horizon_d + 5)
    dates = sorted(d for d in panel["date"].unique() if d <= last_entry)[-months:]
    picks = (panel[panel["date"].isin(dates)].sort_values("dollar_volume", ascending=False)
             .groupby("date").head(top))
    by_month = {pd.Timestamp(d).to_period("M"): sorted(g["ticker"]) for d, g in picks.groupby("date")}
    names = sorted(picks["ticker"].unique())
    settings = build_settings(names[0], "")
    start = pd.Timestamp(dates[0]) - pd.DateOffset(years=settings.lookback_years + 1, days=10)
    print(f"{len(dates)} month-ends, {len(names)} names; downloading OHLC from {start:%Y-%m}...", flush=True)
    market = loader.load_ohlcv([*names, settings.benchmark], start, end)
    clean = cleaning.align(market, settings.benchmark)
    names = [t for t in names if t in clean.close]
    weights = settings.get("signal_weights", {}) or {}
    memo: dict[pd.Timestamp, pd.DataFrame] = {}

    def feats(d: pd.Timestamp) -> pd.DataFrame:
        if d not in memo:
            alive = [t for t in by_month.get(d.to_period("M"), []) if t in clean.close]
            memo[d] = analysis.features_at(settings, clean, alive, d, None, {})
            print(f"  {d:%Y-%m-%d}: {len(memo[d])} names", flush=True)
        return memo[d]

    for h in (21, p.horizon_d):
        last = clean.calendar[-1 - h]
        month_ends = [d for d in sv.month_end_dates(clean.calendar, pd.Timestamp(dates[0]), last, 10_000)
                      if d.to_period("M") in by_month]
        res = sv.validate(feats, sv.forward_returns(clean.close[names], month_ends, h), weights, horizon=h)
        if not res.get("available"):
            print(f"h {h}: {res.get('reason')}")
            continue
        print(f"h {h:2}: {res['months']} months, {res['avg_names']:.0f} names/date | IC {res['ic_mean']:+.4f} "
              f"t {res['ic_t']:+.2f} hit {res['ic_hit']:.0%} | top-bottom tercile {res['spread_mean']:+.2%} "
              f"t {res['spread_t']:+.2f}")
        comp = res["components"]
        if len(comp):
            for name, r in comp.sort_values("ic_t").iterrows():
                print(f"      {name:22} weight {r['weight']:.2f}  IC {r['ic_mean']:+.4f}  t {r['ic_t']:+.2f}")


SHORT_GRID = [(s, tp) for s in (2.0, 2.5, 3.0) for tp in ((1.0, 2.0), (1.5, 3.0), (2.0, 4.0))]


def short_rule(market, p) -> None:
    side_rule(market, p, -1)


def long_rule(market, p) -> None:
    side_rule(market, p, 1)


def side_rule(market, p, side: int) -> None:
    """Rejilla de stop (ATR) y objetivos (R) para un lado (los cortos con el stop
    por tiempo de su regla), en toda la historia de cada nombre partida por fecha.

    Criterio fijado antes de mirar: la combinacion con mejor E(R) media en la
    PRIMERA mitad (una para todos: el plan usa una sola regla) sustituye a la
    actual solo si en la SEGUNDA mitad mejora +0,05R o mas, con t >= 2 entre
    nombres y en el 60% de ellos o mas. E(R) es comparable entre stops porque el
    tamano se fija por riesgo: el stop cuesta lo mismo del capital.
    """
    import dataclasses  # noqa: PLC0415

    rule = tp.for_side(p, side)
    current = (rule.stop_atr, tuple(rule.tp_r))
    rows = []
    for t, f in market.items():
        c, h, lo, _ = _series(f)
        for stop_atr, tps in SHORT_GRID:
            # Tambien los campos del corto: `for_side` los aplica dentro de la simulacion.
            q = dataclasses.replace(p, stop_atr=stop_atr, tp_r=tps, short_stop_atr=stop_atr, short_tp_r=tps,
                                    history_d=100_000)
            a = tp.atr(h, lo, c)
            sims = _sim(c, h, lo, a, side, q)
            if len(sims) < 400:
                continue
            cut = sims.index[len(sims) // 2]
            hw = tp.holding_window(sims.loc[sims.index < cut], q)
            if side < 0 and p.short_time_stop and hw.get("review_day"):
                sims = _sim(c, h, lo, a, side, q, int(hw["review_day"]))
            rows.append({"ticker": t, "combo": (stop_atr, tps),
                         "er1": tp.trade_stats(sims.loc[sims.index < cut])["exp_r"],
                         "er2": tp.trade_stats(sims.loc[sims.index >= cut])["exp_r"]})
    df = pd.DataFrame(rows)
    first = df.groupby("combo")["er1"].mean().sort_values(ascending=False)
    second = df.groupby("combo")["er2"].mean()
    print("combo (stop ATR, TP1 R, TP2 R)   first half   second half")
    for combo, v in first.items():
        mark = "  <- current" if combo == current else ""
        print(f"  {combo[0]:.1f}  {combo[1][0]:.1f} / {combo[1][1]:.1f}            {v:+.3f}R      {second[combo]:+.3f}R{mark}")
    best = first.index[0]
    wide = df.pivot(index="ticker", columns="combo", values="er2")
    diff = (wide[best] - wide[current]).dropna()
    t = diff.mean() / newey_west_se(diff, 0) if len(diff) > 2 and diff.std() > 0 else float("nan")
    print(f"picked on the first half: {best}; second half vs current: {diff.mean():+.3f}R, t {t:+.2f}, "
          f"improved in {(diff > 0).mean():.0%} of {len(diff)} names")
    adopt = best != current and diff.mean() >= 0.05 and t >= 2 and (diff > 0).mean() >= 0.6
    print("verdict: " + (f"ADOPT the new rule for {'shorts' if side < 0 else 'longs'}" if adopt
                         else "KEEP the current rule"))


OLD_RULE = (2.5, (1.5, 3.0))   # la de los dos lados hasta octubre de 2026


def rule_holdout(end: pd.Timestamp, months: int, top: int, p) -> None:
    """Las reglas de cada lado contra la anterior, en nombres que no las eligieron.

    `short-rule` y `long-rule` eligieron con los 50 pares de referencia; esto
    mira los `top` mas liquidos de cada cierre de mes del panel SIN esos 50,
    con una entrada por nombre y mes, y compara solo los niveles (sin stop por
    tiempo en ninguna de las dos). Criterio fijado antes de mirar: la regla de
    un lado se confirma si nueva menos anterior da t >= 2 (Newey-West sobre la
    serie mensual, retardo 3).
    """
    import dataclasses  # noqa: PLC0415

    dates, picks = _holdout_picks(end, months, top, p)
    names = sorted(picks["ticker"].unique())
    print(f"{len(dates)} month-ends, {len(names)} names (reference peers left out)", flush=True)
    market = loader.load_ohlcv(names, pd.Timestamp(dates[0]) - pd.DateOffset(months=3), end).ohlcv
    rules = {}
    for side in (1, -1):
        new = tp.for_side(p, side)
        for label, (stop_atr, tps) in (("new", (new.stop_atr, tuple(new.tp_r))), ("old", OLD_RULE)):
            rules[(side, label)] = dataclasses.replace(p, stop_atr=stop_atr, tp_r=tps, short_stop_atr=stop_atr,
                                                       short_tp_r=tps, history_d=1)
    rows = []
    for t, grp in picks.groupby("ticker"):
        if t not in market:
            continue
        c, h, lo, a = _series(market[t])
        for d in grp["date"]:
            i = int(c.index.searchsorted(d, side="right"))
            if i < 20 or i + p.horizon_d >= len(c):
                continue
            cut = slice(0, i + p.horizon_d + 1)
            for (side, label), q in rules.items():
                sims = tp.simulate_trades(c.iloc[cut], h.iloc[cut], lo.iloc[cut], a.iloc[cut], side, q,
                                          cost=COST, borrow=BORROW if side < 0 else 0.0)
                if len(sims):
                    rows.append({"date": d, "ticker": t, "side": side, "rule": label,
                                 "r": float(sims["r_mult"].iloc[-1])})
    df = pd.DataFrame(rows)
    print(f"{len(df) // 4:,} entries per rule and side")
    for side, name in ((1, "longs"), (-1, "shorts")):
        monthly = df[df.side == side].groupby(["date", "rule"])["r"].mean().unstack()
        diff = (monthly["new"] - monthly["old"]).dropna()
        t = diff.mean() / newey_west_se(diff, 3)
        ok = t >= 2
        print(f"{name:7} new {monthly['new'].mean():+.3f}R  old {monthly['old'].mean():+.3f}R  "
              f"difference {diff.mean():+.3f}R t {t:+.2f}, better in {(diff > 0).mean():.0%} of months -> "
              + ("CONFIRMED" if ok else "NOT CONFIRMED"))


# (lado, condicion de `technical.timing_flags`, signo que afirma la ficha)
TIMING_CLAIMS = [(1, "extended", -1), (1, "against_trend", -1), (1, "with_trend", 1),
                 (-1, "extended", -1), (-1, "against_trend", -1), (-1, "with_trend", 1)]
# (lado, condicion de `technical.level_flags`, signo que afirma la ficha)
LEVEL_CLAIMS = [(1, "stop_behind_level", 1), (1, "level_inside_stop", -1), (1, "level_before_tp1", -1),
                (-1, "stop_behind_level", 1), (-1, "level_inside_stop", -1), (-1, "level_before_tp1", -1)]
MIN_PER_GROUP = 5


def _holdout_picks(end: pd.Timestamp, months: int, top: int, p):
    from sfc_tfsig import cli as mf_cli  # noqa: PLC0415
    from sfc_tfsig.config import load_config  # noqa: PLC0415
    from sfc_tfsig.data import cache  # noqa: PLC0415

    peers = {str(t).upper() for t in load_yaml(DEFAULT_YAML)["scoring"]["reference_peers"]}
    panel = cache.read_frame(mf_cli._panel_cache_name(load_config(None)))  # noqa: SLF001
    panel = panel[~panel["ticker"].isin(peers)]
    last_entry = end - pd.offsets.BDay(p.horizon_d + 5)
    dates = sorted(d for d in panel["date"].unique() if d <= last_entry)[-months:]
    picks = (panel[panel["date"].isin(dates)].sort_values("dollar_volume", ascending=False)
             .groupby("date").head(top))
    return dates, picks


def timing_table(df: pd.DataFrame, claims=None) -> pd.DataFrame:
    """Cada afirmacion de la ficha contra las entradas, con su p de Holm.

    `df`: una fila por entrada con `date`, `side`, `r` y una columna booleana
    por condicion. La diferencia es mensual (con la condicion menos sin ella,
    meses con al menos `MIN_PER_GROUP` entradas en cada grupo), con t de
    Newey-West de retardo 3: las operaciones de 63 sesiones se solapan unos
    tres meses.
    """
    out = []
    for side, flag, claim in (claims or TIMING_CLAIMS):
        sub = df[df.side == side]
        g = sub.groupby(["date", flag])["r"].agg(["mean", "size"]).unstack(flag)
        n_true = g["size"][True] if True in g["size"] else pd.Series(0, index=g.index)
        n_false = g["size"][False] if False in g["size"] else pd.Series(0, index=g.index)
        both = g[(n_true.fillna(0) >= MIN_PER_GROUP) & (n_false.fillna(0) >= MIN_PER_GROUP)]
        diff = (both["mean"][True] - both["mean"][False]).dropna() if len(both) else pd.Series(dtype=float)
        t = diff.mean() / newey_west_se(diff, 3) if len(diff) > 6 and diff.std() > 0 else float("nan")
        p_two = 2 * (1 - stats_mod.NormalDist().cdf(abs(t))) if np.isfinite(t) else float("nan")
        # La tasa de stop es descriptiva: la ficha decia "el stop es mas probable";
        # el veredicto sale del E(R), que es lo que cobra la regla.
        stopped = sub["stopped"] if "stopped" in sub else pd.Series(np.nan, index=sub.index)
        hit_tp1 = sub["tp1"] if "tp1" in sub else pd.Series(np.nan, index=sub.index)
        out.append({"side": side, "flag": flag, "claim": claim, "share": float(sub[flag].mean()),
                    "er_flag": float(sub.loc[sub[flag], "r"].mean()),
                    "er_rest": float(sub.loc[~sub[flag], "r"].mean()),
                    "stop_flag": float(stopped[sub[flag]].mean()), "stop_rest": float(stopped[~sub[flag]].mean()),
                    "tp1_flag": float(hit_tp1[sub[flag]].mean()), "tp1_rest": float(hit_tp1[~sub[flag]].mean()),
                    "diff": float(diff.mean()) if len(diff) else float("nan"), "t": t, "p": p_two,
                    "months": len(diff)})
    res = pd.DataFrame(out)
    res["p_holm"] = stats_mod.holm(res["p"])
    sig = res["p_holm"] < 0.05
    agrees = np.sign(res["diff"]) == res["claim"]
    res["verdict"] = np.where(sig & agrees, "SUPPORTED", np.where(sig, "CONTRADICTED", "NOT SUPPORTED"))
    return res


def timing(end: pd.Timestamp, months: int, top: int, p) -> None:
    """Los consejos de entrada de la ficha (`technical.timing_advice`), medidos.

    La ficha dice que comprar sobrecomprado o contra una tendencia bajista
    fuerte sale peor, y que la tendencia a favor "apoya" la operacion (y lo
    mismo, al reves, en los cortos). Aqui se abre la regla del plan (niveles de
    cada lado, sin stop por tiempo, como en `multifactor` y `rule-holdout`) la
    primera sesion tras cada cierre de mes, en los `top` nombres mas liquidos
    SIN los 50 pares de referencia, y se compara el E(R) de las entradas con la
    condicion contra el de las demas del mismo mes y lado. Las condiciones son
    las de `technical.timing_flags`, las mismas de la ficha, fijadas antes de
    mirar; nada se ajusta aqui.

    Criterio fijado antes de mirar: una afirmacion se sostiene si la diferencia
    mensual tiene el signo que afirma la ficha y su p de Newey-West, ajustado
    por Holm sobre las seis afirmaciones, es < 0,05 (`timing_table`).
    """
    import dataclasses  # noqa: PLC0415

    dates, picks = _holdout_picks(end, months, top, p)
    names = sorted(picks["ticker"].unique())
    print(f"{len(dates)} month-ends {pd.Timestamp(dates[0]):%Y-%m} to {pd.Timestamp(dates[-1]):%Y-%m}, "
          f"{len(names)} names (reference peers left out); downloading OHLC...", flush=True)
    # 200 sesiones de SMA200 y 20 de su pendiente antes de la primera entrada.
    market = loader.load_ohlcv(names, pd.Timestamp(dates[0]) - pd.DateOffset(months=13), end).ohlcv
    one = dataclasses.replace(p, history_d=1)
    rows = []
    for t, grp in picks.groupby("ticker"):
        if t not in market:
            continue
        c, h, lo, a = _series(market[t])
        ind = technical.indicator_series(c, h, lo)
        for d in grp["date"]:
            i = int(c.index.searchsorted(d, side="right"))     # primera sesion DESPUES del cierre de mes
            if i + p.horizon_d >= len(c) or not np.isfinite(ind["sma200_slope"].iloc[i]):
                continue
            state = ind.iloc[i]
            cut = slice(0, i + p.horizon_d + 1)
            for side in (1, -1):
                sims = tp.simulate_trades(c.iloc[cut], h.iloc[cut], lo.iloc[cut], a.iloc[cut], side, one,
                                          cost=COST, borrow=BORROW if side < 0 else 0.0)
                if len(sims):
                    prefix = "long" if side > 0 else "short"
                    rows.append({"date": d, "ticker": t, "side": side, "r": float(sims["r_mult"].iloc[-1]),
                                 "stopped": sims["exit"].iloc[-1] == "stop",
                                 **{f: bool(state[f"{prefix}_{f}"])
                                    for f in ("extended", "against_trend", "with_trend")}})
    df = pd.DataFrame(rows)
    print(f"{len(df) // 2:,} entries per side")
    for _, r in timing_table(df).iterrows():
        print(f"{'long ' if r['side'] > 0 else 'short'} {r['flag']:13} {r['share']:5.1%} of entries | "
              f"E(R) with {r['er_flag']:+.3f}R, without {r['er_rest']:+.3f}R | stopped {r['stop_flag']:.0%} vs "
              f"{r['stop_rest']:.0%} | monthly difference "
              f"{r['diff']:+.3f}R t {r['t']:+.2f} p {r['p']:.3f} Holm {r['p_holm']:.3f} ({r['months']} months) "
              f"-> {r['verdict']}")


def levels(end: pd.Timestamp, months: int, top: int, p) -> None:
    """Los avisos de niveles de la ficha (`technical.level_flags`), medidos.

    Mismas entradas que `timing` (870 nombres sin los 50 pares, primera sesion
    tras cada cierre de mes). En cada entrada, soportes y resistencias con
    `technical.pivots` y `nearest_levels` sobre los precios HASTA esa sesion,
    y el stop y el TP1 de la regla de cada lado. La ficha dice que un stop
    justo detras del nivel es buena colocacion, que un nivel dentro del stop
    hace que un test lo toque y que un nivel antes del TP1 frena el movimiento.

    Criterio fijado antes de mirar: el de `timing` (signo afirmado y p de
    Newey-West ajustado por Holm sobre estas seis < 0,05). Tasas de stop y de
    TP1, descriptivas.
    """
    import dataclasses  # noqa: PLC0415

    dates, picks = _holdout_picks(end, months, top, p)
    names = sorted(picks["ticker"].unique())
    print(f"{len(dates)} month-ends, {len(names)} names (reference peers left out); downloading OHLC...", flush=True)
    market = loader.load_ohlcv(names, pd.Timestamp(dates[0]) - pd.DateOffset(months=13), end).ohlcv
    one = dataclasses.replace(p, history_d=1)
    rows = []
    for t, grp in picks.groupby("ticker"):
        if t not in market:
            continue
        c, h, lo, a = _series(market[t])
        for d in grp["date"]:
            i = int(c.index.searchsorted(d, side="right"))
            if i < 130 or i + p.horizon_d >= len(c) or not np.isfinite(a.iloc[i]):
                continue
            res, sup = technical.pivots(h.iloc[: i + 1], lo.iloc[: i + 1])
            support, resistance = technical.nearest_levels(float(c.iloc[i]), res, sup)
            state = {"atr": float(a.iloc[i]), "support": support, "resistance": resistance}
            cut = slice(0, i + p.horizon_d + 1)
            for side in (1, -1):
                lv = tp.levels(float(c.iloc[i]), float(a.iloc[i]), side, p)
                flags = technical.level_flags(state, side, float(c.iloc[i]), lv["stop"], lv["tp1"])
                sims = tp.simulate_trades(c.iloc[cut], h.iloc[cut], lo.iloc[cut], a.iloc[cut], side, one,
                                          cost=COST, borrow=BORROW if side < 0 else 0.0)
                if len(sims):
                    ex = sims["exit"].iloc[-1]
                    rows.append({"date": d, "ticker": t, "side": side, "r": float(sims["r_mult"].iloc[-1]),
                                 "stopped": ex == "stop", "tp1": str(ex).startswith("tp1"), **flags})
    df = pd.DataFrame(rows)
    print(f"{len(df) // 2:,} entries per side")
    for _, r in timing_table(df, LEVEL_CLAIMS).iterrows():
        print(f"{'long ' if r['side'] > 0 else 'short'} {r['flag']:17} {r['share']:5.1%} of entries | "
              f"E(R) with {r['er_flag']:+.3f}R, without {r['er_rest']:+.3f}R | stopped {r['stop_flag']:.0%} vs "
              f"{r['stop_rest']:.0%} | TP1 {r['tp1_flag']:.0%} vs {r['tp1_rest']:.0%} | monthly difference "
              f"{r['diff']:+.3f}R t {r['t']:+.2f} p {r['p']:.3f} Holm {r['p_holm']:.3f} ({r['months']} months) "
              f"-> {r['verdict']}")


def main(argv: list[str] | None = None) -> int:
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("study", choices=["time-stop", "persistence", "multifactor", "quant-score", "short-rule",
                                      "long-rule", "rule-holdout", "timing", "levels"])
    ap.add_argument("--end", default=None, help="ultima fecha (AAAA-MM-DD); por defecto, hoy")
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--months", type=int, default=36, help="multifactor: cierres de mes evaluados")
    ap.add_argument("--top", type=int, default=600, help="multifactor: nombres mas liquidos por fecha")
    args = ap.parse_args(argv)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    if args.study in ("multifactor", "quant-score", "rule-holdout", "timing", "levels"):
        {"multifactor": multifactor, "quant-score": quant_score, "rule-holdout": rule_holdout,
         "timing": timing, "levels": levels}[args.study](end, args.months, args.top, tp.PlanParams())
        return 0
    peers = [str(t).upper() for t in load_yaml(DEFAULT_YAML)["scoring"]["reference_peers"]]
    market = loader.load_ohlcv(peers, end - pd.DateOffset(years=args.years), end).ohlcv
    p = tp.PlanParams()
    {"time-stop": time_stop, "persistence": persistence, "short-rule": short_rule,
     "long-rule": long_rule}[args.study](market, p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
