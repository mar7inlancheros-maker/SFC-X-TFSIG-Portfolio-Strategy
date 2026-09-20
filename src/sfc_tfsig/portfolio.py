"""De los scores a los pesos: que se compra y cuanto. Funciones puras.

Tres decisiones separadas, en este orden:

1. **Seleccion.** Las `n_positions` mejores por score compuesto, con un
   amortiguador: una posicion que ya esta en cartera se mantiene mientras siga
   dentro del top `buffer_rank`. Sin amortiguador, una accion que oscila entre
   el puesto 30 y el 31 se compra y se vende todos los meses, y cada vuelta paga
   comisiones y spread. Con el, la senal tiene que deteriorarse de verdad para
   provocar una venta. Es de las pocas cosas que mejora el retorno neto sin
   tocar la senal.

2. **Ponderacion.** Tres esquemas, todos long-only y sin apalancamiento:
   - `equal`: 1/N. El mas robusto y el mas dificil de batir. Es el listo por
     defecto contra el que hay que justificar cualquier otro.
   - `score_tilt`: mas peso a mayor score, de forma acotada. Expresa conviccion
     sin concentrar.
   - `inverse_vol`: peso inversamente proporcional a la volatilidad. Iguala la
     contribucion al riesgo en vez del capital -- la idea de paridad de riesgo
     de Bridgewater, aplicada aqui dentro de una sola clase de activo.

3. **Restricciones.** Techo por nombre, techo por sector, suelo por nombre. El
   techo sectorial es el que impide que el modelo, siendo coherente consigo
   mismo, acabe con el 60% del libro en energia porque todo el sector cotiza
   barato a la vez. Un fondo con LPs no puede explicar eso en una carta
   trimestral.

Nada de esto mira al futuro ni al pasado: entra una seccion cruzada de una fecha
y salen pesos que suman 1 (menos el colchon de caja).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import Config


def select_names(
    scored: pd.DataFrame,
    *,
    n_positions: int,
    buffer_rank: int,
    held: set[str] | None = None,
    score_col: str = "score_composite",
    max_per_sector: int | None = None,
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Seleccion con amortiguador de rotacion y cupo por sector.

    `held` son los tickers que ya estan en cartera. Se ordena por score, se
    mantienen los que siguen dentro del top `buffer_rank`, y se rellena hasta
    `n_positions` con los mejores que no estaban.

    **El cupo sectorial se aplica AQUI, no solo en los pesos.** Si las diez
    mejores del mes son todas del mismo sector, ningun reparto de pesos puede
    respetar un techo del 30%: no hay a quien darle el exceso. Recortar pesos
    sin recortar seleccion deja la cartera con un solo sector y el tope
    incumplido en silencio, que es la peor de las tres opciones. Se limita el
    numero de nombres por sector y se rellena con los mejores de los demas.
    """
    ranked = scored.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    if ranked.empty:
        return ranked

    ranked = ranked.reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    held = held or set()
    sectors = (
        ranked[sector_col] if sector_col in ranked.columns
        else pd.Series("Unknown", index=ranked.index)
    )

    counts: dict[str, int] = {}
    chosen: list[int] = []

    def _try_take(idx: int) -> bool:
        sector = sectors.iloc[idx]
        if max_per_sector is not None and counts.get(sector, 0) >= max_per_sector:
            return False
        counts[sector] = counts.get(sector, 0) + 1
        chosen.append(idx)
        return True

    # 1) Las que ya estan en cartera y siguen dentro del amortiguador.
    for idx in ranked.index[
        ranked["ticker"].isin(held) & (ranked["rank"] <= buffer_rank)
    ]:
        if len(chosen) >= n_positions:
            break
        _try_take(idx)

    # 2) Se rellena por orden de score con las que quepan en su sector.
    for idx in ranked.index:
        if len(chosen) >= n_positions:
            break
        if idx in chosen:
            continue
        _try_take(idx)

    selection = ranked.loc[chosen]
    return selection.sort_values("rank").reset_index(drop=True)


def _raw_weights(selection: pd.DataFrame, scheme: str, score_col: str) -> pd.Series:
    n = len(selection)
    if n == 0:
        return pd.Series(dtype="float64")

    if scheme == "equal":
        return pd.Series(1.0 / n, index=selection.index)

    if scheme == "inverse_vol":
        vol = pd.to_numeric(selection.get("volatility"), errors="coerce")
        # Sin volatilidad no hay inverso: se le da el peso de la mediana en vez
        # de excluirla, porque la falta de dato es un problema de cobertura, no
        # una opinion sobre su riesgo.
        vol = vol.fillna(vol.median())
        vol = vol.where(vol > 0, vol[vol > 0].median() if (vol > 0).any() else 1.0)
        inverse = 1.0 / vol
        return inverse / inverse.sum()

    if scheme == "score_tilt":
        scores = pd.to_numeric(selection[score_col], errors="coerce")
        # Se desplaza a positivo y se suaviza: el peso crece con el score pero
        # una diferencia de score no se traduce linealmente en capital. Sin el
        # suelo, un score ligeramente negativo daria peso negativo (una venta en
        # corto que este modelo no contempla).
        shifted = scores - scores.min() + 0.25
        tilt = shifted / shifted.sum()
        equal = pd.Series(1.0 / n, index=selection.index)
        # Mezcla mitad y mitad con 1/N: conserva la conviccion y evita que tres
        # nombres se lleven medio libro.
        return 0.5 * tilt + 0.5 * equal

    raise ValueError(f"esquema de ponderacion desconocido: {scheme}")


def apply_caps(
    weights: pd.Series,
    sectors: pd.Series,
    *,
    max_weight: float,
    max_sector_weight: float,
    max_iter: int = 50,
) -> pd.Series:
    """Aplica techo por nombre y por sector. Devuelve pesos que suman <= 1.

    Recortar y redistribuir es iterativo: dar mas peso a un nombre libre puede
    empujarlo a el, o a su sector, por encima del techo. En cada vuelta se
    recorta lo que se pasa y el sobrante se reparte SOLO entre quienes tienen
    holgura en las dos dimensiones a la vez.

    **Cuando los topes son imposibles, mandan los topes y sobra caja.** Con dos
    sectores disponibles y un techo sectorial del 40%, no hay forma de invertir
    el 100%: 2 x 40% = 80%. La version anterior de esta funcion renormalizaba al
    final, lo que devolvia 50%/50% -- respetaba la suma y se saltaba el limite
    sin decirlo. Un limite de riesgo que se incumple en silencio es peor que no
    tenerlo, porque nadie lo vuelve a mirar. Ahora el resto se queda sin
    invertir y aparece como caja en el backtest y en el reporte.

    En el universo real (once sectores) esto no se activa: es una red de
    seguridad, no un modo de funcionamiento.
    """
    w = weights.astype(float).copy()
    if w.empty:
        return w
    w = w / w.sum()
    tolerance = 1e-12

    for _ in range(max_iter):
        w = w.clip(upper=max_weight)

        totals = w.groupby(sectors).sum()
        for sector, total in totals[totals > max_sector_weight + tolerance].items():
            members = sectors == sector
            w[members] *= max_sector_weight / total

        deficit = 1.0 - w.sum()
        if deficit <= 1e-10:
            break

        # Holgura de cada nombre: la menor entre la suya y la que le queda a su
        # sector repartida entre los miembros que aun pueden crecer.
        name_room = (max_weight - w).clip(lower=0.0)
        sector_room = (max_sector_weight - w.groupby(sectors).sum()).clip(lower=0.0)
        sector_room_per_name = sectors.map(sector_room).astype(float)
        can_grow = name_room > tolerance
        members_growing = can_grow.groupby(sectors).transform("sum").replace(0, np.nan)
        room = pd.concat(
            [name_room, sector_room_per_name / members_growing], axis=1
        ).min(axis=1).fillna(0.0).clip(lower=0.0)

        total_room = room.sum()
        if total_room <= tolerance:
            break  # no cabe mas: el resto queda en caja

        w = w + room / total_room * min(deficit, total_room)

    return w


def build_portfolio(
    scored: pd.DataFrame,
    cfg: Config,
    *,
    held: set[str] | None = None,
    score_col: str = "score_composite",
) -> pd.DataFrame:
    """Seccion cruzada de una fecha -> cartera objetivo.

    Devuelve ticker, sector, score, rank y `weight`. Los pesos suman
    `1 - cash_buffer`.
    """
    pf = cfg.section("portfolio")
    n_positions = int(pf["n_positions"])
    # Cupo de nombres por sector coherente con el techo de peso: con reparto
    # igualitario, `max_sector_w x n_positions` nombres agotan justo el techo.
    # Se redondea hacia arriba y nunca por debajo de 1, para no dejar sectores
    # sin representacion posible cuando el techo es pequeno.
    max_per_sector = max(1, math.ceil(float(pf["max_sector_w"]) * n_positions))

    selection = select_names(
        scored,
        n_positions=n_positions,
        buffer_rank=int(pf["buffer_rank"]),
        held=held,
        score_col=score_col,
        max_per_sector=max_per_sector,
    )
    if selection.empty:
        return selection.assign(weight=pd.Series(dtype="float64"))

    weights = _raw_weights(selection, str(pf["weighting"]), score_col)
    sectors = selection.get("sector", pd.Series("Unknown", index=selection.index))
    weights = apply_caps(
        weights,
        sectors,
        max_weight=float(pf["max_weight"]),
        max_sector_weight=float(pf["max_sector_w"]),
    )

    # Suelo por nombre: una posicion del 0.3% no mueve el resultado y si paga
    # comision. Se elimina y se reparte entre las que quedan.
    min_weight = float(pf["min_weight"])
    keep = weights >= min_weight
    if keep.any() and (~keep).any():
        weights = weights[keep]
        selection = selection.loc[weights.index]
        sectors = sectors.loc[weights.index]
        weights = apply_caps(
            weights / weights.sum(),
            sectors,
            max_weight=float(pf["max_weight"]),
            max_sector_weight=float(pf["max_sector_w"]),
        )

    invested = 1.0 - float(pf["cash_buffer"])
    out = selection.copy()
    out["weight"] = weights * invested
    columns = [c for c in ("ticker", "name", "sector", "country", "price", score_col, "rank", "weight")
               if c in out.columns]
    return out[columns].reset_index(drop=True)


def turnover(previous: pd.Series, target: pd.Series) -> float:
    """Rotacion de un rebalanceo: suma de |cambio de peso| / 2.

    Dividido entre dos para que un cambio completo de cartera sea 100% y no
    200%. Es la convencion de la industria y la que se reporta al comite.
    """
    all_names = previous.index.union(target.index)
    before = previous.reindex(all_names).fillna(0.0)
    after = target.reindex(all_names).fillna(0.0)
    return float((after - before).abs().sum() / 2.0)
