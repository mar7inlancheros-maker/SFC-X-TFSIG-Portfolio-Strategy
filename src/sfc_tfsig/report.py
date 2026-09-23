"""Reporte del modelo en texto plano (Markdown), para el comite.

Tres principios:

1. **Todo numero lleva su contexto.** Un CAGR sin el del benchmark y sin el
   drawdown no informa: informa de que alguien quiere que el resultado parezca
   bueno.
2. **Las limitaciones van EN el reporte**, no en un anexo que nadie abre. Si el
   universo tiene sesgo de supervivencia, eso se lee antes que el Sharpe.
3. **El fingerprint de la configuracion va en la cabecera.** Un reporte sin el
   no se puede reproducir y no deberia discutirse en un comite.

Salida en ASCII puro y `enable_utf8_stdout()` en los puntos de entrada: los
reportes se redirigen a fichero constantemente y Windows cambia de codificacion
al hacerlo.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping

import pandas as pd

from .backtest import BacktestResult
from .config import Config
from .metrics import Performance, yearly_returns
from .paths import REPORT_DIR

_LIMITATIONS = """\
## Limitaciones declaradas

Estas no son advertencias de cortesia: cada una desplaza el resultado en una
direccion conocida, y el comite deberia descontarlas antes de decidir nada.

1. **Sesgo de supervivencia.** El universo se construye con las cotizadas que
   existen HOY. Las que quebraron o fueron excluidas no estan. Esto INFLA el
   retorno historico; la magnitud tipica en renta variable estadounidense es de
   1 a 2 puntos de CAGR al ano. No se corrige sin datos de pago.

2. **Cobertura canadiense parcial.** Los fundamentales salen del XBRL de la SEC,
   que cubre a los emisores canadienses inscritos (40-F / 20-F / 10-K) y no a
   los exclusivos de TSX. El tramo canadiense del universo es, por tanto, el de
   las grandes cotizadas con presencia en EE.UU.

3. **Clasificacion sectorial por SIC.** El SIC es de 1987 y no distingue bien el
   software moderno ni el comercio electronico. La neutralizacion sectorial
   hereda ese ruido.

4. **Sin impacto de mercado.** Los costes modelados son comision, spread y
   slippage fijos. A escala de capital simulado son razonables; a escala
   institucional no lo serian.

5. **Fundamentales como primera publicacion.** Se usa el dato tal como se
   presento, sin reexpresiones posteriores. Es lo correcto para evitar
   look-ahead, y significa que algunos datos historicos son los "equivocados"
   -- los mismos que tenia el mercado ese dia.
"""


def _fmt_pct(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{value * 100:.{decimals}f}%"


def _fmt_num(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{value:.{decimals}f}"


def _fmt_money(value: float) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{value:,.0f}"


def _table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Tabla Markdown sin dependencias externas."""
    if df is None or len(df) == 0:
        return "_(sin datos)_\n"
    frame = df.copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(
                lambda v: "" if pd.isna(v) else floatfmt.format(v)
            )
    headers = [str(frame.index.name or "")] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for index, row in frame.iterrows():
        lines.append("| " + " | ".join([str(index)] + [str(v) for v in row]) + " |")
    return "\n".join(lines) + "\n"


def performance_section(perf: Performance, cfg: Config) -> str:
    benchmark = cfg.get("backtest.benchmark")
    rows = [
        ("Periodo", f"{perf.start.date()} a {perf.end.date()} ({perf.years:.1f} anos)"),
        ("Retorno total", _fmt_pct(perf.total_return)),
        ("CAGR", _fmt_pct(perf.cagr)),
        (f"CAGR {benchmark}", _fmt_pct(perf.benchmark_cagr)),
        ("Exceso anual", _fmt_pct(perf.excess_cagr)),
        ("Volatilidad", _fmt_pct(perf.volatility)),
        ("Sharpe", _fmt_num(perf.sharpe)),
        ("Sortino", _fmt_num(perf.sortino)),
        ("Maximo drawdown", _fmt_pct(perf.max_drawdown)),
        (f"Maximo drawdown {benchmark}", _fmt_pct(perf.benchmark_max_drawdown)),
        ("Dias bajo el agua", str(perf.underwater_days) if perf.underwater_days else "n/d"),
        ("Calmar", _fmt_num(perf.calmar)),
        ("Meses positivos", _fmt_pct(perf.hit_rate_monthly)),
        ("Beta", _fmt_num(perf.beta)),
        ("Alfa anual", _fmt_pct(perf.alpha)),
        ("Tracking error", _fmt_pct(perf.tracking_error)),
        ("Information ratio", _fmt_num(perf.information_ratio)),
    ]
    lines = ["## Rendimiento", "", "| Metrica | Valor |", "|---|---|"]
    lines += [f"| {label} | {value} |" for label, value in rows]
    return "\n".join(lines) + "\n"


def costs_section(result: BacktestResult, perf: Performance) -> str:
    if result.rebalances.empty:
        return "## Costes y rotacion\n\n_(sin rebalanceos)_\n"
    rebalances = result.rebalances
    total_cost = result.total_costs
    initial = float(rebalances["nav"].iloc[0])
    # Rebalanceos por ano REALES, no 12 fijo. Con el 12 cableado, una corrida
    # trimestral reportaba cuatro veces su rotacion anual y parecia la opcion
    # cara justo cuando es la barata.
    per_year = len(rebalances) / perf.years if perf.years else float("nan")
    annual_turnover = float(rebalances["turnover"].sum() / perf.years) if perf.years else float("nan")
    # Arrastre en puntos de NAV: suma de lo que cada rebalanceo le costo al NAV
    # de ESE momento. Dividir los dolares totales entre el capital inicial -- la
    # version anterior -- inflaba la cifra, porque los costes de 2025 se pagan
    # sobre un NAV varias veces mayor que el de 2012. Esta es la que se resta
    # directamente del CAGR.
    annual_cost = (
        float(rebalances["cost_bps_of_nav"].sum() / perf.years / 10_000.0)
        if perf.years else float("nan")
    )
    lines = [
        "## Costes y rotacion",
        "",
        "| Metrica | Valor |",
        "|---|---|",
        f"| Rebalanceos | {len(rebalances)} ({_fmt_num(per_year, 1)} por ano) |",
        f"| Rotacion media por rebalanceo | {_fmt_pct(result.average_turnover)} |",
        f"| Rotacion anualizada | {_fmt_pct(annual_turnover)} |",
        f"| Arrastre anual de costes (puntos de NAV) | {_fmt_pct(annual_cost)} |",
        f"| Coste total | {_fmt_money(total_cost)} |",
        f"| Coste sobre capital inicial | {_fmt_pct(total_cost / initial if initial else float('nan'))} |",
        f"| Coste medio por rebalanceo (bps de NAV) | {_fmt_num(rebalances['cost_bps_of_nav'].mean(), 1)} |",
        f"| Posiciones medias | {_fmt_num(rebalances['n_positions'].mean(), 1)} |",
        "",
        "El coste no es un detalle contable: con la rotacion de arriba, una",
        "diferencia de 10 bps por operacion cambia el CAGR de forma visible.",
        "Si la estrategia solo funciona con costes optimistas, no funciona.",
        "",
    ]
    return "\n".join(lines) + "\n"


def holdings_section(result: BacktestResult, top: int = 40) -> str:
    if result.holdings.empty:
        return "## Cartera actual\n\n_(sin posiciones)_\n"
    last_date = result.holdings["date"].max()
    current = result.holdings[result.holdings["date"] == last_date].copy()
    current = current.sort_values("weight", ascending=False).head(top)
    current["weight"] = current["weight"].map(lambda v: _fmt_pct(v))
    current["price"] = current["price"].map(lambda v: f"{v:,.2f}")
    current["score"] = current["score"].map(lambda v: _fmt_num(v))
    columns = ["ticker", "sector", "weight", "price", "score"]
    lines = [f"## Cartera al {pd.Timestamp(last_date).date()}", "",
             "| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in current.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")

    sector_weights = (
        result.holdings[result.holdings["date"] == last_date]
        .groupby("sector")["weight"].sum().sort_values(ascending=False)
    )
    lines += ["", "### Exposicion sectorial", "", "| Sector | Peso |", "|---|---|"]
    lines += [f"| {sector} | {_fmt_pct(weight)} |" for sector, weight in sector_weights.items()]
    return "\n".join(lines) + "\n"


def validation_section(validation: Mapping[str, object]) -> str:
    if not validation:
        return ""
    lines = ["## Validacion estadistica", ""]

    ic = validation.get("ic_summary") or {}
    if ic:
        lines += [
            "### Coeficiente de informacion (IC de rangos, mensual)",
            "",
            "| Metrica | Valor |",
            "|---|---|",
            f"| IC medio | {_fmt_num(ic.get('mean'), 4)} |",
            f"| t (Newey-West) | {_fmt_num(ic.get('t_stat'))} |",
            f"| valor p | {_fmt_num(ic.get('p_value'), 4)} |",
            f"| Meses con IC positivo | {_fmt_pct(ic.get('hit_rate'))} |",
            f"| Meses evaluados | {ic.get('n', 'n/d')} |",
            "",
            "Referencia: un factor util en renta variable da un IC medio de 0.02",
            "a 0.05. Por encima de 0.10 lo primero que hay que buscar es una fuga",
            "de datos futuros, no una explicacion economica.",
            "",
        ]

    means = validation.get("quantile_means")
    if means is not None and len(means):
        lines += ["### Retorno medio mensual por quintil (Q1 = mejor score)", "",
                  "| Quintil | Retorno medio |", "|---|---|"]
        for label, value in means.items():
            lines.append(f"| {label} | {_fmt_pct(value)} |")
        spread = validation.get("spread_test") or {}
        lines += [
            "",
            f"Spread Q1-Q5: {_fmt_pct(spread.get('mean'))} mensual, "
            f"t = {_fmt_num(spread.get('t_stat'))}, p = {_fmt_num(spread.get('p_value'), 4)}.",
            f"Monotonicidad: {_fmt_num(validation.get('monotonicity'))} "
            "(-1.00 seria orden perfecto de Q1 a Q5).",
            "",
        ]

    fm = validation.get("fama_macbeth")
    if isinstance(fm, pd.DataFrame) and not fm.empty:
        lines += ["### Fama-MacBeth por factor (errores Newey-West)", "",
                  "| Factor | Coef. medio | t | p |", "|---|---|---|---|"]
        for term, row in fm.iterrows():
            lines.append(
                f"| {term} | {_fmt_num(row['mean'], 5)} | {_fmt_num(row['t_stat'])} | "
                f"{_fmt_num(row['p_value'], 4)} |"
            )
        lines.append("")

    wf = validation.get("walk_forward")
    if isinstance(wf, pd.DataFrame) and not wf.empty:
        lines += ["### Walk-forward: IC dentro y fuera de muestra", "",
                  "| Ventana | IC entrenamiento | IC prueba | Meses |", "|---|---|---|---|"]
        for _, row in wf.iterrows():
            lines.append(
                f"| {row['window']} | {_fmt_num(row['train_ic'], 4)} | "
                f"{_fmt_num(row['test_ic'], 4)} | {int(row['test_months'])} |"
            )
        lines += [
            "",
            "La columna que importa es la de prueba. Si el IC de entrenamiento",
            "es alto y el de prueba ronda cero, el modelo memoriza en vez de",
            "predecir, y el backtest completo no es evidencia de nada.",
            "",
        ]
    return "\n".join(lines) + "\n"


def attribution_section(contributions: pd.DataFrame, yearly_excess: pd.DataFrame | None) -> str:
    """Amplitud del resultado: proceso o posiciones afortunadas.

    Va en el reporte y no en un analisis aparte porque es la primera pregunta
    que un LP hace ante un buen ano: cuantos nombres lo hicieron.
    """
    from . import attribution as attr

    if contributions is None or contributions.empty:
        return ""

    summary = attr.concentration_summary(contributions)
    lines = [
        "## Atribucion",
        "",
        "### Amplitud en toda la muestra",
        "",
        "| Metrica | Valor |",
        "|---|---|",
        f"| Nombres tenidos alguna vez | {summary['n_names_ever_held']} |",
        f"| Nombres con contribucion positiva | {_fmt_pct(summary['pct_names_positive'])} |",
        f"| Peso de los 5 mejores en el bruto | {_fmt_pct(summary['top5_share'])} |",
        f"| Peso de los 10 mejores en el bruto | {_fmt_pct(summary['top10_share'])} |",
        f"| Nombres necesarios para la mitad | {_fmt_num(summary['names_for_half'], 0)} |",
        f"| Bruto sin los 5 mejores | {_fmt_pct(summary['total_without_top5'])} "
        f"de {_fmt_pct(summary['total_gross_contribution'])} |",
        "",
        "Si quitar los cinco mejores nombres de toda la historia hace desaparecer",
        "el resultado, el modelo no tiene un proceso: tuvo cinco aciertos.",
        "",
    ]

    if yearly_excess is not None and not yearly_excess.empty:
        lines += [
            "### Exceso anual y su amplitud",
            "",
            "| Ano | Estrategia | Benchmark | Exceso | Top-5 del bruto | Nombres para la mitad |",
            "|---|---|---|---|---|---|",
        ]
        for year, row in yearly_excess.iterrows():
            lines.append(
                f"| {year} | {_fmt_pct(row['strategy'])} | {_fmt_pct(row['benchmark'])} | "
                f"{_fmt_pct(row['excess'])} | {_fmt_pct(row.get('top5_share'))} | "
                f"{_fmt_num(row.get('names_for_half'), 0)} |"
            )
        excess = yearly_excess["excess"].dropna()
        lines += [
            "",
            f"Anos con exceso positivo: {int((excess > 0).sum())} de {len(excess)}. "
            f"Exceso mediano: {_fmt_pct(excess.median())}.",
            "",
            "Una media positiva con mediana negativa es la firma de un resultado",
            "que depende de pocos anos. Un ano de gran exceso con top-5 alto y",
            "pocos nombres para la mitad no se puede prometer que se repita.",
            "",
        ]

    best = attr.top_names(contributions, n=8)
    if not best.empty:
        lines += ["### Mejores y peores nombres, toda la muestra", "",
                  "| Ticker | Sector | Contribucion | Periodos | Peso medio |",
                  "|---|---|---|---|---|"]
        for ticker, row in best.iterrows():
            lines.append(
                f"| {ticker} | {row['sector']} | {_fmt_pct(row['contribution'])} | "
                f"{int(row['periods_held'])} | {_fmt_pct(row['avg_weight'])} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def decay_section(by_lag: pd.DataFrame | None, by_horizon: pd.DataFrame | None) -> str:
    """Vida util de la senal: lo que decide la frecuencia de rebalanceo."""
    from .validation import rebalance_signal_cost

    if (by_lag is None or by_lag.empty) and (by_horizon is None or by_horizon.empty):
        return ""

    lines = ["## Decaimiento de la senal", ""]
    if by_lag is not None and not by_lag.empty:
        lines += [
            "### IC con la senal retrasada k meses (retorno del mes siguiente)",
            "",
            "| Retraso | IC medio | t | p | Frente a fresca |",
            "|---|---|---|---|---|",
        ]
        for lag, row in by_lag.iterrows():
            lines.append(
                f"| {lag} | {_fmt_num(row['ic_mean'], 4)} | {_fmt_num(row['t_stat'])} | "
                f"{_fmt_num(row['p_value'], 4)} | {_fmt_pct(row['vs_fresh'])} |"
            )
        lines += [
            "",
            "Senal conservada segun la frecuencia de rebalanceo:",
            "",
            f"- Mensual: {_fmt_pct(rebalance_signal_cost(by_lag, 1))}",
            f"- Bimestral: {_fmt_pct(rebalance_signal_cost(by_lag, 2))}",
            f"- Trimestral: {_fmt_pct(rebalance_signal_cost(by_lag, 3))}",
            "",
            "Rebalancear menos ahorra coste solo si la senal sigue viva. Si el",
            "trimestral conserva casi toda la senal, la rotacion mensual es",
            "peaje sin contrapartida.",
            "",
        ]
    if by_horizon is not None and not by_horizon.empty:
        lines += [
            "### IC por horizonte del retorno",
            "",
            "| Horizonte (meses) | IC medio | t (Newey-West, rezagos >= h) | IC por mes |",
            "|---|---|---|---|",
        ]
        for h, row in by_horizon.iterrows():
            lines.append(
                f"| {h} | {_fmt_num(row['ic_mean'], 4)} | {_fmt_num(row['t_stat'])} | "
                f"{_fmt_num(row['ic_per_month'], 4)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def build_report(
    result: BacktestResult,
    perf: Performance,
    cfg: Config,
    *,
    benchmark_nav: pd.Series | None = None,
    validation: Mapping[str, object] | None = None,
    universe_summary: Mapping[str, object] | None = None,
    contributions: pd.DataFrame | None = None,
    yearly_excess: pd.DataFrame | None = None,
    decay_lag: pd.DataFrame | None = None,
    decay_horizon: pd.DataFrame | None = None,
) -> str:
    """Reporte completo en Markdown."""
    title = cfg.get("meta.name")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    header = [
        f"# {title}",
        "",
        f"- Generado: {generated}",
        f"- Configuracion: `{cfg.source.name if cfg.source else 'en memoria'}` "
        f"(fingerprint `{cfg.fingerprint}`)",
        f"- Universo: {cfg.get('universe.exchanges')} | "
        f"rebalanceo {cfg.get('calendar.rebalance')} | "
        f"{cfg.get('portfolio.n_positions')} posiciones | "
        f"ponderacion `{cfg.get('portfolio.weighting')}`",
        f"- Pesos de factores: {cfg.factor_weights}",
        f"- Costes: {cfg.total_cost_bps:.0f} bps por lado",
        "",
        "> Dos resultados con el mismo fingerprint son comparables. Con",
        "> fingerprint distinto, no lo son: ha cambiado alguna decision de",
        "> inversion entre una corrida y otra.",
        "",
    ]

    sections = ["\n".join(header)]

    if universe_summary:
        sections.append(
            "## Universo\n\n"
            f"- Nombres en la ultima fecha: {universe_summary.get('n_names', 'n/d')}\n"
            f"- Sectores representados: {universe_summary.get('n_sectors', 'n/d')}\n"
            f"- Por pais: {universe_summary.get('by_country', {})}\n"
        )

    sections.append(performance_section(perf, cfg))

    annual = yearly_returns(result.nav, benchmark_nav)
    if not annual.empty:
        sections.append("## Retorno ano a ano\n\n" + _table(annual, "{:.2%}"))

    sections.append(costs_section(result, perf))
    sections.append(holdings_section(result))

    if contributions is not None:
        sections.append(attribution_section(contributions, yearly_excess))

    if validation:
        sections.append(validation_section(validation))

    if decay_lag is not None or decay_horizon is not None:
        sections.append(decay_section(decay_lag, decay_horizon))

    if result.warnings:
        unique = list(dict.fromkeys(result.warnings))[:20]
        sections.append(
            "## Avisos del motor\n\n"
            + "\n".join(f"- {w}" for w in unique)
            + ("\n\n_(mostrando los primeros 20)_\n" if len(result.warnings) > 20 else "\n")
        )

    sections.append(_LIMITATIONS)
    return "\n".join(sections)


def save_report(text: str, name: str = "backtest") -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = REPORT_DIR / f"{name}_{stamp}.md"
    path.write_text(text, encoding="utf-8")
    return path


def save_artifacts(result: BacktestResult, name: str = "backtest") -> dict[str, Path]:
    """Guarda NAV, operaciones, posiciones y rebalanceos para poder auditarlos.

    Un reporte sin los datos que lo sustentan obliga a creerselo. Con estos
    ficheros, cualquiera del equipo puede recalcular las metricas por su cuenta.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    paths: dict[str, Path] = {}

    nav_path = REPORT_DIR / f"{name}_{stamp}_nav.csv"
    result.nav.rename("nav").to_csv(nav_path, index_label="date")
    paths["nav"] = nav_path

    for label, frame in (("trades", result.trades), ("holdings", result.holdings),
                         ("rebalances", result.rebalances)):
        if frame is None or frame.empty:
            continue
        path = REPORT_DIR / f"{name}_{stamp}_{label}.csv"
        frame.to_csv(path, index=False)
        paths[label] = path

    return paths


def save_charts(result: BacktestResult, benchmark_nav: pd.Series | None = None,
                name: str = "backtest") -> Path | None:
    """Curva de NAV y drawdown. Opcional: requiere matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    from .metrics import drawdown_series

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = REPORT_DIR / f"{name}_{stamp}_curva.png"

    fig, (ax_nav, ax_dd) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax_nav.plot(result.nav.index, result.nav.to_numpy(), label="Estrategia", linewidth=1.4)
    if benchmark_nav is not None and len(benchmark_nav):
        aligned = benchmark_nav.reindex(result.nav.index).ffill()
        ax_nav.plot(aligned.index, aligned.to_numpy(), label="Benchmark",
                    linewidth=1.1, alpha=0.75)
    ax_nav.set_yscale("log")
    ax_nav.set_ylabel("NAV (escala log)")
    ax_nav.legend(loc="upper left")
    ax_nav.grid(alpha=0.3)

    dd = drawdown_series(result.nav)
    ax_dd.fill_between(dd.index, dd.to_numpy() * 100, 0, alpha=0.4, color="crimson")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.grid(alpha=0.3)

    fig.suptitle("Curva de capital y drawdown")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
