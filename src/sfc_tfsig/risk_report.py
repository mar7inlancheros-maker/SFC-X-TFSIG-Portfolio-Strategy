"""Reporte de riesgo en Markdown, para el comite. Y el resumen previo a operar.

Mismos principios que `report.py`:

1. **Todo numero lleva su contexto.** Un VaR sin su backtest, o un Monte Carlo
   sin el descuento que se le aplico, no informa.
2. **Porcentaje y dolares.** El comite decide en dolares: "VaR 99% = 2.7%" se
   discute distinto que "5.400 USD en uno de cada cien dias".
3. **El semaforo va primero.** Lo que exige una decision se lee antes que lo
   que la explica.

Salida en ASCII puro: el reporte se redirige a fichero y Windows cambia de
codificacion al hacerlo.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .metrics import TRADING_DAYS
from .paths import REPORT_DIR
from .risk.analysis import RiskAnalysis

_METHOD_LABELS = {
    "historico": "Historico",
    "normal": "Normal",
    "cornish_fisher": "Cornish-Fisher",
    "fhs": "FHS (vol de hoy)",
    "ewma_normal": "Normal con vol EWMA",
    "historico_cartera": "Historico, cartera actual (1 ano)",
    "mc_normal": "Monte Carlo normal",
    "mc_t_student": "Monte Carlo t de Student",
}

_LIMITATIONS = """\
## Limitaciones del analisis de riesgo

1. **El VaR no es la peor perdida.** Es donde empieza la cola. El ES dice cuanto
   pesa en media, y ni siquiera el ES acota lo que no esta en la muestra.
2. **El bootstrap no inventa crisis.** Remuestrea dias reales del backtest, que
   empieza en 2012: no contiene 2008. Las pruebas de estres existen para eso.
3. **El backtest hereda el sesgo de supervivencia.** Por eso el bootstrap se
   corre con el descuento anual declarado en `config/risk.toml`. Sin el, las
   probabilidades de perder saldrian optimistas.
4. **Covarianza de un ano.** El riesgo ex-ante supone que el proximo mes se
   parece al ultimo ano. En un cambio de regimen, las correlaciones suben
   justo cuando mas importa; la t de Student y el estres cubren parte de eso.
5. **Estres lineal.** Los escenarios hipoteticos y la prueba inversa usan la
   beta y choques declarados: ordenes de magnitud, no previsiones. Los
   nombres sin precios en un episodio historico se aproximan por su beta.
6. **Los limites vigilan, no restringen.** Un limite excedido no cambia la
   cartera: se lleva al comite.
"""


# ---------------------------------------------------------------------------
#  Formato
# ---------------------------------------------------------------------------


def _pct(value, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/d"
    return f"{value * 100:.{decimals}f}%"


def _num(value, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/d"
    return f"{value:,.{decimals}f}"


def _money(value) -> str:
    if value is None or not np.isfinite(value):
        return "n/d"
    return f"{value:,.0f}"


def _limit_value(value: float, unit: str) -> str:
    if unit == "pct":
        return _pct(value)
    if unit == "days":
        return _num(value, 2)
    return _num(value, 2)


def _rows(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


# ---------------------------------------------------------------------------
#  Secciones
# ---------------------------------------------------------------------------


def limits_section(analysis: RiskAnalysis) -> str:
    rows = []
    for _, row in analysis.limits.iterrows():
        sign = "<=" if row["direction"] == "max" else ">="
        rows.append([
            row["label"],
            _limit_value(row["value"], row["unit"]),
            f"{sign} {_limit_value(row['threshold'], row['unit'])}",
            _pct(row["usage"], 0),
            row["status"],
            row["action"],
        ])
    lines = [f"## Semaforo de limites: {analysis.status}", ""]
    lines += _rows(["Limite", "Valor", "Politica", "Uso", "Estado", "Que hacer"], rows)
    lines += [
        "",
        "ALERTA salta al llegar al umbral de aviso de `config/risk.toml`. Un",
        "limite EXCEDIDO no cambia la cartera: se escala al comite, que decide",
        "si actua o lo acepta por escrito.",
        "",
    ]
    return "\n".join(lines) + "\n"


def ex_ante_section(analysis: RiskAnalysis, top: int = 10) -> str:
    ea, conc = analysis.ex_ante, analysis.concentration
    nav = analysis.nav
    rows = [
        ["Volatilidad ex-ante anual", _pct(ea.volatility)],
        [f"Volatilidad {analysis.benchmark}", _pct(ea.benchmark_volatility)],
        ["Beta", _num(ea.beta)],
        ["Tracking error ex-ante", _pct(ea.tracking_error)],
        ["Correlacion con el benchmark", _num(ea.correlation)],
        ["Riesgo sistematico (fraccion de la varianza)", _pct(ea.systematic_share, 1)],
        ["Ratio de diversificacion", _num(ea.diversification_ratio)],
        [f"VaR normal {ea.confidence:.0%} 1 dia", f"{_pct(ea.var_normal_1d)} ({_money(ea.var_normal_1d * nav)} USD)"],
        ["Contraccion Ledoit-Wolf", _pct(ea.shrinkage, 1)],
        ["Posiciones", str(conc.get("n_names", "n/d"))],
        ["Invertido / caja", f"{_pct(conc.get('invested'), 1)} / {_pct(conc.get('cash'), 1)}"],
        ["Mayor peso", f"{conc.get('max_weight_name', '')} {_pct(conc.get('max_weight'))}"],
        ["Cinco mayores pesos", _pct(conc.get("top5_weight"))],
        ["Numero efectivo de nombres", _num(conc.get("effective_names"), 1)],
        ["Mayor sector", f"{conc.get('max_sector', '')} {_pct(conc.get('max_sector_weight'))}"],
    ]
    lines = ["## Cartera actual: riesgo ex-ante", "",
             "Covarianza del ultimo ano de retornos diarios, contraida hacia",
             "correlacion constante (Ledoit-Wolf). Pesos derivados a precio de hoy,",
             "no los objetivo del ultimo rebalanceo.", ""]
    lines += _rows(["Medida", "Valor"], rows)

    contrib = ea.contributions.head(top)
    lines += ["", f"### Contribucion al riesgo: {top} mayores", ""]
    lines += _rows(
        ["Ticker", "Sector", "Peso", "Volatilidad", "Beta", "Contribucion", "Del riesgo total"],
        [[t, r["sector"], _pct(r["weight"]), _pct(r["volatility"], 1), _num(r["beta"]),
          _pct(r["contribution"]), _pct(r["risk_share"], 1)] for t, r in contrib.iterrows()],
    )
    lines += ["", "### Por sector", ""]
    lines += _rows(
        ["Sector", "Peso", "Del riesgo total", "Nombres"],
        [[s, _pct(r["weight"]), _pct(r["risk_share"], 1), int(r["n_names"])]
         for s, r in ea.sectors.iterrows()],
    )
    lines += [
        "",
        "Cuando la fraccion de riesgo de un nombre o sector supera claramente su",
        "peso, el tope de peso no lo esta conteniendo: aporta volatilidad o",
        "correlacion por encima de lo que su tamano sugiere.",
        "",
    ]

    fx = analysis.factor_exposure
    if fx is not None and not fx.empty:
        lines += ["### Exposicion a factores (z-score medio ponderado)", ""]
        lines += _rows(
            ["Factor", "Cartera", "Universo", "Activa", "Peso con dato"],
            [[f, _num(r["portfolio"]), _num(r["universe"]), _num(r["active"]),
              _pct(r["weight_covered"], 0)] for f, r in fx.iterrows()],
        )
        lines += ["",
                  "Una exposicion activa alta en momentum es el riesgo que el escenario",
                  "'Crash de momentum' pone a prueba.", ""]
    return "\n".join(lines) + "\n"


_METHOD_ORDER = {m: i for i, m in enumerate(_METHOD_LABELS)}


def _var_rows(table: pd.DataFrame, nav: float) -> list[list[str]]:
    rows = []
    ordered = table.assign(_order=table["method"].map(_METHOD_ORDER))
    for _, r in ordered.sort_values(["horizon_d", "confidence", "_order"]).iterrows():
        rows.append([
            _METHOD_LABELS.get(r["method"], r["method"]),
            f"{int(r['horizon_d'])}d",
            f"{r['confidence']:.0%}",
            _pct(r["var"]),
            _money(r["var"] * nav),
            _pct(r["es"]),
            _money(r["es"] * nav),
        ])
    return rows


def var_section(analysis: RiskAnalysis, top: int = 10) -> str:
    nav = analysis.nav
    headers = ["Metodo", "Horizonte", "Confianza", "VaR", "VaR USD", "ES", "ES USD"]
    lines = [
        "## VaR y Expected Shortfall",
        "",
        "Perdida positiva en fraccion del NAV y en dolares al NAV de hoy",
        f"({_money(nav)} USD). VaR 99% a 1 dia = perdida que se supera uno de",
        "cada cien dias; ES = perdida media de esos dias.",
        "",
        "### Cartera actual",
        "",
    ]
    lines += _rows(headers, _var_rows(analysis.book_var, nav))
    lines += [
        "",
        "La t de Student y la normal comparten covarianza: la diferencia entre",
        "ambas es la prima de cola gorda que un VaR normal no ve.",
        "",
    ]
    if analysis.strategy_var is not None and not analysis.strategy_var.empty:
        lines += ["### Estrategia: historia del backtest", ""]
        lines += _rows(headers, _var_rows(analysis.strategy_var, nav))
        lines += [
            "",
            "FHS reescala la cola historica a la volatilidad EWMA de hoy: si sale",
            "por encima del historico, el mercado esta hoy mas nervioso que la",
            "media de la muestra. Es el VaR de referencia.",
            "",
        ]

    comp = analysis.component_es
    if comp is not None and not comp.empty:
        lines += ["### Quien explica la cola: ES 99% 1 dia por componentes (t de Student)", ""]
        lines += _rows(
            ["Ticker", "Sector", "Peso", "ES aportado", "USD", "Del ES"],
            [[r["ticker"], r["sector"], _pct(r["weight"]), _pct(r["component_es"], 3),
              _money(r["component_es"] * nav), _pct(r["es_share"], 1)]
             for _, r in comp.head(top).iterrows()],
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def backtest_section(analysis: RiskAnalysis) -> str:
    table = analysis.var_backtest
    if table is None or table.empty:
        return ""
    recent = int(table["recent_obs"].iloc[0])
    lines = ["## Backtest del VaR", "",
             "Cada dia se reestima el VaR con la informacion del cierre anterior y",
             "se compara con la perdida real.", ""]
    lines += _rows(
        ["Modelo", "Dias", "Esperadas", "Excepciones", "Tasa", "Kupiec p",
         "Christoffersen p", "Cobertura condicional p", f"Ultimos {recent} dias", "Basilea"],
        [[_METHOD_LABELS.get(r["model"], r["model"]), int(r["n_obs"]), _num(r["expected"], 1),
          int(r["exceptions"]), _pct(r["rate"]), _num(r["kupiec_p"], 3),
          _num(r["christoffersen_p"], 3), _num(r["conditional_p"], 3),
          int(r["recent_exceptions"]), r["zone"]] for _, r in table.iterrows()],
    )
    lines += [
        "",
        "- Kupiec: p < 0.05 rechaza que la frecuencia de excepciones sea la",
        "  prometida. Falla tanto un VaR que se excede mucho como uno que no se",
        "  excede nunca (capital parado).",
        "- Christoffersen: p < 0.05 indica excepciones en racimo. El modelo no",
        "  reacciona a los cambios de volatilidad y las perdidas llegan juntas.",
        "- Basilea: verde, amarillo o rojo por la probabilidad binomial de ver",
        "  tantas excepciones con un modelo correcto.",
        "",
    ]
    return "\n".join(lines) + "\n"


def tail_section(analysis: RiskAnalysis) -> str:
    tail = analysis.tail
    if not tail:
        return ""
    rows = [
        ["Dias", str(tail["n_days"])],
        ["Asimetria", _num(tail["skew"])],
        ["Curtosis en exceso", _num(tail["excess_kurtosis"])],
        ["Peor dia", f"{_pct(tail['worst_day'])} ({pd.Timestamp(tail['worst_day_date']).date()})"],
        ["Mejor dia", f"{_pct(tail['best_day'])} ({pd.Timestamp(tail['best_day_date']).date()})"],
        ["Dias por debajo de -3 sigma", f"{_pct(tail['freq_below_3sigma'], 3)} "
                                        f"(normal: {_pct(tail['normal_freq_below_3sigma'], 3)})"],
        ["Ratio de colas (p95 / |p5|)", _num(tail["tail_ratio"])],
    ]
    lines = ["## Colas de la distribucion diaria", ""]
    lines += _rows(["Medida", "Valor"], rows)
    lines += ["", "Curtosis en exceso por encima de 3, o dias bajo -3 sigma varias veces",
              "mas frecuentes que en la normal, invalidan el VaR normal como estimacion.", ""]
    return "\n".join(lines) + "\n"


def bootstrap_section(analysis: RiskAnalysis) -> str:
    summary = analysis.bootstrap_summary
    if summary is None or summary.empty:
        return ""
    boot = analysis.bootstrap
    nav = analysis.nav
    horizons = list(summary.index)
    header = ["Medida"] + [f"{summary.loc[h, 'years']:.0f} anos" for h in horizons]

    def line(label: str, column: str, fmt) -> list[str]:
        return [label] + [fmt(summary.loc[h, column]) if column in summary else "n/d" for h in horizons]

    rows = [
        line("Retorno acumulado p5", "p5", _pct),
        line("Retorno acumulado p25", "p25", _pct),
        line("Retorno acumulado mediano", "p50", _pct),
        line("Retorno acumulado p75", "p75", _pct),
        line("Retorno acumulado p95", "p95", _pct),
        line("Retorno anual mediano", "median_annualized", _pct),
        line(f"Mediana {analysis.benchmark}", "benchmark_p50", _pct),
        line("Probabilidad de perder dinero", "prob_loss", lambda v: _pct(v, 1)),
        line(f"Probabilidad de quedar detras de {analysis.benchmark}", "prob_underperform",
             lambda v: _pct(v, 1)),
        # Un VaR negativo significa que incluso el percentil 5 gana dinero: se
        # dice con palabras, porque "VaR = -9.9%" se lee como una perdida.
        line("VaR 95% del horizonte", "var_95", lambda v: _pct(v) if v > 0 else "sin perdida"),
        line("VaR 95% del horizonte, USD", "var_95",
             lambda v: _money(v * nav) if v > 0 else "sin perdida"),
        line("ES 95% del horizonte", "es_95", _pct),
        line("Drawdown maximo mediano", "median_max_drawdown", _pct),
        line("Drawdown maximo p95", "p95_max_drawdown", _pct),
    ]
    for column in [c for c in summary.columns if c.startswith("prob_dd_")]:
        rows.append(line(f"Probabilidad de drawdown >= {column.removeprefix('prob_dd_')}%",
                         column, lambda v: _pct(v, 1)))
    n_paths = boot.terminal[horizons[0]].shape[0] if boot else 0
    lines = [
        "## Monte Carlo: bootstrap de la historia",
        "",
        f"{n_paths:,} trayectorias por bootstrap estacionario de bloques sobre los",
        "retornos diarios reales de la estrategia y el benchmark, con los mismos",
        "indices para ambos. Descuento aplicado a la estrategia antes de remuestrear:",
        f"{_pct(boot.haircut if boot else float('nan'))} anual (sesgo de supervivencia declarado).",
        "",
    ]
    lines += _rows(header, rows)
    lines += [
        "",
        "El drawdown maximo p95 es el que hay que estar dispuesto a aguantar: una",
        "de cada veinte trayectorias lo supera. Si el comite no lo aguantaria, la",
        "estrategia es demasiado arriesgada para el mandato, con independencia de",
        "su retorno esperado.",
        "",
    ]
    return "\n".join(lines) + "\n"


def stress_section(analysis: RiskAnalysis) -> str:
    nav = analysis.nav
    lines = ["## Pruebas de estres", ""]

    hist = analysis.historical
    if hist is not None and not hist.empty:
        lines += ["### Historicas: la cartera de hoy en episodios reales", ""]
        rows = []
        for _, r in hist.iterrows():
            if r.get("status") != "ok":
                rows.append([r["scenario"], r["start"], r["end"], r.get("status", "sin datos")]
                            + [""] * 6)
                continue
            rows.append([
                r["scenario"], r["start"], r["end"], _pct(r["portfolio"]), _pct(r["benchmark"]),
                _pct(r["relative"]), _money(r["portfolio"] * nav), _pct(r["weight_proxied"], 0),
                f"{r['worst_name']} ({_pct(r['worst_contribution'])})", _pct(r["strategy_then"]),
            ])
        lines += _rows(["Episodio", "Desde", "Hasta", "Cartera", analysis.benchmark, "Relativo",
                        "P&L USD", "Peso aproximado por beta", "Peor nombre", "Estrategia entonces"],
                       rows)
        lines += [
            "",
            "'Peso aproximado por beta': fraccion de la cartera sin precios en el",
            "episodio (no cotizaba), movida con su beta. Por encima del 50%, el",
            "escenario mide la beta de la cartera, no sus nombres. 'Estrategia",
            "entonces': lo que perdio el backtest con la cartera que tenia ese dia.",
            "",
        ]

    hyp = analysis.hypothetical
    if hyp is not None and not hyp.empty:
        lines += ["### Hipoteticas", ""]
        lines += _rows(
            ["Escenario", "Mercado", "Cartera", "Relativo", "P&L USD", "Por beta",
             "Por sector", "Por factor", "Peor nombre"],
            [[r["scenario"], _pct(r["market"]), _pct(r["portfolio"]), _pct(r["relative"]),
              _money(r["portfolio"] * nav), _pct(r["from_beta"]), _pct(r["from_sectors"]),
              _pct(r["from_factors"]), f"{r['worst_name']} ({_pct(r['worst_contribution'])})"]
             for _, r in hyp.iterrows()],
        )
        lines += [
            "",
            "Choque por nombre = beta x mercado + extra de su sector + choque de",
            "factor x su z-score. Los extras son juicio declarado en",
            "`config/risk.toml`, no una calibracion.",
            "",
        ]

    rev = analysis.reverse
    if rev is not None and not rev.empty:
        lines += ["### Inversa: que caida del mercado produce cada perdida", ""]
        lines += _rows(
            ["Perdida de la cartera", "USD", f"Caida necesaria de {analysis.benchmark}"],
            [[_pct(r["loss"], 0), _money(r["loss"] * nav),
              _pct(r["market_move"], 1) if r["plausible"] else "no alcanzable via beta"]
             for _, r in rev.iterrows()],
        )
        lines += ["", f"Lineal en la beta ex-ante ({_num(analysis.ex_ante.beta)}) y sin riesgo",
                  "especifico: un orden de magnitud, no una prevision.", ""]

    worst = analysis.worst_windows
    if worst is not None and not worst.empty:
        lines += ["### Peores ventanas de la estrategia en el backtest", ""]
        lines += _rows(
            ["Sesiones", "Desde", "Hasta", "Estrategia", analysis.benchmark],
            [[int(r["window_d"]), r["start"], r["end"], _pct(r["strategy"]),
              _pct(r.get("benchmark"))] for _, r in worst.iterrows()],
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def liquidity_section(analysis: RiskAnalysis, top: int = 5) -> str:
    s = analysis.liquidity_summary
    lines = ["## Liquidez y capacidad", ""]
    lines += _rows(["Medida", "Valor"], [
        ["Dias para liquidar la posicion menos liquida",
         f"{_num(s.get('max_days'))} ({s.get('max_days_name', '')})"],
        ["Fraccion vendible en 1 sesion", _pct(s.get("liquid_1d"), 1)],
        ["Fraccion vendible en 5 sesiones", _pct(s.get("liquid_5d"), 1)],
        ["Capacidad (NAV maximo al limite de dias)",
         f"{_money(s.get('capacity_nav'))} USD ({s.get('capacity_name', '')})"],
        ["Peso sin volumen medible", _pct(s.get("weight_without_adv"), 1)],
    ])
    table = analysis.liquidity.dropna(subset=["days_to_liquidate"]).head(top)
    if not table.empty:
        lines += ["", f"### {top} posiciones menos liquidas", ""]
        lines += _rows(
            ["Ticker", "Peso", "Valor USD", "Volumen diario USD", "Dias"],
            [[t, _pct(r["weight"]), _money(r["value"]), _money(r["adv"]),
              _num(r["days_to_liquidate"])] for t, r in table.iterrows()],
        )
    lines += ["",
              "La capacidad es el NAV al que la posicion menos liquida tardaria el",
              "limite de dias en venderse. Por encima, los costes fijos del backtest",
              "dejan de ser creibles y habria que modelar impacto de mercado.", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
#  Reporte completo y resumen previo a operar
# ---------------------------------------------------------------------------


def build_risk_report(analysis: RiskAnalysis, cfg: Config, risk_cfg: Config) -> str:
    """Reporte de riesgo completo en Markdown."""
    header = [
        f"# Reporte de riesgo -- {cfg.get('meta.name')}",
        "",
        f"- Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Cartera al: {analysis.as_of.date()} | NAV {_money(analysis.nav)} "
        f"{cfg.get('meta.base_currency')} | benchmark {analysis.benchmark}",
        f"- Estrategia: `{cfg.source.name if cfg.source else 'en memoria'}` "
        f"(fingerprint `{cfg.fingerprint}`)",
        f"- Politica de riesgo: `{risk_cfg.source.name if risk_cfg.source else 'en memoria'}` "
        f"(fingerprint `{risk_cfg.fingerprint}`)",
        f"- Estado general: **{analysis.status}**",
        "",
        "> Dos reportes de riesgo son comparables si coinciden los DOS",
        "> fingerprints. Cambiar la politica de riesgo no cambia la cartera,",
        "> pero si los numeros de este reporte.",
        "",
    ]
    sections = [
        "\n".join(header),
        limits_section(analysis),
        ex_ante_section(analysis),
        var_section(analysis),
        backtest_section(analysis),
        tail_section(analysis),
        bootstrap_section(analysis),
        stress_section(analysis),
        liquidity_section(analysis),
    ]
    if analysis.warnings:
        unique = list(dict.fromkeys(analysis.warnings))
        sections.append("## Avisos\n\n" + "\n".join(f"- {w}" for w in unique) + "\n")
    sections.append(_LIMITATIONS)
    return "\n".join(s for s in sections if s)


def render_pre_trade(analysis: RiskAnalysis) -> str:
    """Resumen de riesgo de la cartera OBJETIVO, para la hoja de ordenes.

    Se mira antes de operar, no despues: si la cartera propuesta excede un
    limite, el comite lo sabe cuando todavia puede no aprobarla.
    """
    ea = analysis.ex_ante
    worst = (analysis.hypothetical.sort_values("portfolio").iloc[0]
             if not analysis.hypothetical.empty else None)
    lines = [
        f"## Riesgo de la cartera objetivo: {analysis.status}",
        "",
        f"- Politica de riesgo: fingerprint `{analysis.risk_fingerprint}`",
        f"- Volatilidad ex-ante {_pct(ea.volatility)} | beta {_num(ea.beta)} | "
        f"tracking error {_pct(ea.tracking_error)}",
        f"- VaR 99% 1 dia (t de Student): {_pct(analysis.measures.get('var_99_1d'))} "
        f"({_money(analysis.measures.get('var_99_1d', float('nan')) * analysis.nav)} USD)",
        f"- Numero efectivo de nombres: {_num(analysis.concentration.get('effective_names'), 1)}",
    ]
    if worst is not None:
        lines.append(f"- Peor escenario hipotetico: {worst['scenario']}, {_pct(worst['portfolio'])} "
                     f"({_money(worst['portfolio'] * analysis.nav)} USD)")
    lines.append("")
    flagged = analysis.limits[analysis.limits["status"].isin(["EXCEDIDO", "ALERTA"])]
    if flagged.empty:
        lines.append("Todos los limites con dato en verde. El drawdown no se evalua antes de operar.")
    else:
        lines += _rows(
            ["Limite", "Valor", "Politica", "Estado", "Que hacer"],
            [[r["label"], _limit_value(r["value"], r["unit"]),
              f"{'<=' if r['direction'] == 'max' else '>='} {_limit_value(r['threshold'], r['unit'])}",
              r["status"], r["action"]] for _, r in flagged.iterrows()],
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
#  Ficheros
# ---------------------------------------------------------------------------


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def save_risk_report(text: str, name: str = "riesgo") -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{name}_{_stamp()}.md"
    path.write_text(text, encoding="utf-8")
    return path


def save_risk_artifacts(analysis: RiskAnalysis, name: str = "riesgo") -> dict[str, Path]:
    """Tablas del reporte en CSV, para recalcular o auditar cualquier cifra."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    frames = {
        "limites": analysis.limits,
        "contribuciones": analysis.ex_ante.contributions.reset_index(),
        "var_cartera": analysis.book_var,
        "var_estrategia": analysis.strategy_var,
        "backtest_var": analysis.var_forecasts,
        "montecarlo": analysis.bootstrap_summary,
        "estres_historico": analysis.historical_detail,
        "estres_hipotetico": analysis.hypothetical_detail,
        "liquidez": analysis.liquidity,
    }
    paths: dict[str, Path] = {}
    for label, frame in frames.items():
        if frame is None or frame.empty:
            continue
        path = REPORT_DIR / f"{name}_{stamp}_{label}.csv"
        keep_index = label in ("backtest_var", "montecarlo", "liquidez")
        frame.to_csv(path, index=keep_index)
        paths[label] = path
    return paths


def save_risk_charts(analysis: RiskAnalysis, name: str = "riesgo") -> Path | None:
    """Abanico del Monte Carlo, backtest del VaR y riesgo frente a peso por sector."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{name}_{_stamp()}_graficos.png"
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    ax_fan, ax_var, ax_sector = axes

    boot = analysis.bootstrap
    if boot is not None and not boot.fan.empty:
        fan = boot.fan
        years = fan.index.to_numpy() / TRADING_DAYS
        ax_fan.fill_between(years, fan["p5"], fan["p95"], alpha=0.2, color="steelblue", label="p5-p95")
        ax_fan.fill_between(years, fan["p25"], fan["p75"], alpha=0.35, color="steelblue", label="p25-p75")
        ax_fan.plot(years, fan["p50"], color="navy", linewidth=1.4, label="mediana")
        ax_fan.axhline(1.0, color="grey", linewidth=0.8)
        ax_fan.set_xlabel("anos")
        ax_fan.set_ylabel("riqueza (1 = hoy)")
        ax_fan.set_title("Monte Carlo: bootstrap de la historia")
        ax_fan.legend(loc="upper left")
        ax_fan.grid(alpha=0.3)

    forecasts = analysis.var_forecasts
    if forecasts is not None and not forecasts.empty:
        recent = forecasts.tail(750)
        ax_var.plot(recent.index, recent["return"] * 100, linewidth=0.6, color="grey", label="retorno diario")
        ax_var.plot(recent.index, -recent["fhs"] * 100, color="crimson", linewidth=1.0, label="-VaR FHS")
        hits = recent[recent["return"] < -recent["fhs"]]
        ax_var.scatter(hits.index, hits["return"] * 100, color="crimson", s=14, zorder=3, label="excepciones")
        ax_var.set_ylabel("%")
        ax_var.set_title("Backtest del VaR (ultimos 3 anos)")
        ax_var.legend(loc="lower left")
        ax_var.grid(alpha=0.3)

    sectors = analysis.ex_ante.sectors
    if not sectors.empty:
        x = np.arange(len(sectors))
        ax_sector.bar(x - 0.2, sectors["weight"] * 100, width=0.4, label="peso")
        ax_sector.bar(x + 0.2, sectors["risk_share"] * 100, width=0.4, label="riesgo")
        ax_sector.set_xticks(x)
        ax_sector.set_xticklabels(sectors.index, rotation=30, ha="right")
        ax_sector.set_ylabel("%")
        ax_sector.set_title("Peso frente a contribucion al riesgo, por sector")
        ax_sector.legend()
        ax_sector.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
