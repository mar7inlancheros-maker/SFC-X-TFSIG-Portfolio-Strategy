"""Catalogo de mutantes del modulo de riesgo: bugs plausibles que un test DEBE cazar.

Cada mutante sustituye un fragmento exacto del codigo por una version con un
error que un humano podria cometer: mirar al futuro, un signo cambiado, una
formula de libro mal copiada. `tools/risk_mutation.py` aplica cada uno sobre una
copia del repo y corre los tests indicados. Si los tests siguen pasando, el
mutante "sobrevive": hay un bug de esa clase que la suite no veria.

Reglas para mantenerlo:

- `old` tiene que aparecer UNA sola vez en el fichero. Si el codigo cambia y el
  patron deja de existir, `tests/test_risk_tools.py` falla y dice cual.
- Un mutante equivalente (que no cambia el comportamiento) no se anade: siempre
  sobreviviria y ensucia el resultado.
- Los saltos de linea se escriben como `\\n`; la herramienta normaliza CRLF.
"""

from __future__ import annotations

from dataclasses import dataclass

RISK = "src/sfc_tfsig/risk/"


@dataclass(frozen=True)
class Mutant:
    label: str
    path: str          # relativo a la raiz del repo
    old: str
    new: str
    tests: tuple[str, ...]


def _m(label: str, module: str, old: str, new: str, *tests: str) -> Mutant:
    path = module if "/" in module else RISK + module
    return Mutant(label, path, old, new, tuple(f"tests/test_risk_{t}.py" for t in tests))


MUTANTS: tuple[Mutant, ...] = (
    # -- var.py: barreras contra el look-ahead y estadistica ------------------
    _m("VaR historico sin shift (look-ahead)", "var.py",
       "historical = -r.rolling(window).quantile(p).shift(1)",
       "historical = -r.rolling(window).quantile(p)", "var"),
    _m("FHS sin shift (look-ahead)", "var.py",
       "fhs = -standardized.rolling(window).quantile(p).shift(1) * sigma",
       "fhs = -standardized.rolling(window).quantile(p) * sigma", "var"),
    _m("EWMA ve el retorno del dia", "var.py",
       "variance[t] = lam * variance[t - 1] + (1.0 - lam) * values[t - 1] ** 2",
       "variance[t] = lam * variance[t - 1] + (1.0 - lam) * values[min(t, len(values) - 1)] ** 2",
       "var"),
    _m("Kupiec con la cola cambiada", "var.py",
       "log_null = _xlogy(n - x, 1.0 - tail) + _xlogy(x, tail)",
       "log_null = _xlogy(n - x, tail) + _xlogy(x, 1.0 - tail)", "var"),
    _m("Christoffersen con la transicion equivocada", "var.py",
       "pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0",
       "pi01 = n11 / (n00 + n01) if (n00 + n01) else 0.0", "var"),
    _m("Basilea con el corte verde al 99%", "var.py",
       "if cumulative < 0.95:", "if cumulative < 0.99:", "var"),
    _m("Cornish-Fisher con la asimetria al reves", "var.py",
       "return (z + (z ** 2 - 1) * skew / 6", "return (z - (z ** 2 - 1) * skew / 6", "var"),
    _m("VaR normal escalado con h y no con raiz de h", "var.py",
       "normal_v = -(mu * h + z * sd * math.sqrt(h))", "normal_v = -(mu * h + z * sd * h)", "var"),
    _m("ES historico con la cola equivocada", "var.py",
       "return float(-r[r <= cutoff].mean())", "return float(-r[r >= cutoff].mean())", "var"),

    # -- exposure.py ------------------------------------------------------------
    _m("Euler mal repartido", "exposure.py",
       "component = w * marginal", "component = w * marginal * 1.01", "exposure"),
    _m("Beta con la varianza de la cartera", "exposure.py",
       "betas = cross / var_b if var_b > 0", "betas = cross / var_p if var_b > 0", "exposure"),
    _m("Tracking error con el signo de la covarianza cambiado", "exposure.py",
       "te_var = var_p - 2.0 * float(w @ cross) + var_b",
       "te_var = var_p + 2.0 * float(w @ cross) + var_b", "exposure"),
    _m("Contraccion sin tope en 1", "exposure.py",
       "delta = float(min(1.0, max(0.0, kappa / t)))", "delta = float(max(0.0, kappa / t))",
       "exposure"),
    _m("HHI sobre pesos con caja", "exposure.py",
       "normalized = w / invested", "normalized = w", "exposure"),
    _m("Capacidad sin dividir por el peso", "exposure.py",
       "capacity = max_days * daily / w.where(w > 0)", "capacity = max_days * daily", "exposure"),
    _m("Ratio de diversificacion invertido", "exposure.py",
       "diversification_ratio=float(w @ stdev) / vol_p if vol_p > 0",
       "diversification_ratio=vol_p / float(w @ stdev) if vol_p > 0", "exposure"),

    # -- garch.py ---------------------------------------------------------------
    _m("GARCH ve el residuo del dia", "garch.py",
       "out[t] = omega + alpha * eps[t - 1] ** 2 + beta * out[t - 1]",
       "out[t] = omega + alpha * eps[t] ** 2 + beta * out[t - 1]", "garch"),
    _m("GARCH sin condicion de estacionariedad", "garch.py",
       "keep = (a > 0) & (b >= 0) & (a + b < 0.999)", "keep = (a > 0) & (b >= 0)", "garch"),
    _m("Reversion a la media con un exponente de mas", "garch.py",
       "p ** k * (first - fit.long_run_var)", "p ** (k + 5) * (first - fit.long_run_var)", "garch"),
    _m("Simulacion GARCH sin racimos de volatilidad", "garch.py",
       "variance = fit.omega + fit.alpha * eps ** 2 + fit.beta * variance",
       "variance = fit.omega + fit.alpha * fit.sigma_next ** 2 + fit.beta * variance", "garch"),
    _m("Vida media con la formula invertida", "garch.py",
       "return math.log(0.5) / math.log(p) if 0.0 < p < 1.0",
       "return math.log(p) / math.log(0.5) if 0.0 < p < 1.0", "garch"),

    # -- montecarlo.py ----------------------------------------------------------
    _m("Descuento con el signo cambiado", "montecarlo.py",
       "data[:, 0] -= haircut_annual / TRADING_DAYS", "data[:, 0] += haircut_annual / TRADING_DAYS",
       "montecarlo"),
    _m("Bootstrap sin bloques", "montecarlo.py",
       "restart = rng.random(n_paths) < restart_prob", "restart = rng.random(n_paths) < 1.0",
       "montecarlo"),
    _m("t de Student sin reescalar la covarianza", "montecarlo.py",
       "t_scale = np.sqrt((t_dof - 2.0) / chi2)[:, None]", "t_scale = np.sqrt(t_dof / chi2)[:, None]",
       "montecarlo"),
    _m("Drawdown medido sin actualizar el maximo", "montecarlo.py",
       "peak = np.maximum(peak, wealth[:, 0])", "peak = peak", "montecarlo"),
    _m("Racha bajo el agua que nunca se reinicia", "montecarlo.py",
       "run = np.where(wealth[:, 0] < peak, run + 1, 0)", "run = run + 1", "montecarlo"),
    _m("Probabilidad de quedar detras con la comparacion invertida", "montecarlo.py",
       "float((strategy < paths[:, 1]).mean())", "float((strategy > paths[:, 1]).mean())",
       "montecarlo"),

    # -- stress.py ----------------------------------------------------------------
    _m("Choque hipotetico sin corte en -100%", "stress.py",
       "shocks = (beta * market + sector_shock + factor_shock).clip(lower=-1.0)",
       "shocks = (beta * market + sector_shock + factor_shock)", "stress"),
    _m("Precio viejo aceptado como del dia", "stress.py",
       "> max_stale_days)", "> 10_000)", "stress"),
    _m("Relativo historico contra todo el benchmark, no la parte invertida", "stress.py",
       '"benchmark": market,\n            "relative": float(pnl.sum()) - market * float(w.sum()),',
       '"benchmark": market,\n            "relative": float(pnl.sum()) - market,', "stress"),
    _m("Prueba inversa sin la parte invertida", "stress.py",
       "needed = -float(loss) / exposure if exposure and exposure > 0",
       "needed = -float(loss) / portfolio_beta if exposure and exposure > 0", "stress"),
    _m("VaR estresado en la ventana mas tranquila", "stress.py",
       '"stressed_var": float(rolling.max()),', '"stressed_var": float(rolling.min()),', "stress"),
    _m("Correlacion de crisis hacia cero en vez de hacia uno", "stress.py",
       "corr = (1.0 - corr_blend) * corr + corr_blend * np.ones_like(corr)",
       "corr = (1.0 - corr_blend) * corr", "stress"),
    _m("Peso aproximado sin ponderar", "stress.py",
       '(proxied * w).sum(axis=1).rename("weight_proxied")',
       'proxied.mean(axis=1).rename("weight_proxied")', "stress"),

    # -- factor_model.py -------------------------------------------------------
    _m("Modelo factorial con el panel del mismo dia", "factor_model.py",
       'side="left") - 1', 'side="right") - 1', "factor_model"),
    _m("Descomposicion factorial sin dividir por la varianza total", "factor_model.py",
       '"variance_share": x.to_numpy() * fx / total_var if total_var > 0',
       '"variance_share": x.to_numpy() * fx / factor_var if total_var > 0', "factor_model"),
    _m("Riesgo especifico olvidado en la descomposicion", "factor_model.py",
       'groups["specific"] = specific_var / total_var if total_var > 0',
       'groups["specific"] = 0.0 if total_var > 0', "factor_model"),

    # -- history.py ------------------------------------------------------------
    _m("Calibracion con precios futuros", "history.py",
       "visible = close_wide.loc[close_wide.index <= date]", "visible = close_wide", "history"),
    _m("Sesgo medido con la media y no con la dispersion", "history.py",
       "bias = float(z.std(ddof=1))", "bias = float(z.abs().mean())", "history"),
    _m("Beta movil con la varianza de la estrategia", "history.py",
       'out["beta"] = cov / b.rolling(window).var()', 'out["beta"] = cov / r.rolling(window).var()',
       "history"),

    # -- liquidity.py ----------------------------------------------------------
    _m("Impacto lineal en vez de raiz cuadrada", "liquidity.py",
       "return coef * daily_vol * np.sqrt(ratio)", "return coef * daily_vol * ratio", "liquidity"),
    _m("Coste fijo que no se encarece en crisis", "liquidity.py",
       "fixed = cost_bps / 10_000.0 * spread_mult", "fixed = cost_bps / 10_000.0", "liquidity"),
    _m("Capacidad sin el cuadrado de la raiz", "liquidity.py",
       "return float((budget / unit) ** 2)", "return float(budget / unit)", "liquidity"),

    # -- limits.py ---------------------------------------------------------------
    _m("Limite en el borde cuenta como excedido", "limits.py",
       "if value > limit:", "if value >= limit:", "limits"),
    _m("Limite minimo sin zona de alerta", "limits.py",
       'return "WARNING" if value < limit / warn else "OK"', 'return "OK"', "limits"),

    # -- analysis.py -------------------------------------------------------------
    _m("Limite de VaR medido con la normal", "analysis.py",
       '(book_var["method"] == "mc_t_student")', '(book_var["method"] == "mc_normal")', "analysis"),
    _m("Analisis completo ignora la historia larga del estres", "analysis.py",
       "_set_book_history(result, stress_close, risk_cfg, confidences, seed)",
       "(lambda *a: None)(result, stress_close, risk_cfg, confidences, seed)", "analysis"),
    _m("Pesos objetivo en vez de derivados", "analysis.py",
       'value = last["shares"].astype(float) * prices',
       'value = last["weight"].astype(float) * float(nav.iloc[-1])', "analysis"),
    _m("Drawdown actual con el signo cambiado", "analysis.py",
       "result.current_drawdown = float(-drawdown_series(nav_series).iloc[-1])",
       "result.current_drawdown = float(drawdown_series(nav_series).iloc[-1])", "analysis"),
    _m("Liquidacion de crisis con el volumen normal", "analysis.py",
       'adv_mult=float(liq_cfg["stress_adv_mult"]),', "adv_mult=1.0,", "analysis"),

    # -- segunda ronda: bordes, unidades y lo que lee el comite ---------------
    _m("Horizonte multi-dia promediado en vez de compuesto", "var.py",
       "return np.expm1(log.rolling(horizon_d).sum()).dropna()",
       "return np.expm1(log.rolling(horizon_d).mean()).dropna()", "var"),
    _m("Dias bajo -3 sigma contados a -2 sigma", "var.py",
       "threshold = mu - 3.0 * sd", "threshold = mu - 2.0 * sd", "var"),
    _m("Ventana de covarianza con un dia de menos", "exposure.py",
       "prices = visible.reindex(columns=tickers).tail(window + 1)",
       "prices = visible.reindex(columns=tickers).tail(window)", "exposure"),
    _m("Volumen diario con la media y no la mediana", "exposure.py",
       "return dollar.tail(window).median()", "return dollar.tail(window).mean()", "exposure"),
    _m("Nombre sin beta que no se mueve con el mercado", "stress.py",
       "beta = betas.reindex(names).fillna(1.0)", "beta = betas.reindex(names).fillna(0.0)", "stress"),
    _m("Peor ventana con el inicio desplazado un dia", "stress.py",
       "start = nav.index[nav.index.get_loc(end) - h]",
       "start = nav.index[nav.index.get_loc(end) - h + 1]", "stress"),
    _m("Retorno anual mediano con el exponente invertido", "montecarlo.py",
       '"median_annualized": float((1.0 + np.median(strategy)) ** (1.0 / years) - 1.0)',
       '"median_annualized": float((1.0 + np.median(strategy)) ** years - 1.0)', "montecarlo"),
    _m("Capacidad que opera la rotacion una sola vez", "liquidity.py",
       "order = 2.0 * turnover * w * float(nav)", "order = turnover * w * float(nav)", "liquidity"),
    _m("Uso de un limite minimo calculado al reves", "limits.py",
       "usage = limit / value if value else", "usage = value / limit if value else", "limits"),
    _m("El 99% del limite no entra si la politica no lo pide", "analysis.py",
       'return sorted({float(c) for c in risk_cfg.get("var.confidence")} | {0.99})',
       'return sorted({float(c) for c in risk_cfg.get("var.confidence")})', "analysis"),
    _m("Columna de VaR en dolares sin multiplicar por el NAV", "src/sfc_tfsig/risk_report.py",
       '_pct(r["var"]),\n            _money(r["var"] * nav),',
       '_pct(r["var"]),\n            _money(r["var"]),', "report", "analysis"),
    _m("Politica de un limite minimo mostrada como maximo", "src/sfc_tfsig/risk_report.py",
       'sign = "<=" if row["direction"] == "max" else ">="', 'sign = "<="', "report"),
    _m("Escenario con inicio igual al fin aceptado", "src/sfc_tfsig/config.py",
       "if start >= end:", "if start > end:", "config"),
    _m("Correlacion de crisis fuera de rango aceptada", "src/sfc_tfsig/config.py",
       '_fraction(sc.get("corr_blend", 0.0), f"{where}.corr_blend")',
       'float(sc.get("corr_blend", 0.0))', "config"),
)
