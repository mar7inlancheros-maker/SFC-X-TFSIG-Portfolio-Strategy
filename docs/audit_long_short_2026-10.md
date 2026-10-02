# Auditoría del motor long/short (octubre de 2026)

Revisión del Quantitative Long/Short Research Engine (`src/quant_engine/`,
`config/quant_engine.yaml`) frente a su especificación original de 35 secciones.
Rama `audit/long-short-review`, desde `main` en `d49f706`.

**Reglas de la auditoría:**

- Un commit por corrección.
- Ningún valor de `config/quant_engine.yaml` cambia.
- No se toca el modelo multifactor (`src/sfc_tfsig/`, `strategy.toml`).
- Los problemas de código compartido se reportan, no se editan.

**Corrida de referencia.** Todas las cifras "antes → después" salen de la misma
orden, con la caché de precios del día:

```bash
python main.py research --long AAPL,MSFT,NVDA,AMZN,META \
    --short TSLA,INTC,BA,PYPL,NKE --width 160
```

Usa los defaults del YAML: SPY, 5 años, `risk_parity`, mensual, 0,05% de coste
y 0,25% de préstamo. WRDS no estaba disponible en esta máquina (no hay pgpass),
así que todas las corridas usan la ruta de respaldo: Yahoo + ETF + SIC.

## Mapeo de la especificación

| § | Tema | Estado | Nota |
|---|---|---|---|
| 1 | No es research | OK | El Quant Score no sobrescribe la señal de research |
| 2 | Entrada | Desvío | Con Enter, la primera mitad va LONG, y la §2 lo prohíbe (#27) |
| 3 | Capa de datos | OK / Doc | Caché, calidad y calendario correctos. rf constante declarada (#31). Sesión abierta: #7 |
| 4 | Calidad de datos | OK | — |
| 5 | Retornos y riesgo por activo | OK | — |
| 6 | Beta / alpha | OK | HAC; beta móvil de 60, 120 y 252 días |
| 7–9 | Momentum, reversión, volatilidad | OK | Umbrales en el YAML y documentados |
| 10–11 | Correlación y covarianza | OK | Clusters jerárquicos, Ledoit-Wolf, número de condición |
| 12 | Modelo de factores | OK | Si falta un factor, se marca como no disponible y no se inventa |
| 13 | Cinco métodos | OK salvo #4 | — |
| 14 | Beta neutral | OK | — |
| 15 | Sector neutral | OK | — |
| 16 | Optimización | Desvío | `risk_parity` deja la neta libre (#12). La especificación pone el tope en 20%; el YAML usa 0,35 (#18) |
| 17–18 | Señales y acuerdo | OK | Texto de los pesos: #22 |
| 19 | LONG vs SHORT | OK | La mediana no se muestra en la terminal (#30) |
| 20 | Matriz de sección cruzada | OK | — |
| 21 | Modos A / B | Desvío | La trayectoria A alimentaba salidas sin rótulo y el resumen final (#8) |
| 22 | Métricas del backtest | Parcial | Los retornos anuales y mensuales no se reportan (#28) |
| 23 | Estrés | OK | — |
| 24 | Monte Carlo | OK | — |
| 25 | Robustez | Desvío | Se calcula sobre la trayectoria A (#25) |
| 26 | Sobreajuste | OK | Walk-forward |
| 27 | Terminal | OK | Añade una sección [14] TRADE PLAN (#26) |
| 28 | Exportaciones | OK | Tablas cortadas: #21 |
| 29 | Gráficos | OK | Rótulos: #8 |
| 30 | Arquitectura | OK (adaptada) | `loader.py` en lugar de downloader y cache, `inference/` en lugar de `statistics/`. La especificación permite adaptar |
| 31 | Ingeniería | OK | Log en `engine.log`; 104 tests al empezar |
| 32 | Matemática | OK salvo #4 | — |
| 33 | Menú final | OK | — |
| 34 | No recomendar | Incumple | El plan de operación (#26) |
| 35 | Proceso | n/a | — |

## Hallazgos

| # | § | Hallazgo | Evidencia | Severidad | Estado |
|---|---|---|---|---|---|
| 3 | 32 | Anualización, Sharpe, CAGR, Calmar, beta/alpha HAC, VaR/CVaR, P&L del corto, coste, préstamo, rotación, bruta y neta: correctos frente a un cálculo independiente | script con datos sintéticos | — | OK |
| 4 | 22, 32 | El Sortino del backtest usa otra definición que la del motor; la desviación a la baja usa el retorno bruto | `backtest/metrics.py:47-50` | Media | Corregido |
| 5 | 3 | Liquidez con cierre ajustado: efecto despreciable | `factors/liquidity.py:18` | Baja | OK |
| 6 | 3 | Con WRDS, la capitalización es el precio de hoy × `cshoq` del último trimestre, sin ajustar los splits posteriores | `data/wrds_data.py` | Media | Corregido (sin WRDS real) |
| 7 | 3, 16 | No se excluye la sesión en curso | `analysis.py:215`, `data/loader.py:46` | Media | Corregido |
| 8 | 21 | La trayectoria A aparecía sin rótulo y alimentaba el resumen final | ver la entrada #8 | Media | Corregido |
| 9 | 16, 26 | Estimación walk-forward sin look-ahead | `backtest/engine.py` | — | OK |
| 10 | 19 | Tests estadísticos y aviso de N = 5 | `inference/tests.py` | — | OK |
| 11 | 13 | Bruta 2,0 y tope respetado en los 5 métodos | salida | — | OK |
| 12 | 16 | `risk_parity` deja la neta libre (+0,23 hoy) | `portfolio/construction.py` | Media | Corregido |
| 13 | 14 | Beta neutral: 0,000 ex-ante, −0,035 realizada | salida | — | OK |
| 14 | 11 | Número de condición entre 24 y 26 | salida | — | OK |
| 15 | — | El plan de operación con `max_sharpe` siempre caía a `equal_weight` | `trade_plan.py:148` | Baja | Corregido |
| 16 | — | Tras los recortes, el plan queda con bruta 1,17 y neta +0,21, sin aviso | salida | Baja | Corregido |
| 17 | — | Constantes fijas en el código (`history_d`, umbrales de calidad) | `trade_plan.py:54`, `data/validation.py:22` | Baja | Para el comité |
| 18 | 16 | `max_position` = 0,35 frente al 20% de la especificación | YAML | Media | Para el comité |
| 19 | — | `borrow_cost` plano de 0,25% | YAML | Baja | Fila de sensibilidad añadida |
| 20 | 28 | Las exportaciones coinciden con la terminal | — | — | OK |
| 21 | 28 | En modo interactivo, las tablas salen cortadas en txt y html | `app.py:148,162` | Baja | Corregido |
| 22 | 17 | Los pesos se muestran nominales y redondeados | `reporting/terminal.py:210` | Baja | Corregido |
| 23 | 31 | Avisos de ruff | 3 sitios | Baja | Corregido |
| 24 | — | Código compartido: `cli.py`, techo de pandas, Sortino de `sfc_tfsig` | — | Media | Va en el PR del multifactor |
| 25 | 25, 21 | La robustez se calcula entera sobre la trayectoria A | `robustness.py:31-35` | Media | Propuesta |
| 26 | 34 | El plan de operación emite órdenes: BUY / SELL SHORT, acciones, "do not trade" | `trade_plan.py`, `reporting/terminal.py:413-458` | Alta | Propuesta |
| 27 | 2 | Enter = la primera mitad va LONG | `app.py:88-95` | Baja | Propuesta |
| 28 | 22 | Retornos anuales y mensuales sin reportar | `backtest/metrics.py:100-110` | Baja | Propuesta |
| 30 | 19 | Mediana calculada pero no mostrada | `reporting/terminal.py:254` | Baja | Propuesta |
| 31 | 3 | rf constante, sin FRED | YAML | Baja | Documentado |

## Registro de cambios

### #23 — Avisos de ruff

- **Qué estaba mal:** `ruff --select E9,F` daba tres avisos:
  - `port_daily` se calculaba y nunca se usaba (`analysis.py:359`);
  - `pandas` se importaba sin usarse (`reporting/charts.py:16`);
  - había un f-string sin placeholders (`reporting/terminal.py:474`).
- **Cambio:** se borra la línea muerta, se quita el import y el prefijo `f`.
- **Impacto:** ninguno en los resultados. `static_returns` es pura, así que la
  línea borrada no tenía efectos. Ruff queda limpio y siguen pasando los 104
  tests.

### #21 — Tablas cortadas en el modo interactivo

- **Qué estaba mal:** en el modo interactivo, `Console(record=True)` tomaba
  el ancho del terminal (`app.py:148,162`). Con 80 columnas, las tablas
  anchas salían cortadas con "…". El txt y el html se exportan de lo que
  graba la consola, así que se cortaban igual. En la corrida de la Fase 1, la
  tabla de señales era ilegible.
- **Cambio:** las cuatro consolas del modo interactivo usan
  `report_console()`, con un ancho fijo `REPORT_WIDTH = 160`. Es el mismo
  default que ya tenía el modo `research`.
- **Test:** `test_la_consola_interactiva_tiene_ancho_fijo`.
- **Impacto:** ninguno en los resultados.
  - Corrida interactiva con `COLUMNS=80`, contando "…" en el txt exportado:
    antes 40, después 0.
