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
  - Antes: la salida de la corrida interactiva de la Fase 1 tenía 429 "…".
  - Después: con `COLUMNS=80`, el txt exportado tiene 0.

### #22 — Pesos del Quant Score mostrados como nominales

- **Qué estaba mal:** la sección [8] imprimía los pesos del YAML. Sin WRDS,
  los componentes `value`, `quality`, `analyst` y `short_interest` no tienen
  datos, y `quant_score` renormaliza sobre el resto. El reporte decía
  "momentum 20%" cuando en realidad pesaba el 40%. Además, `:.0%` convertía
  el 2,5% en "2%" (`reporting/terminal.py:210`).
- **Cambio:**
  - Nueva función `composite.effective_weights`: los pesos del YAML,
    renormalizados sobre los componentes que tienen algún dato.
  - Nueva función `terminal.weights_line`: muestra "YAML → efectivo" con un
    decimal y lista aparte los componentes sin datos.
- **Tests:**
  - `test_pesos_efectivos_excluyen_componentes_sin_datos`;
  - `test_linea_de_pesos_muestra_decimales_y_los_inactivos`.
- **Impacto:** el score no cambia; solo cambia el texto.
  - Antes: `momentum 20%, value 15%, … liquidity 2%, statistical 2%`.
  - Después: `momentum 20.0% -> 40.0%, … liquidity 2.5% -> 5.0%`, más la
    lista de componentes sin datos (`value, quality, analyst, short_interest`).

### #16 — Plan de operación no neutral sin aviso

- **Qué estaba mal:** el plan parte de la cartera reconstruida (bruta 2,0,
  neta 0), pero tres cosas la mueven sin respetar las patas:
  - REDUCE recorta el nombre a la mitad;
  - NO TRADE y FLIP sacan el nombre o lo cambian de pata;
  - el tope de riesgo por operación recorta los nombres de stop ancho.

  En la corrida de referencia, el libro final quedaba en bruta 1,17 y neta
  +0,21, y solo se veía en una línea de cifras.
- **Cambio:** nueva función `trade_plan.book_warnings`. Avisa si la neta se
  aleja más de 0,05 del objetivo o si la bruta queda más de 0,05 por debajo.
  - Los avisos salen en la sección [14] y en la lista final de avisos.
  - No se redistribuye nada: el plan sigue igual y ahora lo dice.
- **Tests:**
  - `test_aviso_si_el_plan_no_queda_neutral`;
  - `test_sin_aviso_si_el_plan_cumple`.
- **Impacto:** ninguno en los resultados. Aparecen dos avisos nuevos:
  - "net +0.21 vs target +0.00";
  - "gross 1.17 vs target 2.00".

### #8 — Trayectoria del Modo A: rótulo en todas las salidas y fuera del resumen

Decisión del comité: el Modo A se mantiene, pero cumpliendo la sección 21. La
trayectoria va rotulada como hipotética en todas las salidas y no alimenta el
resumen final como evidencia.

- **Qué estaba mal:** solo la sección [10] y el gráfico 1 avisaban de que la
  cesta de hoy se aplica al pasado. Lo demás salía sin rótulo:
  - [11] RISK ANALYSIS: CAGR, Sharpe, drawdown, VaR y operativa del Modo A;
  - el backtest beta neutral de [10];
  - los episodios históricos de [12], con los pesos de hoy;
  - el bootstrap de Monte Carlo, que remuestrea la trayectoria A;
  - la robustez de [13] (ver #25);
  - los gráficos de drawdown, volatilidad móvil y cestas;
  - en el JSON, `portfolios.*.summary` y `robustness`.

  Además, el resumen final citaba el Sharpe y el drawdown del Modo A y el
  rango de Sharpe de la robustez (`reporting/terminal.py:472,488`).
- **Cambio:**
  - Un único rótulo, `MODE_A_LABEL`, definido en `backtest/engine.py`, junto
    a la definición del modo.
  - Terminal (y por tanto txt y html): rótulo en [11] y [13] y etiqueta
    "Mode A, hypothetical" en el beta neutral, los episodios y Monte Carlo.
  - Resumen final: se quitan las dos líneas del Modo A. Se añade una línea
    que dice que esas métricas no son evidencia y quedan fuera, y otra con el
    Modo B (CAGR, Sharpe, drawdown, IR) cuando hay señales con fecha.
  - Gráficos 1, 2, 3 y 7: título rotulado y nota al pie "hypothetical, not
    evidence".
  - JSON: añade `mode_a_notice`, `path` en cada cartera y en `robustness`,
    `historical_path` en `stress` y `bootstrap_source` en `montecarlo`. Añade
    también `mode_b_historical`, que antes no se exportaba.
  - CSV: ninguno lleva la trayectoria. Se revisaron las columnas:
    - `portfolio_*.csv` son pesos actuales;
    - `signals`, `risk` y `correlation` son datos por activo;
    - `trade_plan` es el plan.

    No hay nada que rotular.
- **Qué se deja en el resumen y por qué:** el spread diario LONG − SHORT con
  su p-valor (Newey-West) también usa las cestas de hoy. Se mantiene, porque
  es la comparación que pide la sección 19 y la sección 34 lo pone de ejemplo
  ("the statistical difference between the LONG and SHORT baskets has a
  p-value of X%"). Sigue rotulado "in-sample".
- **Tests** (`tests/test_qe_reporting.py`):
  - `test_resumen_final_no_usa_la_trayectoria_del_modo_a`;
  - `test_resumen_final_cita_el_modo_b_cuando_existe`;
  - `test_analisis_de_riesgo_rotula_el_modo_a`;
  - `test_json_rotula_cada_trayectoria_del_modo_a`.
- **Impacto:** ningún número cambia. El resumen pierde dos líneas:
  - "Sharpe was 1.26 with max drawdown −17.3%";
  - "Sharpe across robustness variants ranges 0.82 to 1.40".

### #7 — La sesión en curso entraba como si fuera un cierre

- **Qué estaba mal:**
  - `as_of` era `Timestamp.today()` (`analysis.py:215`) y la descarga pedía
    hasta hoy + 1 (`data/loader.py:46`). Con el mercado abierto, Yahoo
    devuelve una barra intradía para hoy, y el motor la trataba como un
    cierre.
  - Esa barra quedaba 24 horas en caché, así que una corrida a las 12:00
    seguía viendo el precio de las 12:00 por la tarde y al día siguiente.
  - Afectaba al precio de entrada, al ATR, a los stops, al último retorno, al
    VaR y al último día del backtest.
- **Cambio:**
  - Nueva función `loader.last_complete_session(now)`: la última sesión cuya
    barra ya es definitiva, a partir de las 16:15 de Nueva York (salta
    fines de semana). `as_of` sale de ahí.
  - `load_ohlcv` corta en `<= end` todo lo que viene de la caché o de Yahoo.
  - Nuevo registro `qe_ohlcv_fetched` con la hora de cada descarga. Una caché
    escrita antes de las 16:15 del día `end` lleva la barra intradía, así que
    se vuelve a pedir. Las cachés antiguas, sin registro, se piden una vez.
  - La sección [1] dice hasta qué sesión llegan los datos.
- **Tests** (`tests/test_qe_loader.py`):
  - `test_ultima_sesion_completa`, con 4 casos: viernes abierto, recién
    cerrado, cerrado y sábado;
  - `test_la_barra_de_la_sesion_en_curso_no_entra`;
  - `test_cache_bajada_antes_del_cierre_se_vuelve_a_pedir`;
  - `test_cache_bajada_tras_el_cierre_se_reutiliza`.
- **Impacto.** Antes de las corridas se respaldó la caché `qe_ohlcv` fuera
  del repo. Corrida "antes" con el código de `e45334c`, con caché vacía y el
  mercado abierto (2 de octubre, 12:02 de Nueva York). Corrida "después" con
  el código nuevo, sobre la caché que dejó la anterior:

  | | Antes (barra intradía del 2-oct) | Después (cierre del 1-oct) |
  |---|---|---|
  | Última sesión | 2026-10-02 (intradía) | 2026-10-01 |
  | Sharpe `risk_parity` (Modo A) | 1,271 | 1,263 |
  | CAGR `risk_parity` (Modo A) | 33,61% | 33,36% |
  | Sharpe `min_variance` (Modo A) | 1,405 | 1,395 |
  | Rango de Sharpe en robustez | 0,863 – 1,405 | 0,815 – 1,395 |
  | Entrada / stop AAPL | 332,65 / 315,07 | 330,32 / 312,20 |
  | Entrada / stop NKE | 33,22 / 36,06 | 35,15 / 37,60 |
  | Entrada / stop TSLA | 372,78 / 403,84 | 354,11 / 383,67 |
  | Acciones NKE en el plan | 3.514 | 4.077 |
  | Neta del plan | +0,247 | +0,211 |

  "Después" coincide cifra a cifra con la corrida de referencia de `d49f706`.
  Esa corrida se hizo con la caché escrita la noche anterior, después del
  cierre. Es decir: el resultado ya no depende de la hora de la corrida. Dos
  corridas seguidas tras el cambio dan una salida idéntica (0 líneas de
  diferencia).

### #4 — Sortino del backtest con otra definición

- **Qué estaba mal:** `backtest/metrics.py:47-50` calculaba el Sortino de
  carteras y backtests con `sfc_tfsig.metrics.sortino`. Esa función usa la
  desviación estándar de solo los excesos negativos. El motor declara otra
  definición en `factors/performance.py:8-13`: la semidesviación del exceso,
  promediada sobre todas las sesiones. Las secciones [3] (activos) y [10]-[11]
  (carteras) daban así Sortinos no comparables; con datos sintéticos, −0,070
  frente a −0,042. Además, `downside_deviation` usaba el retorno bruto y no el
  exceso sobre rf: una cartera plana salía con desviación a la baja 0.
- **Cambio:**
  - `factors/performance.py` expone `downside_deviation` y `sortino_ratio`.
  - Activos, carteras y backtest usan esas dos funciones.
  - `sfc_tfsig.metrics` no se toca (es compartido; ver #24).
- **Tests:**
  - `test_sortino_del_backtest_usa_la_definicion_del_motor`;
  - `test_desviacion_a_la_baja_es_sobre_el_exceso`.
- **Impacto.** Antes es `f58c2b5` y después este commit, con la misma caché.
  Sharpe, CAGR y drawdown no cambian. Las métricas por activo de [3] tampoco
  (diferencia máxima 0,0):

  | Método (Modo A) | Sortino antes → después | Desv. a la baja antes → después |
  |---|---|---|
  | equal_weight | 1,506 → 1,435 | 15,72% → 15,85% |
  | inverse_vol | 1,785 → 1,690 | 14,23% → 14,36% |
  | risk_parity | 2,095 → 1,969 | 13,76% → 13,89% |
  | min_variance | 2,363 → 2,219 | 13,35% → 13,48% |
  | max_sharpe | 1,129 → 1,284 | 24,38% → 24,49% |

  La definición anterior inflaba el Sortino de las carteras de menor
  volatilidad y desinflaba el de `max_sharpe`, que tiene colas más largas.

### #15 — El plan con `max_sharpe` siempre caía a `equal_weight`

- **Qué estaba mal:** `trade_plan.py:148` reconstruía la cartera con las
  direcciones corregidas usando `mu = 0`. En `max_sharpe` (Charnes-Cooper),
  la restricción `mu'y = 1` es entonces infactible, así que el plan caía
  siempre a `equal_weight` con el aviso "max_sharpe fallo con las direcciones
  corregidas".
- **Cambio:**
  - La cartera principal guarda sus medias contraídas (`primary["mu"]`, de
    James-Stein sobre la ventana de estimación).
  - El plan las usa en la nueva función `trade_plan.plan_weights`.
  - Si el método falla igualmente, la caída a `equal_weight` sigue avisada.
- **Test:** `test_plan_con_max_sharpe_usa_las_medias_contraidas`. Con
  `mu=None` (el comportamiento anterior), la misma llamada devuelve el aviso
  de caída.
- **Impacto.** Antes es `793a822` y después este commit, con
  `--construction max_sharpe`. Con el default `risk_parity` no cambia nada (0
  líneas de diferencia en el resumen de métricas). Con `max_sharpe`:

  | | Antes | Después |
  |---|---|---|
  | Aviso de caída a `equal_weight` | sí | no |
  | Pesos del plan, largos (AAPL, MSFT, NVDA, AMZN, META) | 0,167 / 0,167 / 0,083 / 0,083 / 0,083 | 0,182 / 0,103 / 0,028 / 0,087 / 0,030 |
  | PYPL | −0,125 | −0,052 |
  | Bruta / neta del plan | 1,17 / +0,155 | 0,94 / +0,075 |

  Los nombres con el tope de riesgo activo (TSLA, BA, NKE e INTC tras el FLIP)
  no cambian. La bruta baja porque `max_sharpe` concentra en pocos nombres y
  el tope de riesgo recorta los de stop ancho. El aviso de #16 lo dice.
