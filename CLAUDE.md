# Mapa operativo del repositorio

Para quien vaya a modificar código aquí — humano o asistente. El `README.md`
explica *qué* hace el modelo y *por qué*; esto explica *dónde tocar* y *qué se
rompe en silencio*.

## Antes de nada

```bash
python -c "import sys; print(sys.prefix)"   # comprueba el entorno
python -m pytest                            # 600+ tests, sin red, ~1 minuto
```

Los tests no tocan la red. Si un cambio los rompe, se rompió la lógica, no
Yahoo.

## Orden de dependencias

```
config ──> universe ──┐
                      ├──> panel ──> factors ──> portfolio ──> backtest ──> metrics ──> report
data/{sec,prices} ────┘                                    └──> validation ──┘
                                                           └──> orders

backtest, metrics, universe ──> risk/{var, exposure, garch, montecarlo, factor_model,
                                        stress, history, liquidity, limits}
                                  ──> risk/analysis ──> risk_report
```

`risk/` solo mira hacia la izquierda: lee la cartera y los precios, nunca
decide qué se compra. `cli.py` es el único que junta `orders` y `risk`.

`paths.py` y `console.py` no dependen de nada. Todo lo demás importa hacia
abajo, nunca hacia arriba: si un módulo necesita importar a otro que está a su
derecha en ese diagrama, casi siempre significa que la lógica está en el sitio
equivocado.

## Trampas que el código no dice en voz alta

- **`sec.as_of()` es la barrera contra el look-ahead.** Filtra por `filed`
  (fecha de presentación). Si alguien la rodea y filtra por `period_end`, el
  backtest deja de ser válido y nada falla ni avisa. Cualquier cambio ahí exige
  correr `tests/test_sec.py` y mirar los resultados, no solo el "passed".

- **Smart App Control de Windows bloquea binarios concretos, y cambia de
  criterio sin aviso.** El 3 de octubre de 2026 empezo a bloquear
  `pyarrow\_parquet.pyd` (pyarrow 25.0.1; las 22-24 cargan) y `_cvxcore.pyd`,
  que el dia 1 cargaban. Cayo todo: cache, backtest y motor long/short. Por eso
  la cache es pickle (Python puro) y `cvxpy` resuelve con
  `canon_backend=cp.SCIPY_CANON_BACKEND`. No volver a parquet ni quitar ese
  argumento. Para diagnosticar un `DLL load failed ... Control de
  aplicaciones`: `Get-WinEvent -LogName 'Microsoft-Windows-CodeIntegrity/Operational'`
  dice que fichero se bloqueo. Antes de anadir una dependencia compilada,
  importarla y usarla una vez en esta maquina.
- **El portátil se suspende y congela las corridas.** Con la pantalla apagada
  por inactividad entra en suspensión moderna y Windows congela los programas:
  el 3 y el 4 de octubre de 2026 dos corridas se quedaron horas paradas sin
  error. `console.keep_awake()` envuelve `cli.main` y `modelo.py` y pide
  pantalla encendida mientras corre. Para diagnosticar una corrida que "se
  cuelga" sin salida, mirar primero los eventos 506/507 de Kernel-Power en el
  registro System.
- **pickle ejecuta codigo al cargar.** En `data/cache/` solo ficheros que
  escribio este proyecto; nunca copiar ahi la cache de otra persona. Un pickle
  de otra version de pandas que no carga cuenta como cache ausente.
- **`cache.read_frame()` devuelve `None` o un DataFrame.** Nunca escribir
  `cache.read_frame(x) or default`: el valor de verdad de un DataFrame es
  ambiguo y lanza `ValueError`. Ya pasó una vez en `prices.py`.

- **La consola de Windows usa cp1252 al redirigir.** Todo lo que se imprime va
  en ASCII y los puntos de entrada llaman a `enable_utf8_stdout()`. Los ficheros
  `.py` y `.toml` se escriben **sin tildes**; los `.md` sí las llevan. Al añadir
  salida nueva, probarla redirigida (`python main.py backtest > salida.txt`), no
  solo en la terminal.

- **El tope sectorial actúa en la selección, no solo en los pesos.** Si las diez
  mejores del mes son del mismo sector, ningún reparto de pesos respeta un techo
  del 30%. `select_names` limita nombres por sector; `apply_caps` ajusta lo que
  queda y, si los topes son imposibles, **deja caja** en vez de renormalizar. Un
  límite de riesgo que se incumple en silencio es peor que no tenerlo.

- **La rotación se mide sobre el nocional operado**, no comparando la cartera
  objetivo de un mes con la del anterior. Entre rebalanceos los pesos derivan
  con los precios: dos objetivos idénticos pueden exigir operaciones grandes.
  La cifra de rotación tiene que cuadrar con el coste de la misma fila.

- **La caché del panel guarda el fingerprint de la configuración.** Si no
  coincide, se reconstruye. No desactivar esa comprobación para ir más rápido:
  es lo que impide mezclar un panel de una configuración con el backtest de
  otra.

- **El prefiltro de liquidez usa el máximo histórico**, no el valor actual. Es
  un filtro de coste de descarga, no de inversión. Cambiarlo a "liquidez de hoy"
  metería sesgo de supervivencia por la puerta de atrás.

- **`entityType` de EDGAR no sirve para filtrar operativas.** La SEC marca como
  `other` a casi todo emisor extranjero, Royal Bank of Canada incluido. Filtrar
  por `operating` borraba la mitad canadiense del mandato sin error. Los
  vehículos de inversión se excluyen por SIC.

- **`METRIC_DIRECTION` en `financials.py` es la tabla de signos.** Una métrica
  sin entrada ahí lanza `KeyError` a propósito: sin dirección declarada, el
  modelo podría comprar exactamente lo contrario de lo que se pretende.

- **`OBSERVATIONS_VERSION` en `sec.py` hay que subirla a mano.** Las
  observaciones fundamentales se cachean en `facts_obs_<versión>/`. Cualquier
  cambio en `select_by_priority`, `quarterly_segments`,
  `trailing_twelve_months` o `point_in_time_balances` exige subir la versión.
  Si no se sube, el panel sigue leyendo observaciones calculadas con la lógica
  vieja y nada falla. Las filas crudas (`facts_raw/`) no dependen de la versión:
  subirla recalcula desde ellas, sin tocar EDGAR. La clave de caché del panel
  (`cli._panel_cache_name`) lleva también esta versión: antes no la llevaba y
  subirla recalculaba observaciones que el panel en caché nunca leía.

- **La prioridad de etiquetas XBRL se resuelve por periodo, no por empresa.**
  Elegir una sola etiqueta por empresa borraba toda la historia de ingresos
  anterior a ASC 606 (2018): la cobertura de ingresos caía al 52%. No volver a
  "la primera etiqueta con datos gana".
- **En ventas gana el MAYOR valor entre etiquetas del periodo, no la primera**
  (`MAX_VALUE_CONCEPTS`, desde `OBSERVATIONS_VERSION` v3). Una etiqueta ASC 606
  más específica puede ser un desglose: Gold.com publica 25.513 M de
  `Revenues` y 618 M de `RevenueFromContractWithCustomer...`. Con prioridad
  cambiaban el 6,8% de los ejercicios anuales en 719 de 3.222 empresas, y en
  227 las ventas eran menos de la mitad de las reales (OSCR 28,6 M en vez de
  11.700 M; Mercury General 29,6 M en vez de 5.990 M). La bruta de impuestos
  especiales (`...IncludingAssessedTax`) no compite con su neta del mismo
  periodo, pero sí cuenta si la neta no existe (Cooper-Standard: `Revenues` =
  0). De cada etiqueta se toma su primera publicación: el point-in-time no
  cambia. Efecto medido con la misma fecha de fin (30-oct-2026): CAGR 19,87% →
  18,81%, IR 0,45 → 0,38, IC 0,0243 → 0,0241, t del factor valor 1,78 → 1,76.
  La señal no cambia; cambia qué nombres entran al top 30. La diferencia de IR
  está muy por debajo de su error estándar (~0,27). No volver a la prioridad
  porque "rendía más": era un bug que excluía al azar.
- **El recálculo de observaciones va en paralelo**
  (`sec.derive_missing_observations`, 8 procesos). Los hijos reciben el
  directorio de caché del padre (`_init_worker`); sin eso, en Windows (spawn)
  un test con la caché redirigida escribiría en la caché real.

- **SIC 6221 está excluido y debe seguir así.** Ahí viven todos los ETF y
  fideicomisos de materias primas y divisas (GBTC, GLD, SLV, USO, FXC...). Sin
  la exclusión el modelo los compraba por momentum al peso máximo.

- **`require_fundamental_score` exige score de valor Y de calidad.** Sin eso,
  un nombre sin ingresos ni patrimonio entra en cartera puntuado solo por
  precio. Solo quitaba el 2,8% del panel, pero el 12,6% del top-30, y esos
  nombres rendían +0,67% al mes frente al +1,96% de sus reemplazos.

- **El panel se cachea por fingerprint de configuración**
  (`panel_scored_<fingerprint>`). Varias configuraciones conviven. No volver a
  un único slot: pisaba el panel de una configuración con el de otra.

- **La caché de precios decide qué descargar por el registro `prices_fetched`,
  no por las fechas que hay en la caché.** Mirando las fechas, una empresa que
  salió a bolsa en 2015 parece "sin historia" frente a un inicio en 2009, y una
  que dejó de cotizar parece "sin datos recientes": ambas se volvían a pedir
  enteras en cada corrida (~2.400 tickers). El registro anota lo pedido, venga
  o no dato. Borrar `prices_fetched.json` provoca una descarga extra, no un
  error.

- **Con `calendar.end` vacío, el panel en caché caduca al cerrar cada mes.**
  `panel.expected_last_date` da el último fin de mes (o trimestre) cerrado; si
  al panel le falta, `_require_panel` lo reconstruye. Antes no caducaba nunca:
  el 4 de octubre `ordenes` sacó las órdenes con el panel del 31 de agosto, sin
  avisar.

- **Un error de Yahoo no es "no hay datos".** En lotes de 150, Yahoo a veces
  corta y yfinance anota `TypeError: 'NoneType' object is not subscriptable`
  para empresas vivas (VRT, TPG, OZK). Antes, en una actualización de cola, el
  ticker se anotaba en el registro como bajado hasta hoy aunque no llegara
  nada: un hueco permanente que nunca se volvía a pedir (71 tickers en
  septiembre de 2026). Ahora `transient_failures` separa el error del "no hay"
  por el mensaje de yfinance, se reintenta en lotes de 25 y lo que siga
  fallando no se anota ni como pedido ni como fallo de 7 días.

- **La barra de hoy nunca entra en la caché de precios.** Con el mercado
  abierto, Yahoo devuelve un precio de media sesión. El 30 de septiembre de
  2026 una descarga de las 10:40 lo guardó como cierre y el registro dio el día
  por bajado. `settled_end` limita lo anotado a ayer y `_drop_unsettled` tira
  la barra de hoy. Por la derecha se pide la cola en cuanto falta UN día
  hábil: la tolerancia de 5 días dejaba sin bajar el 1 y el 2 de octubre.

- **El backtest no renormaliza cuando falta un precio de ejecución.** Ese peso
  queda en caja, igual que en `apply_caps`. Renormalizar repartió el peso de
  24 nombres sin precio entre 7: 11-17% cada uno con un tope del 6%, y el
  reporte de riesgo analizó ese libro como si fuera la estrategia.

- **`PANEL_VERSION` en `panel.py` también se sube a mano.** La caché del panel
  lleva en el nombre el fingerprint del TOML *y* esta versión. El fingerprint
  solo cambia con el TOML; si cambia el código que lo aplica, sin subir la
  versión la caché sirve el panel viejo. Pasó con `min_history_months`:
  estaba declarado y validado pero ningún módulo lo aplicaba, y tres salidas a
  bolsa de 3-4 meses encabezaban el ranking de agosto de 2026.

- **`apply_liquidity_filters` exige `history_months` sin valor por defecto.**
  Olvidar pasarlo tiene que fallar, no dejar pasar a las OPIs en silencio.

- **La rotación anualizada usa los rebalanceos reales por año**, no 12 fijo.
  Con el 12 cableado, la configuración trimestral salía cuatro veces más cara
  de lo que es.

- **`config/risk.toml` es un fichero aparte a propósito.** Tiene su propio
  fingerprint. Meter la política de riesgo en `strategy.toml` cambiaría el
  fingerprint de la estrategia con cada ajuste de un límite, invalidaría la
  caché del panel y haría incomparables dos backtests idénticos. Solo se muda a
  `strategy.toml` un límite que pase a *restringir* la cartera.

- **`LIMIT_SPECS` en `risk/limits.py` es la tabla de límites.** Cada clave de
  `[limits]` necesita su entrada (medida, dirección, unidad, acción). Una clave
  sin entrada lanza `ConfigError` antes de calcular: un límite que nadie sabe
  con qué comparar, ignorado en silencio, es peor que no tenerlo.

- **Las previsiones del backtest del VaR solo ven el pasado.** El cuantil móvil
  lleva `shift(1)` y la sigma EWMA del día `t` se calcula con retornos hasta
  `t-1`. Hay un test que mete un salto el último día y exige que el VaR previsto
  para ese día no cambie. Si se rompe, el backtest del VaR sale perfecto y no
  vale nada.

- **Los sectores de los escenarios se validan contra `universe.SECTORS`.** Un
  sector mal escrito no casaría con ninguna posición y el choque sectorial
  desaparecería sin aviso. Si cambia el esquema sectorial, cambian también los
  escenarios de `config/risk.toml`.

- **El descuento del bootstrap (`return_haircut`) no se pone a cero para que
  salga mejor.** Es el sesgo de supervivencia declarado; sin él, las
  probabilidades de perder y de quedar detrás del benchmark salen optimistas.

- **En el estrés histórico, un nombre sin precios se mueve con su beta, y el
  choque se corta en −100%.** Con betas de 3-4 y caídas de mercado del 50%, el
  corte actúa: el nombre pierde su peso entero. Es conservador y el reporte
  enseña el peso aproximado; no quitar el corte, que daría pérdidas imposibles.

- **El modelo factorial explica el día `d` con el panel ANTERIOR a `d`.**
  `searchsorted(..., side="left") - 1` en `estimate_factor_model`. Con
  `side="right"`, el panel de fin de mes explicaría el retorno de ese mismo día
  con datos de su cierre. Hay un test.

- **Modelo factorial y Ledoit-Wolf pueden discrepar, y no es un fallo.** LW
  promedia cómo se movieron los nombres el último año; el factorial aplica las
  exposiciones de hoy. En sep-2026 la exposición a momentum pasó de −0,22 a
  2,29 en un año y el factorial daba 34% de volatilidad frente a 24% de LW
  (correlación de su predicción con la cartera real: 0,90). El análisis avisa
  por encima de un 25% de diferencia. No "arreglar" una para que cuadre con la
  otra.

- **El GARCH usa objetivo de varianza y rejilla, no scipy.** La varianza de
  largo plazo es la muestral; solo se estiman alfa y beta, con alfa + beta <
  0,999. scipy no es dependencia del proyecto: no introducirla para esto.

- **La calibración del riesgo solo mira hacia atrás en cada rebalanceo.** La
  covarianza de la fecha `e` usa precios hasta `e`. Hay un test que altera los
  precios finales y exige que las previsiones anteriores no cambien. Si se
  rompe, el estadístico de sesgo sale perfecto y no vale nada.

## Añadir un escenario de estrés o un límite

1. Escenario: un bloque `[[stress.historical]]` (nombre, inicio y fin) o
   `[[stress.hypothetical]]` (mercado, sectores y factores) en
   `config/risk.toml`. Los choques sectoriales y de factor son juicio
   declarado: se escriben con su justificación al lado.
2. Límite: la clave en `[limits]` y su `LimitSpec` en `risk/limits.py`. Si la
   medida no existe, se calcula en `risk/analysis._measures`.
3. `python -m pytest tests/test_risk_*.py` y `python main.py riesgo --sin-red`
   redirigido a fichero, para comprobar que la salida sigue en ASCII.
4. Un mutante en `tools/risk_mutants.py` que rompa la lógica nueva, y
   `python tools/risk_mutation.py <palabra>` hasta que salga cazado.

## Calidad de los tests de riesgo

```bash
python tools/risk_coverage.py --min 98        # cobertura de líneas, sin instalar nada
python tools/risk_mutation.py --jobs 4        # ¿cazan los tests bugs plausibles?
python tools/risk_mutation.py --list          # el catálogo
```

- **La cobertura dice qué se ejecuta; la mutación, si algún test se daría
  cuenta de que está mal.** Con 99% de líneas cubiertas, la primera pasada de
  mutación dejó vivos 15 de 50 bugs plausibles: la t de Student sin reescalar
  la covarianza (VaR un 29% inflado), Cornish-Fisher con la asimetría al
  revés, el GARCH sin condición de estacionariedad, la beta con la varianza de
  la cartera, la liquidación de crisis con el volumen normal. Hoy están los 64
  del catálogo cazados; mantenerlo así.
- **La mutación nunca toca el árbol de trabajo.** Copia `src/`, `tests/` y
  `config/` a un directorio temporal y muta ahí. Se puede correr con el editor
  abierto o con otra sesión trabajando en el repo.
- **Cada mutante describe un fragmento exacto del código.** Si una línea cambia
  y su mutante deja de aplicarse, `tests/test_risk_tools.py` falla y dice cuál:
  actualizar el fragmento en el catálogo, no borrar el mutante.
- **Un mutante que sobrevive es un test que falta**, no un mutante que sobra.
  Solo se quita si es equivalente (no cambia ningún resultado), y se dice por
  qué en el commit.
- **`tests/risk_helpers.py` tiene los datos sintéticos compartidos** (mercado,
  cartera, estrategia completa y política ligera). Las fixtures de cada fichero
  los llaman; no copiar el mercado sintético en un fichero nuevo.
- **`tests/test_risk_cli.py` sustituye disco y red con `monkeypatch`.** Si
  `cmd_riesgo` u `ordenes` empiezan a leer algo nuevo (un fichero, una
  descarga), hay que sustituirlo ahí también o el test tocará la máquina real.

## Añadir una métrica nueva

1. Calcularla en `financials.py` (`value_metrics` o `quality_metrics`), pasando
   siempre por `_safe_divide`.
2. Declarar su dirección en `METRIC_DIRECTION`.
3. Añadirla a la lista del factor en `config/strategy.toml`.
4. Correr `python main.py panel` y mirar su **cobertura**. Por debajo del 60%
   se descarta sola, y eso es una decisión, no un error.

## Añadir un factor nuevo

1. Entrada en `factors.weights` del TOML y en `_KNOWN_FACTORS` (`config.py`).
2. Si es de precio, sus métricas van en `FACTOR_METRICS` (`composite.py`); si es
   fundamental, en su propia sección `[factors.<nombre>]` del TOML.
3. Validarlo **por separado** antes de meterlo en el compuesto: IC propio,
   quintiles propios, Fama-MacBeth. Un factor que no se sostiene solo no mejora
   la combinación, la diluye.

## Convenciones

- Nada de rutas calculadas a mano: todo sale de `paths.py`.
- Las funciones marcadas `[puro]` en el README no hacen red ni disco. Mantenerlo
  así: es lo que permite probarlas con datos sintéticos.
- Los mensajes de error dicen qué hacer, no solo qué pasó.
- Los comentarios explican **por qué**, no qué. El qué ya está en el código.


## Motor long/short (`src/quant_engine/`)

Paquete aparte del modelo multifactor. Importa de `sfc_tfsig` (caché de disco,
métricas, VaR, Ledoit-Wolf, perfiles SEC) pero `sfc_tfsig` no importa de él,
salvo `cli.py`, que lo lanza. Orden interno:

```
settings ──> data/{loader, cleaning, validation} ──> factors/* ──> inference/* ──> signals/*
                                                                         └──> portfolio/* ──> backtest/* ──> risk/*
analysis.py orquesta todo ──> reporting/{terminal, charts, export}      robustness.py reusa backtest
```

Trampas:

- **Nada de `import statsmodels.api`.** Lo bloquea el Control de aplicaciones de
  Windows en esta máquina. Usar `from .. import _sm as sm`.
- **`rich` interpreta `[texto]` como estilo.** Todo texto calculado que se
  imprima (avisos, notas, celdas) pasa por `rich.markup.escape`, o se come
  prefijos como `[max_sharpe]`.
- **El estrés histórico es comprar-y-mantener**, no componer retornos diarios
  negados: eso es un ETF inverso, no un corto.
- **La deriva neutra del Monte Carlo es rf sobre el capital**, no rf por
  posición con signo (daba bruta × rf).
- **Los topes por nombre deben dejar libertad.** Si `n_por_pata × max_position`
  iguala la pata, todos los métodos coinciden; `analysis.py` avisa.
- **`ffill(limit=0)` revienta en pandas**; `cleaning._ffill` lo trata aparte.
- **El plan de operación reconstruye la cartera con los signos corregidos.**
  No reutiliza los pesos de `primary`: tras un FLIP o un NO TRADE esos pesos
  apuntan al lado equivocado. Si solo queda una pata, no hay cartera y el plan
  da niveles sin tamaño.
- **El plan cubre la beta que dejan sus propias correcciones.** `risk_parity`
  no es beta-neutral por construcción; que la primaria saliera con beta ~0 fue
  casualidad. Tras un NO TRADE en un corto de beta alta y los recortes por
  riesgo, el libro quedó en +0,22. `beta_hedge` lo lleva a cero con el
  benchmark; no quitarlo para que el plan "se vea más limpio".
- **La cobertura se calcula sobre las posiciones FINALES**, después de los
  HOLD y de los NO TRADE que mantienen una posición contra research, no sobre
  el plan teórico. Si no, se cubre un libro que no es el que queda.
- **Una posición en el benchmark es la cobertura.** Si la beta no necesita
  cobertura, el plan la cierra. Con `hedge_beta: false` pasa a NOT ANALYZED.
- **Una canasta solo LONG se puntúa contra pares de referencia.** Los z-scores
  del Quant Score son de sección cruzada: con dos nombres salen vacíos y con
  todos LONG la mitad queda en contra por construcción. `analysis.reference_peers`
  añade `scoring.reference_peers` del YAML cuando la canasta es solo LONG o
  tiene menos de `scoring.min_cross_section` nombres. Los pares entran en
  features, WRDS y scores, y se recortan antes de cualquier tabla
  (`_restrict`, `scores.loc[tickers]`). Si un par aparece en el reporte o en el
  plan, se rompió ese recorte.
- **Solo LONG = 100% largo, sin cobertura de beta y sin cortos.** `settings.exposure`
  da (1, 1) y no la bruta del YAML (200% en una pata sería apalancamiento 2x).
  `trade_plan.long_only_signs` convierte un FLIP en NO TRADE, y `beta_hedge` no
  se aplica: la beta de mercado es la exposición buscada.
- **`simulate_trades` es la regla del plan, no una aproximación.** Si cambian
  `stop_atr`, `take_profit_r`, la salida parcial o el break-even en el plan,
  tienen que cambiar igual ahí, o las expectativas medirán otra estrategia.
  `plan_backtest` alinea las MISMAS fechas para todos los nombres a propósito:
  sumar distribuciones independientes borraba la correlación y subestimaba la
  cola. En `rich`, los encabezados van con paréntesis (`E(R)`): con corchetes,
  "[ret]" se lee como estilo y desaparece.
- **En `trade_plan.hit_rate`, un día que toca stop y TP cuenta como stop.** Sin
  datos intradía no se sabe el orden; suponer el TP inflaría la tasa base.
- **`clean.close/high/low` van en USD; los niveles, en moneda local.**
  `cleaning.align` convierte con el tipo de cambio del día (`fx`), así que
  retornos, beta, liquidez y backtests son los de un inversor en dólares. Para
  entrada, stop, objetivos y gráficos de niveles se usa `clean.local_prices(t)`.
  En `trade_plan`, todo lo que es dinero (acciones, nocional, riesgo, P&L,
  banda de HOLD) multiplica por `plan["fx"]`. Dividir USD entre un precio en CAD
  compraba un 30% de menos (en Londres, que cotiza en peniques, ~76 veces
  menos). Un sufijo de bolsa nuevo va en
  `loader.QUOTE_CURRENCY`; sin él, el nombre no se convierte y se avisa.
- **Compustat no conoce los tickers de Yahoo de Toronto.** `ATZ.TO` es `ATZ.`
  y `TECK-B.TO` es `TECK.B` (`wrds_data.compustat_ticker`). El listado de
  Toronto suele no tener `ibtic`: `match_securities` lo toma de una emisión
  hermana de la misma empresa. Las EPS salen de `statsum_epsus` UNION
  `statsum_epsint`. El interés corto de Compustat es solo de EE.UU.: un nombre
  canadiense queda sin él y el componente se renormaliza.
- **Los ratios de valor usan el precio en la moneda en que reporta la
  empresa** (`curcdq`, `wrds_data.price_in_currency`). Shopify cotiza en CAD en
  Toronto y reporta en USD; Aritzia, en CAD las dos cosas. Mezclar monedas
  movía el rendimiento por beneficio un 30-40%, sin error.
- **La validación del Quant Score usa las MISMAS funciones que el análisis**
  (`analysis.features_at` llama a `_per_asset`, `_features` y `ttm_panel` con
  `slice_clean(clean, d)`). Si cambia una característica del score, cambia
  también en la validación sin tocar nada. No escribir una versión "rápida"
  aparte: validaría otro score. Hay un test que triplica los precios
  posteriores a `d` y exige que las características de `d` no cambien, y otro
  que mete un trimestre publicado después de `d` y exige que no cuente.
  Analistas (`analyst_panel`) e interés en corto (`short_interest_at`) son las
  mismas funciones de hoy con `as_of = d`. El interés en corto histórico lleva
  `SHORT_INTEREST_LAG_DAYS` (14): la fecha de liquidación no es la de
  publicación. Hoy va sin retraso, porque lo que está en WRDS ya se publicó.
- **No reajustar `signal_weights` con el IC de la validación.** Son 36 meses
  de un universo: elegir pesos con la muestra que los evalúa es sobreajuste, y
  la siguiente validación saldría buena por construcción.
- **Un FLIP pasa por `trade_plan.flip_allowed`.** Sin evidencia de que el score
  predice (t del IC < `flip_min_ic_t`), `unproven_flip` decide: `reduce` (por
  defecto) lo opera a `reduce_size` con convicción LOW; `no_trade` lo deja en
  NO TRADE, y `score_verdict` guarda lo que dijo el score. Validación apagada
  (`{}`) respeta la decisión; validación sin datos (`available: False`) cuenta
  como sin evidencia. No bajar el umbral para que salgan FLIPs a tamaño
  completo: en octubre de 2026 el IC fue −0,004.
- **Las fichas (`playbook.py`) solo presentan.** Dirección, niveles y tamaño
  salen de `build_plan`; la ficha no recalcula nada salvo el tamaño propio
  (`trade_plan.solo_size`, cuando no hay libro) y la cobertura suelta. Si
  cambia una regla del plan (stop, objetivos, break-even), los pasos de
  ejecución la repiten tal cual: cambiar las dos cosas a la vez.
- **`check_plan` comprueba identidades, no recalcula el plan.** Si un cambio
  hace fallar una identidad (acciones × precio × FX ≠ nocional, riesgo por
  encima del tope), el error está en el cambio: no relajar la tolerancia.
- **Avisos de datos en la ficha desde `clean.stats`, no desde el texto de la
  calidad.** `playbook.execution_flags` se queda con lo que cambia la orden de
  hoy (precio viejo, último salto, splits, movimientos de más del 50%). Los
  días rellenados por festivos de otra bolsa no están a propósito.
- **Las fechas de resultados son de Yahoo y pueden faltar.**
  `loader.load_earnings_dates` espera como mucho 20 s y cachea un día. Sin
  fecha, la ficha dice "check it": nunca se inventa.
- **`month_end_dates` toma fines de mes del calendario COMPLETO.** Filtrar
  primero por `last` convertía un 17 de septiembre en "cierre de septiembre".
- **`composite.winsorize` no recorta si los percentiles coinciden.** Con 51
  valores iguales y uno distinto, recortar al 5–95 borraba la única
  diferencia.
- **El loader del motor reintenta y cae a la caché vieja.** `download_with_retry`
  pide lo que falte en lotes de 25. Si sigue faltando y la caché cubre el
  inicio, se usa y `MarketData.stale` lo registra para el aviso. Sin esto, un
  corte de Yahoo borraba 16 de 50 pares y la validación corría con 38 nombres.
- **El stop por tiempo es regla en los cortos y solo en ellos**
  (`trade_plan.rule_time_stop`, `trade_plan.short_time_stop` en el YAML). Se
  decidió fuera de muestra en los 50 pares (día de revisión de la primera
  mitad de cada historia, medido en la segunda): cortos 80% de nombres mejor,
  +0,05R, t 3,7; largos 40%, −0,03R, t −1,8. En un corto el backtest de la
  regla, la ventana de tenencia y `plan_backtest` se calculan CON el stop; la
  E(R) sin él queda en `bt_exp_r_no_time_stop` para comparar. A un largo no se
  le aconseja cortar aunque dentro de muestra parezca ayudar. No extenderlo a
  los largos sin repetir esa prueba con más canastas.
- **`simulate_trades(time_stop=None)` es la regla sin stop por tiempo.** Con
  `time_stop` cierra sin TP1 en ese día. El día sale de las mismas
  operaciones (percentil 75 del día del TP1): la mejora en un nombre es
  dentro de muestra y la ficha lo dice.
- **La prueba del stop por tiempo es `time_stop_out_of_sample`**: día de
  revisión de la primera mitad de las entradas y E(R) medida en la segunda. La
  mejora dentro de muestra (`bt_exp_r_time_stop`) no sirve para decidir:
  siempre sale a favor.
- **El multifactor solo cuenta si `multifactor.usable`**: panel de 45 días o
  menos y t del IC ≥ 2 en el último reporte OFICIAL (`backtest_AAAAMMDD_HHMM.md`).
  Un contrafactual o un experimento va a otra carpeta
  (`output/reports/counterfactual/`): con el mismo patrón de nombre se colaría
  como evidencia. Solo mueve la convicción (±1 escalón) y los FLIP sin
  evidencia; los veredictos no.
- **El score fuera del universo (`multifactor.outside_universe`) usa las
  funciones del modelo, no una copia**: `financials.compute_all` y
  `composite.build_scores` sobre la última sección cruzada con las filas
  añadidas. Una fila más mueve un poco media y desviación: un nombre idéntico a
  uno de dentro cae casi en el mismo sitio (hay un test), no exactamente. Ese
  score nunca entra en `mf_rows` ni mueve la convicción.
- **`universe.max_names` es una regla de exclusión más.** Si en una fecha hay
  más elegibles que el tope, se quedan los más líquidos. `exclusion_reasons`
  la mide con el volumen del último que entró; sin ella, Gold.com salía como
  "cambio de ticker", que era falso. Desde noviembre de 2026 el tope es 5.000
  y no actúa (máximo histórico de elegibles: 2.207). Con 1.200 recortaba hasta
  el 45% del universo, y quitarlo subió la t del IC de 3,00 a 3,47 sin cambiar
  el coste. No volver a bajarlo "por coste de descarga": se aplica después de
  la descarga y no ahorra nada.
- **La ficha limitada (`playbook.limited_card`) nunca da tamaño ni pasos de
  ejecución.** Un nombre sin score ni backtest no tiene base para dimensionar.
  Si se le añade tamaño, deja de ser "limitada" y pasa a ser una señal sin
  validar.
- **`tp1_day` en la salida de `simulate_trades` solo se anota.** Si cambia la
  lógica del TP1, el día tiene que seguir siendo el de la sesión en que se
  tocó, o `holding_window` medirá otra cosa.
- **La t del backtest de la regla es de Newey-West** (`trade_plan.hac_t_stat`,
  retardo = la operación más larga). Las entradas diarias se solapan: con el
  error estándar ingenuo, 756 entradas parecerían 756 operaciones
  independientes y cualquier E(R) saldría significativo.

## DCF automático (`src/quant_engine/valuation/`)

```
data (SEC XBRL | Compustat) ──> assumptions ──> dcf (puro, vectorizado) ──> engine ──> excel / cli / fichas
```

- **`dcf.value_vector` es la única fórmula.** Escenarios, rejillas, Monte Carlo
  y DCF inverso la llaman con vectores. Si se cambia una línea del FCFF, se
  cambia ahí y en `excel.py` (las fórmulas de la hoja DCF), y se vuelve a
  comprobar que el Excel recalculado da lo mismo. Hoy coincide hasta 1e-13:
  `test_el_excel_recalculado_da_lo_mismo_que_el_motor` abre Excel por COM,
  recalcula los tres escenarios y compara EV y valor por acción. Sin Excel
  se salta: correrlo en esta máquina tras tocar `excel.py` o `dcf.py`.
- **Ventas = máximo entre etiquetas por ejercicio** (`data.annual_from_sec`),
  no la primera por prioridad: un desglose ASC 606 ganaba al total (Gold.com).
  El modelo multifactor usa la misma regla desde `OBSERVATIONS_VERSION` v3.
- **La deuda no incluye arrendamientos operativos de US GAAP**: su coste ya está
  en el EBIT. Sí los financieros, y todos los de IFRS 16.
- **Las acciones de portada (dei) solo se usan si cuadran ±15% con las
  diluidas**: en empresas con varias clases la SEC publica una sola (ARES: 103
  M de unas 300).
- **Compustat consolida la deuda de los fondos en las gestoras** (ARES): el
  patrimonio sale negativo. No es un fallo del DCF: un FCFF no vale para
  gestoras, bancos ni aseguradoras, y el aviso lo dice.
- **`CRITICAL` en `engine.py` decide qué avisos invalidan el número.** Un aviso
  nuevo que deba invalidar va ahí, con un prefijo de texto estable.
- **El suelo de reinversión usa el ventas/capital propio si es mayor que 5**
  (tope 25). Con 5 fijo, un comerciante de metales (18x) necesitaba 2.000 M
  de capital al año y valía 0.
- **El valor por acción tiene suelo en 0** (responsabilidad limitada); el
  patrimonio negativo se conserva para el aviso.
- **Las consultas nuevas de WRDS (funda, consenso de ventas) van al final de
  `wrds_data.load` y cada una en su try.** Un error SQL deja la transacción
  abortada; si fueran antes, tumbarían todo lo demás.
- **rf de monedas distintas del USD vive en el YAML** (`valuation.risk_free`) y
  caduca: revisarlo cada trimestre.

## WRDS (`src/sfc_tfsig/data/wrds.py`)

- **Instalar `wrds` con `--no-deps`.** Fija `pandas<2.3`; en Python 3.14 pip
  intenta compilar pandas 2.2 y falla pidiendo Visual Studio. Ver
  `pyproject.toml`, grupo `wrds`.
- **Usar `with connection() as conn:` para varias consultas.** WRDS limita las
  conexiones simultaneas; `query`/`fetch_table` sin `conn` abren y cierran una
  cada vez.
- **La cache de WRDS lleva el hash del SQL en el nombre.** Cambiar el filtro con
  el mismo `cache_name` es otra entrada, no datos viejos. `refresh=True` para
  traer meses recientes.
- **Sin pgpass y sin terminal, `get_connection` falla a proposito** en vez de
  quedarse esperando la contrasena. Configurarlo una vez con
  `python main.py wrds --setup`. No usar `wrds.Connection()` a secas para eso:
  sin usuario explicito ofrece el de Windows por defecto, y con Enter la
  autenticacion falla y el pgpass nunca se crea.
- **El motor long/short usa WRDS si esta disponible** (`quant_engine/data/wrds_data.py`):
  Fama-French 5 + momentum, GICS, fundamentales de Compustat point-in-time por
  `rdq`, IBES y interes corto. Si WRDS falla, `load` devuelve `available=False`
  y el motor sigue con Yahoo + ETFs + SIC; nunca lanza. Los componentes
  `value`, `quality`, `analyst` y `short_interest` del Quant Score quedan
  vacios y el resto se renormaliza.
- **Las consultas de WRDS del motor llevan la fecha del analisis en el SQL**:
  asi la cache (hash del SQL) se renueva cada dia sin `refresh`.
- **`_probe` prueba la conexion antes de `wrds.Connection`.** Si la
  autenticacion falla, `wrds` no lanza: pide usuario por teclado (EOFError sin
  terminal) y, con una aprobacion Duo pendiente, se cuelga sin limite. El
  sondeo usa psycopg2 con `connect_timeout` y traduce el fallo a que hacer.
  Los tests que llaman a `get_connection` tienen que parchear `_probe`, o
  tocan la red y tardan 45 s.
- **Un plan sin WRDS se avisa en rojo** en la seccion 14, en la senal final y
  en el HTML. Cambia los veredictos (INTC: NO TRADE con WRDS, FLIP sin el).
- **`pd.NA` de WRDS revienta `float()`**: todo pasa por `wrds_data._numeric`.
- **La revision de EPS compara el MISMO ejercicio fiscal** (`fpedats`) y va
  acotada a +-100%. Con base cerca de cero explotaba (Boeing: -133%).
