# Mapa operativo del repositorio

Para quien vaya a modificar código aquí — humano o asistente. El `README.md`
explica *qué* hace el modelo y *por qué*; esto explica *dónde tocar* y *qué se
rompe en silencio*.

## Antes de nada

```bash
python -c "import sys; print(sys.prefix)"   # comprueba el entorno
python -m pytest                            # 50+ tests, sin red, ~2 segundos
```

Los tests no tocan la red. Si un cambio los rompe, se rompió la lógica, no
Yahoo.

## Orden de dependencias

```
config ──> universe ──┐
                      ├──> panel ──> factors ──> portfolio ──> backtest ──> metrics ──> report
data/{sec,prices} ────┘                                    └──> validation ──┘
                                                           └──> orders

backtest, metrics, universe ──> risk/{var, exposure, montecarlo, stress, limits}
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
  subirla recalcula desde ellas, sin tocar EDGAR.

- **La prioridad de etiquetas XBRL se resuelve por periodo, no por empresa.**
  Elegir una sola etiqueta por empresa borraba toda la historia de ingresos
  anterior a ASC 606 (2018): la cobertura de ingresos caía al 52%. No volver a
  "la primera etiqueta con datos gana".

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

## Añadir un escenario de estrés o un límite

1. Escenario: un bloque `[[stress.historical]]` (nombre, inicio y fin) o
   `[[stress.hypothetical]]` (mercado, sectores y factores) en
   `config/risk.toml`. Los choques sectoriales y de factor son juicio
   declarado: se escriben con su justificación al lado.
2. Límite: la clave en `[limits]` y su `LimitSpec` en `risk/limits.py`. Si la
   medida no existe, se calcula en `risk/analysis._measures`.
3. `python -m pytest tests/test_risk_*.py` y `python main.py riesgo --sin-red`
   redirigido a fichero, para comprobar que la salida sigue en ASCII.

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
