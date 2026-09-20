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
```

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
