# SFC × TFSIG — Portfolio Strategy

Modelo **multifactor long-only** sobre renta variable de Estados Unidos y Canadá,
para el portafolio conjunto del Asset Management desk de Sabana Finance Club y
TFSIG.

No es un screener. Un screener ordena por múltiplos y se queda en una lista;
esto construye un panel **point-in-time**, puntúa cada empresa en cuatro
dimensiones, arma una cartera con restricciones de riesgo reales, la contrasta
fuera de muestra, y produce las órdenes concretas del rebalanceo del mes.

> **La pregunta del modelo:** entre las empresas invertibles de hoy, ¿cuáles
> combinan precio razonable, calidad de negocio y comportamiento de mercado
> favorable — medido todo con la información que existía en cada fecha, no con
> la de hoy?

---

## Los cuatro factores

| Factor | Qué mide | Peso | Origen |
|---|---|---|---|
| **Valor** | Rendimiento por beneficio, flujo de caja libre, libros, ventas y EBIT/EV | 30% | Fama-French (1992); rendimientos en vez de múltiplos |
| **Calidad** | ROIC, ROE, rentabilidad bruta, margen de caja, cobertura, apalancamiento, devengo, dilución | 30% | Quality Minus Junk — Asness, Frazzini, Pedersen (2019); devengo de Sloan (1996) |
| **Momentum** | Retorno de 12 meses saltando el último | 30% | Jegadeesh y Titman (1993) |
| **Baja volatilidad** | Volatilidad realizada de 252 sesiones, invertida | 10% | Betting Against Beta — Frazzini y Pedersen (2014) |

Se combinan **ponderados, no en cascada**. Un embudo descarta de forma
irreversible a una empresa excelente en tres factores que falla el corte del
cuarto por poco; la suma ponderada deja que las fortalezas compensen debilidades.
El precio de esa decisión es que una debilidad se puede esconder dentro del
compuesto, y por eso el reporte muestra siempre los cuatro scores por separado
en cada posición.

Cada métrica se estandariza **dentro de su sector**. Un banco con apalancamiento
5x no es una empresa mal gestionada: es un banco. Un z-score global no mediría
calidad, mediría sector.

---

## Cómo se usa

```bash
# 1. Una sola vez
cp .env.example .env          # y pon tu nombre y correo reales (la SEC los exige)
pip install -e ".[fast,charts,dev]"

# 2. Ver qué empresas son elegibles y por qué
python main.py universo

# 3. Descargar datos y ensamblar el panel (lento la primera vez, luego cacheado)
python main.py panel

# 4. Correr la estrategia: reporte con rendimiento, costes y validación
python main.py backtest

# 5. Generar las órdenes del rebalanceo de este mes (con control de riesgo previo)
python main.py ordenes --capital 200000

# 6. Reporte de riesgo: VaR, Monte Carlo, estrés, liquidez y límites
python main.py riesgo
```

Todo lo que se descarga se cachea en `data/cache/`. La primera corrida tarda;
las siguientes son minutos. `python main.py limpiar` vacía la caché.

---

## Toda decisión de inversión vive en `config/strategy.toml`

Número de posiciones, pesos de factores, topes por nombre y sector, costes,
universo, fechas, ventanas de validación. **Si cambiar un número cambia lo que se
compra, ese número está en el TOML y no en el código.**

El motor calcula un `sha256` del contenido y lo estampa en cada backtest y cada
reporte. Dos resultados con el mismo fingerprint son comparables; con fingerprint
distinto no lo son. Eso zanja la discusión de "pero a mí me daba otra cosa".

La configuración se valida antes de tocar datos, y los mensajes de error dicen
qué hacer:

```
[portfolio]: max_weight (0.03) x n_positions (30) = 0.90 < 1.
Los topes impiden invertir el 100% del capital
```

---

## Estructura

```
.
├── config/strategy.toml          TODA decisión de inversión, en un sitio
├── config/risk.toml              política de riesgo: confianza, escenarios, límites
├── main.py                       lanzador sin instalar; delega en cli.py
├── src/sfc_tfsig/
│   ├── paths.py                  raíz del proyecto; único sitio donde se calcula
│   ├── console.py                UTF-8 en stdout (trampa de Windows)
│   ├── config.py                 carga, valida y firma el TOML
│   ├── data/
│   │   ├── cache.py              caché en disco, escritura atómica
│   │   ├── sec.py                XBRL de la SEC → fundamentales point-in-time
│   │   └── prices.py             precios ajustados, momentum, volatilidad
│   ├── universe.py               universo investible; SIC → sector; país
│   ├── financials.py             ratios derivadas                      [puro]
│   ├── panel.py                  ENSAMBLA la rejilla fecha × empresa
│   ├── factors/
│   │   ├── scoring.py            winsorizar, z-score, neutralidad sectorial [puro]
│   │   └── composite.py          los cuatro factores y su combinación   [puro]
│   ├── portfolio.py              selección, pesos y restricciones       [puro]
│   ├── backtest.py               motor mensual con costes y contabilidad en acciones
│   ├── metrics.py                CAGR, Sharpe, drawdown, IR             [puro]
│   ├── validation.py             IC, quintiles, Fama-MacBeth, walk-forward [puro]
│   ├── report.py                 reporte Markdown para el comité
│   ├── orders.py                 hoja de órdenes del rebalanceo
│   ├── risk/
│   │   ├── var.py                VaR, ES y su backtest                  [puro]
│   │   ├── exposure.py           covarianza, contribuciones, liquidez   [puro]
│   │   ├── montecarlo.py         bootstrap y simulación paramétrica     [puro]
│   │   ├── stress.py             escenarios históricos, hipotéticos, inverso [puro]
│   │   ├── limits.py             límites de vigilancia y semáforo       [puro]
│   │   └── analysis.py           junta todo sobre una cartera           [puro]
│   ├── risk_report.py            reporte de riesgo y control previo a operar
│   └── cli.py                    los seis comandos
└── tests/                        242 tests del modelo (346 con el motor long/short), sin red
```

`[puro]` significa funciones sin red, sin disco y sin estado: entra un DataFrame
y sale otro. Se prueban con datos inventados y resultados calculables a mano.

---

## Cómo se evita el look-ahead

Es la única parte del modelo que no admite discusión, porque un fallo aquí hace
que todo lo demás sea decorativo.

1. **Los fundamentales entran por fecha de presentación, no de cierre contable.**
   Un balance a 31-dic-2023 se presentó en febrero o marzo de 2024. El panel de
   enero de 2024 no lo ve. La barrera es `sec.as_of()`, y filtra por `filed`.

2. **Se usa la primera publicación, no la reexpresión.** Cuando un periodo
   aparece varias veces, gana el `filed` más temprano: es el número que el
   mercado tuvo delante ese día.

3. **La ejecución va con retardo.** La señal se calcula al cierre de `d` y se
   opera al cierre de `d+1`. Hay un test dedicado: si el precio salta el día
   siguiente a la señal, el modelo compra *después* del salto.

4. **Los filtros de liquidez usan los datos de esa fecha**, no los de hoy, y con
   el **precio que cotizaba ese día**. El cierre ajustado por splits y
   dividendos depende de lo que la empresa hizo después: hasta la auditoría de
   octubre de 2026 el filtro de precio y la capitalización lo usaban, y NVDA
   (20,05 USD a cierre de 2014, 0,48 ajustado) quedaba fuera del universo hasta
   2017. Ahora el precio y la capitalización usan el cierre real; retornos,
   momentum y volatilidad siguen con el ajustado.

5. **El retorno futuro vive en una columna con nombre explícito** y solo lo
   consume `validation.py`. El backtest no lo mira: ejecuta operaciones y
   contabiliza costes.

---

## Cómo se valida

Un backtest con una curva bonita no es evidencia: es una hipótesis que sobrevivió
a un solo experimento. `python main.py backtest` corre además:

- **Coeficiente de información** (IC de rangos mensual) con t de Newey-West.
  Referencia: 0,02–0,05 es un buen factor. Por encima de 0,10, lo primero que hay
  que buscar es una fuga de datos futuros.
- **Carteras por quintiles** y monotonicidad. Un spread que depende solo del peor
  quintil dice que el modelo detecta basura, no que encuentra ganadores — y una
  estrategia long-only no puede vender la basura en corto.
- **Fama-MacBeth por factor** con errores Newey-West. Sin esa corrección, la
  autocorrelación de un factor lento infla el estadístico t hasta el doble.
- **Walk-forward**: IC dentro y fuera de muestra, ventana a ventana. Si el de
  entrenamiento es alto y el de prueba ronda cero, el modelo memoriza.

## Gestión de riesgo

`python main.py riesgo` mide la cartera de hoy y la historia de la estrategia, y
las compara con una política de riesgo escrita. Nada de esto cambia lo que se
compra: el modelo decide la cartera, el riesgo dice si cabe en el mandato.

**La política vive en `config/risk.toml`, no en `strategy.toml`.** Niveles de
confianza, escenarios y límites no deciden qué se compra; si vivieran junto a
la estrategia, mover un límite de VaR cambiaría su fingerprint e invalidaría la
caché del panel. El reporte de riesgo estampa los dos fingerprints. El día que
un límite pase a *restringir* la cartera, se muda a `strategy.toml`.

Qué calcula, y por qué cada pieza:

- **Riesgo ex-ante de la cartera actual.** Covarianza del último año con
  contracción de Ledoit-Wolf hacia correlación constante. Volatilidad, beta,
  tracking error y descomposición de Euler: cuánto del riesgo aporta cada
  nombre y cada sector. El tope de peso controla el tamaño, no la volatilidad
  ni la correlación; esta tabla enseña lo que se escapa.
- **VaR y Expected Shortfall** por cuatro métodos (histórico, normal,
  Cornish-Fisher y simulación histórica filtrada con volatilidad EWMA), a 1 y 21
  sesiones, al 95% y al 99%, en porcentaje y en dólares. Para la cartera actual,
  además, Monte Carlo normal y t de Student con la misma covarianza: la
  diferencia es la prima de cola gorda.
- **Backtest del VaR.** Cada día se reestima con información del cierre
  anterior y se cuentan las excepciones: Kupiec (frecuencia), Christoffersen
  (racimos) y semáforo de Basilea. Un VaR sin backtest es una opinión.
- **Monte Carlo por bootstrap estacionario** de la historia real, estrategia y
  benchmark con los mismos índices, a 1, 3 y 5 años: percentiles, probabilidad
  de perder, de quedar detrás de SPY y de sufrir drawdowns del 10, 20 y 30%. Se
  corre con un descuento anual del 2% por el sesgo de supervivencia declarado.
- **Pruebas de estrés.** Históricas: la cartera de hoy con los precios reales de
  doce episodios, de Lehman a los aranceles de 2025 (los nombres que no cotizaban
  se aproximan por su beta, y el reporte dice cuánto peso se aproximó).
  Hipotéticas: beta × mercado + extra sectorial + choque de factor × z-score,
  incluido un crash de momentum. Inversa: qué caída del mercado produce una
  pérdida dada.
- **Liquidez y capacidad**: días para liquidar cada posición al 20% del volumen
  diario, y el NAV a partir del cual los costes fijos del backtest dejan de ser
  creíbles.
- **Semáforo de límites**: OK, ALERTA (80% del límite) o EXCEDIDO. Un límite
  excedido no toca la cartera: se escala al comité, que actúa o lo acepta por
  escrito.

`python main.py ordenes` corre la misma comprobación sobre la cartera
**objetivo** y la añade a la hoja de órdenes: el comité ve si la propuesta
excede un límite antes de aprobarla, no después. `--sin-riesgo` la omite.

---

## Motor long/short de research (`python main.py`)

Una herramienta distinta del modelo multifactor, en `src/quant_engine/`. No
elige acciones: recibe una lista LONG y una SHORT del equipo de equity research
y mide si se sostienen cuantitativamente como cartera long/short.

```bash
python modelo.py                                 # interactivo: pide tickers y corre todo
python main.py research --long AAPL,MSFT,NVDA,AMZN,META --short TSLA,INTC,BA,PYPL,NKE
python main.py research --long ... --short ... --signals senales.csv   # modo B
```

Sin argumentos abre el modo interactivo; los subcomandos del modelo
multifactor (`universo`, `panel`, `backtest`, `ordenes`, `riesgo`) no cambian.
Defaults en `config/quant_engine.yaml`. Todo se guarda en
`output/quant_engine/`: reporte en texto, JSON, HTML, CSV de pesos, señales,
riesgo, correlación y plan de operación, y diez gráficos.

**Qué hace, en catorce secciones:** calidad de datos; rendimiento y riesgo por
activo; beta y alfa sobre retornos en exceso con errores HAC; momentum,
reversión a la media y régimen de volatilidad con umbrales declarados;
correlación, clusters y cuatro estimadores de covarianza con su número de
condición; exposición a factores construidos con ETF reales; Quant Score y
acuerdo con research; contraste LONG frente a SHORT; cinco métodos de
construcción con backtest walk-forward; beta y sector neutral; estrés; Monte
Carlo; robustez; plan de operación.

**El plan de operación (sección 14) corrige al research.** Con el Quant Score
alineado (score × +1 si research dice LONG, × −1 si dice SHORT):

| Score alineado | Veredicto | Qué se hace |
|---|---|---|
| ≥ +0,15 | CONFIRM | se opera como dijo research |
| entre −0,15 y +0,15 | REDUCE | se opera a la mitad del tamaño |
| entre −0,50 y −0,15 | NO TRADE | el dato contradice; revisar la tesis |
| ≤ −0,50 | FLIP | el modelo sugiere el lado contrario |

Para cada nombre da la acción (BUY / SELL SHORT / NO TRADE), la entrada
(último cierre), el stop a 2,5 × ATR(14), TP1 y TP2 a 1,5R y 3R (R = distancia
al stop; 50% en cada uno, stop a la entrada tras TP1) y el tamaño: el peso de
la cartera reconstruida con las direcciones corregidas, recortado para que un
stop no cueste más del 1% del capital. Los niveles van por volatilidad, no por
porcentajes fijos: un 5% es ruido en TSLA y mucho en MSFT. `P(TP1)` y `P(TP2)`
son la fracción de entradas pasadas en esa acción, con los mismos múltiplos de
ATR, que tocaron el objetivo antes que el stop en 63 sesiones: una tasa base
histórica, no un pronóstico. Todo configurable en la sección `trade_plan` del
YAML.

**Dos modos, y la diferencia importa:**

- **Modo A — instantánea.** La cesta de hoy aplicada al pasado. Los pesos se
  estiman walk-forward, pero la *selección* usa información de hoy: el
  research ya sabía qué subió. Es una trayectoria hipotética, no un backtest
  del proceso de research, y el reporte lo dice en la cabecera de la sección.
- **Modo B — señales históricas.** Un CSV `date,ticker,signal` con las
  recomendaciones tal como se emitieron (LONG, SHORT, FLAT). Esto sí es un
  backtest.

**Decisiones de diseño que no son obvias:**

- **El signo lo pone research y ningún optimizador lo cambia.** Así la
  exposición bruta es lineal y todas las optimizaciones son convexas.
- **Paridad de riesgo long/short por cambio de signo.** Con Σ' = SΣS, las
  contribuciones de riesgo de `w` en Σ son las de `x = |w|` en Σ'.
- **Beta neutral libera la neta.** Beta cero y neta cero a la vez rara vez
  caben; se fija la bruta y se reporta la neta resultante.
- **Tope por nombre 35%, no 20%.** Con 5 nombres por pata, 5 × 20% = 100% obliga
  a equiponderar y los cinco métodos dan la misma cartera. El motor avisa si
  vuelve a pasar.
- **Monte Carlo con deriva neutra.** Los retornos del modo A llevan dentro la
  selección retrospectiva; remuestrearlos tal cual prometía +37% esperado a un
  año. Se centran en la tasa libre de riesgo: la simulación mide riesgo, no
  retorno.
- **Coste de préstamo de los cortos** (0,25% anual por defecto), que la
  especificación original no pedía: sin él cualquier long/short sale
  sobreestimado.
- **Sin `statsmodels.api`.** El Control de aplicaciones de Windows de la
  máquina del club bloquea una de sus extensiones compiladas; se usa la ruta
  directa del OLS (`quant_engine/_sm.py`).

**Con WRDS** (si `python main.py wrds` conecta), el motor usa además:
Fama-French 5 factores + momentum en vez de ETFs, sectores GICS en vez de SIC,
valor y calidad de Compustat *point-in-time* (solo trimestres ya publicados),
consenso y revisiones de estimaciones de IBES, e interés corto — que en los
nombres cortos avisa de riesgo de squeeze. Sin WRDS corre igual, con Yahoo, y
lo dice en la cabecera del reporte.

El motor describe; no recomienda. Nunca dice "compra NVDA": dice "NVDA tiene el
mayor score de momentum del universo".

---

## Limitaciones declaradas

Cada una desplaza el resultado en una dirección conocida. Van también dentro de
cada reporte, no en un anexo.

1. **Sesgo de supervivencia.** El universo se construye con las cotizadas que
   existen hoy; las que quebraron o fueron excluidas no aparecen. **Infla** el
   retorno histórico, típicamente 1–2 puntos de CAGR al año en renta variable
   estadounidense. No se corrige sin datos de pago.

2. **Cobertura canadiense parcial.** Los fundamentales salen del XBRL de la SEC,
   que cubre a los emisores canadienses inscritos (40-F, 20-F, 10-K) y **no** a
   los exclusivos de TSX. El tramo canadiense es el de las grandes con presencia
   en EE.UU. — Royal Bank, TD, Enbridge, BCE, Suncor. Casi todos reportan en
   CAD: sus cifras se convierten a USD (flujos al tipo medio del periodo, saldos
   al de cierre, solo con tipos ya conocidos al presentar). Otras monedas no se
   convierten. Canadian National sigue fuera: presenta sus estados en 6-K, que
   el modelo no lee. Todo cotiza en USD, así que el libro no carga riesgo de
   divisa. Cubrir TSX puro exige un proveedor de pago y es una decisión del
   comité, no un pendiente técnico.

3. **Clasificación sectorial por SIC.** El SIC es de 1987 y no distingue bien el
   software moderno. La neutralización sectorial hereda ese ruido. Es el único
   sitio a cambiar si algún día hay GICS.

4. **Sin impacto de mercado.** Los costes son comisión, spread y slippage fijos
   (20 bps por lado por defecto). Razonable a escala de capital simulado;
   insuficiente a escala institucional.

5. **Fracciones de acción permitidas.** Realista en paper trading con IBKR; con
   un bróker que no las admita, las posiciones pequeñas redondean a la baja.

---

## Principios de trabajo

- **Cuestionar el backtest antes de entusiasmarse.** El primer reflejo ante un
  Sharpe alto es buscar el error, no la explicación económica.
- **Robustez por encima de resultados bonitos.** Un modelo que gana un 2% menos
  y se entiende vale más que uno que gana un 2% más y nadie sabe por qué.
- **Nada se calibra mirando el resultado.** Los parámetros del TOML son
  decisiones declaradas de antemano; cambiarlos hasta que el backtest mejore es
  sobreajuste con pasos extra.
- **Cada decisión de riesgo se puede explicar en una reunión del comité.** Si un
  tope no se puede justificar en una frase, sobra.

---

## Estado

Motor completo y probado (242 tests del modelo). Historial de iteraciones, con lo que cada
una cambió y por qué:

| Iteración | Cambio | Efecto |
|---|---|---|
| v1 | Motor completo | CAGR 16,9% vs SPY 14,7%; IR 0,24; exceso concentrado en 2021 y 2024 |
| v1.1 | Prioridad de etiquetas XBRL por periodo | Cobertura de ingresos 52% → 90%; ninguna métrica bajo el umbral |
| v1.2 | Prefiltro de liquidez por mediana, no máximo | 30% menos descargas de EDGAR |
| v2 | Excluir SIC 6221 y exigir fundamentales | CAGR 20,7%; IR 0,50 (ver advertencia abajo) |
| v2-Q | Rebalanceo trimestral (alternativa, no oficial) | Mismo exceso (6,05% vs 6,07%) e IR (0,50); costes 1,09% → 0,65% del NAV al año; drawdown −37,7% → −40,3% |

**Advertencia sobre v2.** La mejora es direccionalmente real — los nombres sin
fundamentales rendían +0,67% al mes frente al +1,96% de sus reemplazos — pero
solo explica unos 2 de los 3,9 puntos de CAGR ganados; el resto es dependencia
de trayectoria. La diferencia tiene p = 0,08. Y la regla se diseñó después de
ver el resultado de v1, así que v2 **no es fuera de muestra** para esa
decisión. Exceso honesto estimado, descontando el sesgo de supervivencia: 2-3
puntos anuales, no 6. **Tras la auditoría (v3)** el exceso medido es de 1,3
puntos (mensual) y 2,3 (trimestral) *antes* de descontar ese sesgo de 1-2
puntos: el exceso honesto está entre cero y un punto.

Pendiente, por orden:

- [ ] Decidir con el comité la cadencia de rebalanceo. Evidencia tras la
      auditoría (v3): la señal conserva el 85% de su poder predictivo con
      rebalanceo trimestral (antes se citaba 89%), pero con tres meses de
      retraso el IC ya no es significativo (t = 1,68). Backtest: el trimestral
      ahorra 0,62 puntos de costes al año y gana 0,9 puntos de exceso, con 2,9
      puntos más de drawdown máximo. `config/strategy_trimestral.toml`.
- [ ] Validar cada factor por separado. En Fama-MacBeth (v3, mensual) solo
      momentum es significativo (t = 2,46); valor (t = 0,34), calidad
      (t = 0,35) y baja volatilidad (t = −0,45) no aportan con esta muestra.
      Valor caía de t = 1,78 a 0,34 al corregir la capitalización.
- [ ] Decidir presupuesto para datos con deslistadas: sin ellos, el sesgo de
      supervivencia (1-2 puntos) es del mismo orden que el exceso.
- [ ] Revisar unidades de acciones en emisores con ADR (BTI sale con una
      capitalización absurda).
- [ ] Fijar con el comité la política de riesgo (`config/risk.toml`). Con los
      valores propuestos, la cartera del 23-sep-2026 excedía seis de diez
      límites. La cartera de v3 al 1-oct-2026 no excede ninguno y está en
      ALERTA en tres: tracking error ex-ante del 13,3% (límite 15%), un nombre
      con el 9,5% del riesgo (límite 10%) y −40,2% en el escenario tipo 2008
      (límite 45%). Volatilidad ex-ante 17,0%, beta 0,84, VaR 99% t de Student
      2,8%. El bootstrap da un drawdown p95 del −37% a un año. Ningún VaR pasa
      Kupiec sobre la historia (histórico p = 0,001, FHS p = 0,008).
- [ ] Revisar la prioridad de etiquetas de ingresos: `RevenueFromContractWithCustomer…`
      gana a `Revenues` y en emisores con ventas fuera de ASC 606 recoge solo una
      parte (Enbridge: 29 B CAD frente a 65 B CAD en 2025).
- [ ] Enlazar la historia de los CIK anteriores a una reorganización (Broadcom
      2018, Alphabet 2015 entran tarde al panel).
- [ ] Registro de decisiones del comité.
