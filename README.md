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

La caché se guarda en pickle comprimido (`.pkl.gz`), no en parquet. Smart App
Control de Windows bloquea desde octubre de 2026 la DLL de parquet de pyarrow
25.0.1 y el núcleo C++ de `cvxpy` en esta máquina; el proyecto ya no depende de
ninguno de los dos (los optimizadores usan la canonicalización en Python de
`cvxpy`). Las cachés parquet antiguas se migran solas si hay un pyarrow que
cargue.

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
│   │   ├── garch.py              GARCH(1,1) y Monte Carlo filtrado      [puro]
│   │   ├── factor_model.py       riesgo por sectores, estilos y propio  [puro]
│   │   ├── stress.py             históricos, hipotéticos, inverso, VaR estresado [puro]
│   │   ├── history.py            riesgo móvil y calibración del modelo  [puro]
│   │   ├── liquidity.py          coste de salida en crisis y capacidad  [puro]
│   │   ├── limits.py             límites de vigilancia y semáforo       [puro]
│   │   └── analysis.py           junta todo sobre una cartera           [puro]
│   ├── risk_report.py            reporte de riesgo y control previo a operar
│   └── cli.py                    los seis comandos
├── tools/                        cobertura y prueba de mutación de los tests de riesgo
└── tests/                        600+ tests, sin red, con datos sintéticos
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

4. **Los filtros de liquidez usan los datos de esa fecha**, no los de hoy.

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
- **Modelo factorial de riesgo.** Regresión transversal diaria del universo
  sobre sectores, los cuatro estilos del modelo y tamaño. Dice de dónde viene
  el riesgo: mercado y sectores, un estilo concreto o el riesgo propio de cada
  nombre. Aplica las exposiciones de *hoy*: si la cartera está hoy más cargada
  de un estilo que durante el año, ve más riesgo que Ledoit-Wolf, y el reporte
  lo avisa.
- **Volatilidad condicional GARCH(1,1) y Monte Carlo filtrado.** Estimado por
  máxima verosimilitud; los residuos históricos se reinyectan en la recursión
  desde la volatilidad de hoy. Da el VaR a 1, 21 y 63 sesiones con racimos de
  volatilidad y reversión a la media, para la estrategia y para la cartera
  actual sobre su historia reconstruida desde 2007.
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
  pérdida dada. **VaR estresado** (Basilea 2.5): el VaR de la cartera de hoy en
  la peor ventana de un año de su historia. **Choques de covarianza**:
  volatilidades ×1,5 y ×2 y correlaciones de crisis, por separado y juntas.
- **Historia y calibración del riesgo.** Volatilidad, beta y tracking error
  móviles frente a su historia. En cada rebalanceo del backtest se compara la
  volatilidad que el modelo de riesgo habría previsto con la realizada
  (estadístico de sesgo de Barra): si sale por encima de su intervalo, todos
  los VaR ex-ante son optimistas.
- **Liquidez y capacidad**: días para liquidar cada posición al 20% del volumen
  diario; coste de liquidar el libro entero en condiciones normales y de crisis
  con impacto por la ley de la raíz cuadrada; y curva de capacidad: el NAV al
  que el impacto de rebalancear, que el backtest no cobra, llega al 1% anual.
- **Tiempo bajo el agua** en el Monte Carlo: la racha más larga por debajo del
  máximo, mediana y p95, a 1, 3 y 5 años.
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
python main.py research --long ... --short ... --positions AAPL:300,TSLA:-100
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
histórica, no un pronóstico.

Las correcciones y los recortes cambian el libro que construyó el optimizador:
quitar un corto de beta alta (INTC, beta ~3) deja la cartera larga de mercado
sin que nadie lo haya decidido. Si la beta del libro final pasa de 0,05, el
plan añade una línea **BETA HEDGE** en el benchmark (vender o comprar SPY) que
la devuelve a cero. Todo configurable en la sección `trade_plan` del YAML.

**Señal final: qué hacer hoy.** El modo interactivo pregunta las posiciones
actuales (`AAPL:300,TSLA:-100`, negativo = corto; Enter = ninguna). Lo último
que imprime el reporte es una línea por ticker con la orden para pasar de lo
que hay al plan:

| Señal | Cuándo |
|---|---|
| BUY / SELL SHORT | abrir, o ampliar del mismo lado |
| SELL / BUY TO COVER | reducir o cerrar |
| FLIP TO LONG / FLIP TO SHORT | la posición está del lado contrario al plan |
| HOLD | mismo lado y el ajuste no llega al 0,5% del capital (`hold_band`) |
| NO TRADE | no hay posición y el plan no abre |
| NOT ANALYZED | posición en un ticker que no está en la lista: no se toca |

Un NO TRADE cierra la posición si va en la dirección de research (el dato la
contradice) y la mantiene si va en contra (el dato tampoco respalda a
research). Una posición en el benchmark se trata como la cobertura. Las
órdenes se guardan en `orders_<fecha>.csv`.

**Qué esperar: el backtest de cada trade y del plan.** El plan ya no da solo
niveles. Para cada nombre simula la regla exacta —entrada al cierre, stop a
2,5 ATR, mitad en TP1 con el stop a la entrada, resto en TP2 o a las 63
sesiones— abierta cada día de los últimos 3 años, neta de costes y, en cortos,
de préstamo. Da aciertos, E(R) (resultado medio en múltiplos del riesgo),
profit factor, días medios y el P&L esperado con el tamaño de hoy. Luego abre el
plan completo en cada fecha histórica a la vez (mismas fechas para todos los
nombres, para conservar su correlación) y da la media, el rango del 90%, la
probabilidad de perder y el peor caso. Avisa si se opera un nombre con
expectativa histórica negativa, y suma el backtest de la canasta como cartera y
su Monte Carlo. Supuestos conservadores: si un día toca stop y objetivo cuenta
como stop, y un gap que salta el stop se cobra al peor precio del día. Es una
tasa base histórica, no un pronóstico. Cada E(R) lleva su **t de Newey-West**:
las entradas de días seguidos comparten casi todo el camino, así que 756
entradas son unas pocas decenas de operaciones independientes. Con |t| < 2 la
expectativa no se distingue de cero.

**¿Predice el Quant Score? Se valida en cada corrida.** Los veredictos se
apoyan en el Quant Score, así que el motor lo pone a prueba. Recalcula el mismo
score en cada uno de los últimos 36 fines de mes, con los datos que había ese
día: precios hasta esa fecha y trimestres de Compustat ya publicados. Luego
mide si ordenaba bien el retorno del mes siguiente (IC de rangos, con su t de
Newey-West, el mismo cálculo del modelo multifactor). El universo es la canasta
más los 50 pares de referencia; con 5 nombres un IC mensual es ruido. Da el IC
del score y el de cada componente. Con WRDS entran todos los componentes:
valor y calidad con los trimestres ya publicados, analistas con los resúmenes
mensuales de IBES hasta esa fecha, e interés en corto con 14 días de retraso
de publicación. Sin WRDS solo se validan los de precio, y el reporte dice
cuáles faltan. Se mide también a 63 sesiones, el horizonte del plan; ahí las
ventanas se solapan y la t es de Newey-West. Los pares son las grandes de hoy:
el sesgo de supervivencia hace la prueba optimista. La primera corrida, en
octubre de 2026 y sin WRDS, dio IC −0,004 (t −0,1) a un mes y +0,014 (t +0,4)
a 63 sesiones: sin evidencia de que el score de precio prediga en ese
universo. Ningún componente llegó a |t| = 2. Los pesos del YAML no se
reajustan con este resultado: elegir pesos con la misma muestra que los
evalúa es sobreajustar.

**El score validado del modelo multifactor entra como evidencia.** El Quant
Score del motor long/short no ha demostrado que prediga. El compuesto del
modelo multifactor sí: en su backtest, IC 0,024 con t de Newey-West 3,0 en 177
meses y quintiles casi perfectamente ordenados. Cada ficha dice en qué
percentil del último panel del multifactor está el nombre (`data/multifactor.py`,
de disco, sin red). Si el nombre está en su quintil extremo:

- respalda la dirección (LONG en Q1, SHORT en Q5): la convicción sube un
  escalón, y un FLIP sin evidencia del Quant Score va a tamaño completo;
- la contradice: la convicción baja un escalón.

La evidencia se lee del último reporte oficial de backtest
(`output/reports/backtest_AAAAMMDD_HHMM.md`). Sin reporte, con un IC sin
significancia o con el panel de hace más de 45 días, el percentil se muestra
pero no mueve nada. Los veredictos no cambian.

**Nombres fuera del universo del multifactor.** La ficha dice qué regla del
modelo los deja fuera: precio por debajo de 5 USD, liquidez, tamaño, historia,
el tope de los 1.200 nombres más líquidos (`universe.max_names`), o falta de
valor o de calidad. Además los puntúa como referencia: sus métricas salen con las mismas
funciones del modelo (fundamentales point-in-time de la SEC, o el TTM de
Compustat si no presentan allí; momentum 12-1 y volatilidad). Se añaden a la
sección cruzada de la última fecha y se puntúan con `build_scores`. Ese score
no mueve la convicción, porque la validación cubre solo el universo. En
octubre de 2026, AMC estaba fuera por precio y en el percentil 10, contra el
LONG de research. GOLD estaba fuera por el tope de liquidez: 18,6 M USD al
día, frente a 49,6 M del último nombre que entró. Quedaba en el percentil 100,
contra el SHORT de research.

**¿Puede el multifactor decidir los veredictos?** Medido en los pares del motor
(unas 41 grandes empresas por fecha, 2012–2026), su IC es 0,018, tan fuerte
como en el universo completo (0,019). Pero con t 1,06: con 41 nombres el ruido
de un IC mensual es grande. Por eso entra como evidencia solo en los quintiles
extremos y no sustituye al Quant Score en los veredictos.

**Valor relativo frente al DCF.** El DCF es valor absoluto. El factor de valor
del multifactor (z sectorial de los rendimientos por beneficio, flujo de caja,
libros y ventas) es valor relativo. La ficha y `python main.py dcf` muestran
los dos y avisan si apuntan en direcciones opuestas.

**Ir contra research a tamaño completo exige esa evidencia.** Un FLIP va a
tamaño completo solo si la t del IC llega a `trade_plan.flip_min_ic_t` (2 por
defecto). Sin esa evidencia, `trade_plan.unproven_flip` decide: `reduce` (por
defecto) toma el lado contrario a `reduce_size` (50%) con convicción LOW;
`no_trade` lo deja en NO TRADE. El plan lo dice nombre por nombre.

**Una señal clara por nombre, y cómo ejecutarla.** Al final del plan, en la
terminal y al principio del HTML, cada nombre tiene una ficha:

- **Señal**: LONG, SHORT o NO TRADE; si va con research o contra; y una
  convicción HIGH / MEDIUM / LOW. La convicción combina la fuerza del score,
  la evidencia de la validación y la t del backtest de la regla.
- **Por qué**: los componentes que respaldan a research y los que van en
  contra, con los números que hay detrás (momentum 12-1 −14%, ROE 25%,
  revisión de estimaciones +5,8%, 11,7% de las acciones en corto…).
- **Fundamentales**, con aviso si un valor no es creíble (flujo de caja libre
  >50% del precio, ROE >200%).
- **Datos a verificar**: último precio viejo o con salto de más del 25% (es
  la entrada del plan), saltos con forma de split, barras reparadas.
- **Cómo ejecutarla, paso a paso**. Para un corto, el primer paso es
  localizar el préstamo, con el riesgo de squeeze y el coste asumido (más de
  5% al año invalida la expectativa). Luego vienen SELL SHORT, BUY STOP por
  encima de la entrada, BUY TO COVER en TP1 y TP2, salida por tiempo con
  fecha, margen Reg T y dividendos. Para un largo, lo mismo con SELL STOP. Si
  ya se tiene el lado contrario, primero se cierra. Se avisa de los
  resultados antes de la salida (fecha de Yahoo) y de la moneda.
- **Tamaño**: el del libro long/short. Si no hay libro (queda una sola pata),
  un tamaño propio por riesgo: el stop cuesta el 1% del capital (por la
  convicción), con tope de nocional. El total de todas esas posiciones se
  muestra junto, con una sola cobertura en SPY.

**Los números se comprueban antes de salir.** `trade_plan.check_plan` verifica:
- precios positivos y stop y objetivos del lado correcto;
- acciones = nocional / precio convertido a USD;
- riesgo por operación bajo el tope;
- órdenes que cuadran con la posición actual;
- riesgo total = suma de las órdenes;
- beta tras la cobertura que cuadra.

Un fallo sale en rojo con "do not send these orders". Además, la limpieza de
precios repara barras con máximo o mínimo incoherente y excluye saltos de x20
en un día (moneda o unidad equivocada). También avisa de un último precio de
hace 3 o más sesiones, de un último cierre que salta más del 25% y de saltos
con forma de split.

**Valores extremos.** Antes de cada z-score, la sección cruzada se recorta al
percentil 5–95 si hay 20 nombres o más. Un solo valor extremo, como el flujo
libre de GOLD (104,6% del precio, por el capital circulante de un comerciante
de metales), inflaba la desviación y aplastaba a los demás nombres hacia cero.

**Yahoo a veces corta.** En lotes grandes yfinance da por "sin datos" a
empresas vivas: el 17 de octubre faltaron 16 de los 50 pares. Lo que falta se
reintenta en lotes de 25. Si aun así falla y hay caché de días anteriores, se
usa y se avisa con la fecha del último precio.

**DCF automático (`src/quant_engine/valuation/`).** Cada nombre de la canasta
se valora por flujo libre de la empresa (FCFF), y también se puede valorar
suelto:

```bash
python main.py dcf PLTR AAPL ATZ.TO          # SEC XBRL; Excel con fórmulas por ticker
python main.py dcf ATZ.TO --wrds             # Compustat y consenso de IBES (aprobación Duo)
```

Misma estructura que los Excel del club (PLTR, CNL), con lo que pediría un
comité:

- **Datos**: Compustat si hay WRDS (cubre Canadá); si no, los 10-K/20-F/40-F de
  la SEC. Las ventas son la MAYOR de las etiquetas de cada ejercicio, porque
  un desglose nunca supera al total. Gold.com publica 25.513 M de "Revenues" y
  618 M de un desglose, y con prioridad de etiquetas sus ventas caían de 9.699
  a 177.
- **Supuestos con regla escrita** y anotada en el resultado:
  - Crecimiento del año 1 del consenso de IBES; sin consenso, de la historia,
    acotado a [−20%, +40%]. Baja en línea recta hasta el terminal en el año 10.
  - Margen: se mantiene el actual. El escenario bajista lleva la reversión.
  - Impuesto efectivo que sube al marginal del país.
  - D&A, CapEx y circulante: medianas de 3 años.
- **Crecer cuesta capital**: la reinversión neta nunca baja de lo nuevo vendido
  entre ventas/capital. Se usa el de la propia empresa si es mayor que 5, con
  tope en 25.
- **Terminal coherente**: el FCFF del año N+1 reinvierte g/RONIC del NOPAT, con
  RONIC = WACC. El Excel de PLTR crecía al 3% perpetuo con un CapEx del 0,6%
  de las ventas: un ROIC infinito, y el 71% del valor.
- **WACC**:
  - rf del bono a 10 años del día (Yahoo ^TNX);
  - beta de la regresión contra SPY, ajustada por Blume;
  - prima de mercado del 4,5%;
  - coste de deuda por rating sintético (cobertura de intereses, escala de
    Damodaran);
  - pesos de mercado.

  El descuento es a mitad de año, también el del terminal.
- **Escenarios** bajista, base y alcista (25/50/25); Monte Carlo de 5.000
  sorteos; **DCF inverso**: qué crecimiento descuenta el precio de hoy;
  sensibilidades WACC × g y crecimiento × margen.
- **Comprobaciones críticas.** Con cualquiera de estas, el DCF sale NOT
  DECISION-GRADE y no opina sobre la señal:
  - margen fino (< 3%);
  - patrimonio negativo;
  - WACC a menos de 2 puntos de g;
  - margen terminal negativo;
  - acciones solo de la portada;
  - intereses sin deuda;
  - un salto de ventas de más de 3x.

  El veredicto sale del caso base, no del ponderado: con el suelo de una acción
  en 0, el ponderado se sesga.
- **Excel con fórmulas vivas** (`output/quant_engine/dcf/`): inputs en azul,
  fórmulas en negro, selector de escenario. Recalculado en Excel 16, da lo
  mismo que el motor hasta 1e-13 en los tres escenarios de AAPL y PLTR.
- **Supuestos del analista** por ticker en `valuation.overrides` del YAML.

El DCF informa a la ficha de cada nombre: si respalda o contradice la señal.
No cambia los veredictos ni el Quant Score: un factor de valor por DCF habría
que validarlo antes.

**Rango de tiempo de cada corto.** La ficha de un corto trae su ventana con
fechas, sacada de la misma regla abierta cada día de los últimos años en ese
valor (`trade_plan.holding_window`):

- cuándo abrir; si no entra en dos sesiones, recalcular;
- duración típica: percentiles 25–75 de los días hasta la salida, con fechas;
- cuándo suele llegar el TP1, y una **fecha de revisión**: el percentil 75 de
  ese día. Si para entonces no llegó, se mira qué fracción de las operaciones
  aún vivas lo tocó después. Si está por debajo del break-even del TP1, se
  cubre en esa fecha. La ficha muestra la E(R) histórica con y sin ese stop
  por tiempo (dentro de muestra, y lo dice);
- salida obligatoria a las 63 sesiones;
- coste del préstamo en la ventana. Con mucho interés en corto, el tramo
  probable: tasa general por debajo del 5% de las acciones, "warm" hasta el
  15%, difícil de pedir prestada por encima (D'Avolio, 2002). Es una regla
  general: el coste real lo da el broker;
- fecha ex-dividendo dentro de la ventana: en corto el dividendo lo pagas tú.

En octubre de 2026 el stop por tiempo mejoraba la E(R) de los tres cortos de
la canasta (ARES −0,39R → −0,25R). Eso es dentro de muestra. La prueba que
cuenta va en cada ficha y en el resumen: el día de revisión se elige con la
primera mitad de la historia del nombre y se mide en la segunda. El plan dice
en cuántos nombres mejora fuera de muestra. Solo con una mejora consistente
(3/4 de los nombres o más, en varias canastas) tendría sentido convertirlo en
regla.

Los largos llevan la misma ventana, sin préstamo ni dividendo.

**Gap de resultados medido.** Si hay resultados antes de la salida, la ficha
da el movimiento típico de ese día en el nombre y lo compara con la distancia
al stop. Con WRDS sale de las fechas reales de publicación (`rdq` de
Compustat): el mayor |retorno| entre ese día y el siguiente. Sin WRDS, de una
aproximación: los cuatro días de mayor movimiento de cada año. La ficha dice
cuál de las dos usó.

**Salidas a bolsa recientes.** Un nombre excluido por historia corta (menos de
126 sesiones) pero con 30 o más tiene una ficha limitada, con niveles por ATR
del lado de research, lectura técnica y DCF. No tiene Quant Score, backtest ni
tamaño: informa una decisión del analista y lo dice.

**Lectura técnica para la entrada.** La ficha de cada nombre añade:
- tendencia (SMA50/200 y su pendiente) y fuerza (ADX con +DI/−DI);
- RSI de Wilder, cruce del MACD y %B de Bollinger con la compresión de
  volatilidad;
- distancia a máximos y mínimos de 52 semanas, soportes y resistencias por
  pivotes confirmados, y volumen.

De ahí salen consejos de ejecución:
- comprar extendido: esperar un retroceso a la SMA20;
- corto sobrevendido: esperar un rebote;
- ir contra una tendencia fuerte;
- un stop justo encima de un soporte: un test lo saltaría;
- una resistencia antes del TP1.

No cambian la dirección. Además, dos factores con evidencia académica entran
con peso 0, validados en cada corrida: la proximidad al máximo de 52 semanas
(George y Hwang, 2004) y el momentum residual frente al mercado (Blitz, Huij y
Martens, 2011). Con WRDS, la validación de octubre de 2026 dio t −0,6 y +0,6:
se quedan en 0.

**Nombres de fuera de EE.UU.** El motor convierte a USD todo lo que cotiza en
otra moneda (`ATZ.TO` en CAD, `VOD.L` en peniques), con el tipo de cambio de
cada día de Yahoo (`CADUSD=X`). Retornos, beta frente a SPY, correlaciones,
liquidez, backtests y tamaños quedan en dólares. Entrada, stop y objetivos se
dan en la moneda en que cotiza el nombre, que es donde se ponen las órdenes.
Las acciones salen de dividir el nocional en USD por el precio convertido; sin
la conversión, un nombre canadiense salía un 30% más pequeño, y uno de
Londres, que cotiza en peniques, unas 76 veces más pequeño. La moneda se
deduce del sufijo de bolsa (`loader.QUOTE_CURRENCY`); un sufijo que no está
ahí se avisa y no se convierte. Para Compustat, `ATZ.TO` se busca como `ATZ.`
y `TECK-B.TO` como `TECK.B`, que es como Compustat guarda los listados de
Toronto. IBES se toma de una emisión hermana si la de Toronto no tiene
identificador, y las estimaciones se leen también del fichero internacional.
Los ratios de valor usan el precio en la moneda en que reporta la empresa.
Antes, ningún nombre canadiense tenía fundamentales ni analistas: se puntuaba
solo por precio.

**Canastas solo LONG.** Si todos los tickers van LONG (o con
`python main.py research --long ARES,AMC` sin `--short`), el motor trabaja en
modo solo LONG:

- la pata larga invierte el 100% del capital (no el 200% bruto del long/short);
- el Quant Score se calcula contra un **universo de referencia** de 50 acciones
  grandes y líquidas de los 11 sectores (`scoring.reference_peers` en el YAML).
  Dentro de una canasta de dos nombres no hay sección cruzada, y dentro de una
  canasta solo LONG la mitad quedaría "en contra" por construcción, porque los
  z-scores suman cero. Lo mismo se aplica a cualquier canasta de menos de 5
  nombres. Los pares solo puntúan: nunca se operan ni salen en las tablas;
- un FLIP no abre un corto: queda NO TRADE;
- no hay cobertura de beta: en una cartera solo LONG la beta de mercado es la
  exposición buscada;
- las secciones que necesitan una pata corta (largos frente a cortos, beta y
  sector neutral) lo dicen y se saltan.

**Cómo se entrega.** En pantalla sale solo lo que hay que leer para operar:
cabecera, resumen de 6 líneas, sección 14 y la señal final (~120 líneas). Las
trece secciones de análisis van a disco y al menú `[5]`. Además, cada corrida
genera `trade_plan_<fecha>.html`: una página autocontenida (CSS y gráficos
dentro) con órdenes, qué corrigió el modelo y por qué (los componentes que
empujan a favor y en contra), niveles con P(TP) frente a su break-even,
cobertura, un gráfico por posición con entrada, stop y objetivos, y los
límites. En modo interactivo se abre sola en el navegador; se imprime a PDF
con Ctrl+P. Por comando: `--open` la abre, `--full` imprime todo en pantalla.

**Si WRDS no conecta, el plan lo dice en rojo.** Sin WRDS faltan valor,
calidad, analistas e interés en corto (la mitad del Quant Score) y los
veredictos cambian: en una misma fecha INTC salió NO TRADE con WRDS y FLIP sin
él. La conexión se prueba primero con un tiempo máximo (45 s,
`WRDS_CONNECT_TIMEOUT`); si WRDS la deja esperando, casi siempre es una
aprobación Duo pendiente en el celular del titular de la cuenta.

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
   en EE.UU. — Royal Bank, Enbridge, Canadian National, BCE, Suncor. Todo cotiza
   en USD, así que el libro no carga riesgo de divisa. Cubrir TSX puro exige un
   proveedor de pago y es una decisión del comité, no un pendiente técnico.

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

Motor completo y probado (600+ tests). Historial de iteraciones, con lo que cada
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
puntos anuales, no 6.

Pendiente, por orden:

- [ ] Decidir con el comité la cadencia de rebalanceo. Evidencia: la señal
      conserva el 89% de su poder predictivo con rebalanceo trimestral, y el
      backtest lo confirma — mismo exceso neto, 0,44 puntos menos de costes al
      año, 2,5 puntos más de drawdown máximo. `config/strategy_trimestral.toml`.
- [ ] Validar cada factor por separado. Calidad (t = 0,93) y baja volatilidad
      (t = −0,07) no aportan en Fama-MacBeth con esta muestra.
- [ ] Decidir presupuesto para datos con deslistadas: sin ellos, el sesgo de
      supervivencia (1-2 puntos) es del mismo orden que el exceso.
- [ ] Revisar unidades de acciones en emisores con ADR (BTI sale con una
      capitalización absurda).
- [ ] Fijar con el comité la política de riesgo (`config/risk.toml`). Con los
      valores propuestos, la cartera del 23-sep-2026 excede seis de diez
      límites: tracking error ex-ante del 19,9%, el 48% del riesgo en
      tecnología con el 23% del peso (memoria y almacenamiento: SNDK, MU, WDC,
      STX), SNDK sola con el 15% del riesgo, y −52% en el escenario tipo 2008.
      El bootstrap da un drawdown p95 del −34% a un año. El VaR FHS es el único
      de los tres que pasa Kupiec (p = 0,06); el histórico y el normal EWMA se
      exceden de más.
- [ ] Registro de decisiones del comité.
