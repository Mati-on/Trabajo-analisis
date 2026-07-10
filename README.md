# Analisis estadistico de ventas - Cruz Morada

Solucion computacional para procesar, limpiar, analizar y modelar el archivo
`data/ventas_completas.csv` del trabajo practico de Computacion Paralela y
Distribuida.

El proyecto esta disenado para trabajar con un CSV grande sin cargarlo completo
en memoria durante la etapa de validacion. Para eso se usa procesamiento por
chunks y ejecucion paralela con `concurrent.futures`.

## Objetivo tecnico

El pipeline busca resolver cinco necesidades del enunciado:

1. Validar y filtrar transacciones corruptas o inconsistentes.
2. Procesar el archivo grande por bloques para controlar memoria.
3. Paralelizar la validacion de chunks para aprovechar CPU.
4. Generar un dataset limpio con variables derivadas.
5. Producir resultados estadisticos, graficos, hipotesis y modelo predictivo.

## Flujo general

```text
data/ventas_completas.csv
        |
        v
scripts/procesar_paralelo.py
        |
        v
data/ventas_validas.parquet temporal + logs.txt
        |
        v
scripts/preprocesamiento.py
        |
        v
data/ventas_limpias.parquet + resultados/resumen_preprocesamiento.json
        |
        +--> scripts/analisis_exploratorio.py
        |        |
        |        v
        |   plots/*.png + resultados/*.json
        |
        +--> scripts/inferencia_modelado.py
                 |
                 v
            resultados/pruebas_hipotesis.json
            resultados/metricas_modelo.json
            plots/modelo_*.png
```

El flujo se ejecuta de forma encadenada. La validacion paralela produce un
archivo validado temporal en Parquet; el preprocesamiento depende de ese archivo
y genera el dataset limpio en Parquet. El analisis exploratorio y el modelo
dependen del archivo limpio. Esta separacion permite auditar cada etapa sin
mantener CSV intermedios pesados.

## Alcance del paralelismo

No todas las etapas del proyecto se ejecutan en paralelo. La paralelizacion se
aplica donde entrega mayor beneficio y menor riesgo: validacion inicial, tablas
del analisis exploratorio e inferencia/modelado por tareas independientes.

| Etapa | Script | Modo de ejecucion | Justificacion |
| --- | --- | --- | --- |
| Exploracion del volumen | `explorar_csv.py` | secuencial por streaming | solo cuenta filas y chunks; no necesita paralelismo |
| Validacion de transacciones | `procesar_paralelo.py` | paralelo por chunks | cada fila puede validarse independientemente |
| Preprocesamiento | `preprocesamiento.py` | secuencial con pandas | requiere operaciones globales como frecuencia por cliente, outliers y z-score |
| Analisis exploratorio | `analisis_exploratorio.py` | paralelo parcial con hilos | calcula estadisticas, normalidad, correlaciones y pruebas en paralelo; genera graficos en orden seguro |
| Inferencia y modelo | `inferencia_modelado.py` | paralelo parcial con hilos | ejecuta hipotesis, modelo predictivo y diagnostico lineal como tareas independientes; protege graficos con lock |
| Orquestacion | `run_pipeline.py` | secuencial | ejecuta etapas en orden y detiene el flujo si una falla |

Esta decision cumple el requisito de procesamiento paralelo o por bloques sin
forzar paralelismo en etapas donde aumentaria complejidad, consumo de memoria o
riesgo de inconsistencias. En el EDA se paraleliza solo la parte tabular porque
las tareas son independientes y escriben archivos JSON distintos. En inferencia
y modelado se paralelizan las tareas completas, no las filas, para mantener la
validez estadistica de cada prueba.

## Estructura del proyecto

```text
trabajo analisis/
|-- data/
|   |-- ventas_completas.csv          # CSV real entregado para el trabajo
|   `-- transacciones_prueba.csv      # dataset pequeno para pruebas
|-- plots/                            # graficos EDA, temporal y modelo
|-- resultados/                       # estadisticas, hipotesis y metricas
|-- scripts/
|   |-- explorar_csv.py               # inspecciona volumen y cantidad de chunks
|   |-- validacion.py                 # reglas de validacion por transaccion
|   |-- procesar_paralelo.py          # validacion paralela por chunks
|   |-- preprocesamiento.py           # limpieza, derivadas, outliers y escala
|   |-- analisis_exploratorio.py      # estadisticas, pruebas EDA y graficos
|   |-- inferencia_modelado.py        # hipotesis y regresion no lineal
|   |-- dashboard.py                  # ventana interactiva de resultados
|   `-- run_pipeline.py               # orquestador del flujo completo
|-- INFORME_TECNICO.md
|-- plandeaccion.md
|-- requirements.txt
|-- logs.txt
`-- README.md
```

## Scripts utilizados

### `scripts/explorar_csv.py`

Inspecciona el archivo `ventas_completas.csv` antes de procesarlo. Cuenta el
total de transacciones, columnas y cuantos chunks se formarian usando el tamano
configurado.

Se usa para dimensionar el trabajo y justificar el procesamiento por bloques.

### `scripts/validacion.py`

Contiene las reglas comunes de validacion:

- cantidad correcta de columnas;
- campos obligatorios no vacios;
- enteros validos para `SKU`, `UNIDADES`, `BOLETA` y `LOCAL`;
- montos positivos;
- descuento entre `0` y `1`;
- RUT chileno valido;
- UUID valido para `CODIGO CLIENTE`;
- fechas validas;
- edad calculada entre `FECHA` y `FECHA NACIMIENTO` dentro de rango plausible
  (`0` a `110` anos);
- genero en valores esperados.

Se separo en un modulo propio para que pueda ser usado por la version paralela y
por herramientas de depuracion sin duplicar logica.

### `scripts/procesar_paralelo.py`

Es la etapa principal de carga eficiente. Lee el CSV por chunks y envia cada
bloque a workers paralelos.

Por que se usa:

- evita cargar los 665 MB del archivo completo en memoria durante la validacion;
- permite aprovechar varios nucleos de CPU;
- registra descartes en `logs.txt`;
- genera `data/ventas_validas.parquet` como dataset temporal;
- mantiene salida determinista escribiendo los chunks en orden.

Parametros relevantes:

- `--chunksize`: cantidad de filas por bloque. Valor usado: `100000`.
- `--workers`: cantidad de workers paralelos.
- `--executor`: `process` para procesos o `thread` como alternativa en Windows.
- `--sample`: limita filas para pruebas rapidas.

Funcionamiento interno:

1. El proceso principal abre `ventas_completas.csv`.
2. Lee el encabezado una sola vez.
3. Agrupa filas en chunks de `--chunksize`.
4. Envia cada chunk a un worker.
5. Cada worker ejecuta `validar_transaccion` sobre sus filas.
6. El worker devuelve al proceso principal:
   - filas validas;
   - mensajes de descarte para `logs.txt`;
   - conteos de filas leidas, validas y descartadas.
7. El proceso principal escribe los resultados en disco.
8. Cada chunk informa descartes totales y descartes por edad.

Los workers no escriben directamente en archivos compartidos. Esta es una
decision preventiva importante: evita que varios procesos o hilos intenten
escribir simultaneamente en el dataset validado o `logs.txt`.

### `scripts/preprocesamiento.py`

Transforma `ventas_validas.parquet` en `ventas_limpias.parquet`.

Acciones principales:

- convierte fechas y columnas numericas;
- elimina filas sin datos esenciales;
- imputa valores faltantes simples;
- calcula `MONTO POR UNIDAD`;
- calcula `EDAD`;
- calcula `FRECUENCIA COMPRA`;
- marca outliers por IQR en monto y unidades;
- genera columnas normalizadas con sufijo `_z`;
- guarda `resultados/resumen_preprocesamiento.json`.

El formato recomendado es Parquet porque reduce peso en disco y acelera las
lecturas posteriores. CSV queda soportado solo como alternativa de
compatibilidad.

Los outliers se marcan, no se eliminan automaticamente. Esto permite analizarlos
y justificar decisiones en el informe.

### `scripts/analisis_exploratorio.py`

Genera el analisis exploratorio requerido por la rubrica:

- estadisticas descriptivas;
- pruebas de normalidad;
- correlacion de Spearman;
- pruebas de asociacion: Chi-cuadrado `CANAL` vs `LOCAL` y prueba
  complementaria `CANAL` vs `GENERO`;
- Spearman especifico entre `UNIDADES`, `MONTO APLICADO` y
  `PORCENTAJE DESCUENTO`, con nota cuando una variable no tiene variacion;
- ANOVA y Kruskal-Wallis para `MONTO APLICADO` por `CANAL` y por `LOCAL`;
- histogramas con densidad;
- boxplot por canal;
- matriz de correlacion;
- serie temporal diaria;
- descomposicion temporal y ACF/PACF cuando hay suficientes dias.

Las salidas tabulares independientes se ejecutan en paralelo con
`ThreadPoolExecutor` usando `--workers`. Cuando el EDA se ejecuta desde
`scripts/run_pipeline.py`, usa la misma cantidad de workers configurada para la
validacion paralela. Por ejemplo, con `--workers 4`, la validacion trabaja con 4
workers y el EDA tambien ejecuta sus calculos tabulares con hasta 4 workers. No
son los mismos procesos o hilos: cada etapa crea sus propios workers. Se usan
hilos para compartir el DataFrame limpio en memoria y evitar copiar millones de
filas. La generacion de graficos queda secuencial porque Matplotlib no es seguro
para dibujar muchas figuras desde varios hilos al mismo tiempo.

Para graficos pesados se usa una muestra reproducible controlada por
`CPYD_SEED`, evitando graficos lentos o ilegibles con millones de filas.

### `scripts/inferencia_modelado.py`

Ejecuta pruebas de hipotesis y un modelo predictivo.

Hipotesis implementadas:

1. El ticket promedio en APP es mayor que en WEB.
2. El descuento afecta significativamente las unidades vendidas.
3. El monto promedio difiere entre canales de compra.
4. El descuento se asocia con el monto comprado.
5. La edad se asocia con el monto comprado.
6. El monto promedio difiere segun genero.

La hipotesis descuento-unidades se marca como no evaluable cuando `UNIDADES` no
presenta variacion. Se conserva en los resultados para demostrar que se reviso
el ejemplo metodologico del enunciado.

Modelo:

- regresion no lineal para predecir `MONTO APLICADO`;
- `HistGradientBoostingRegressor` entrenado sobre `log1p(MONTO APLICADO)`;
- particion train/test con semilla reproducible;
- metricas MAE, RMSE y R2;
- graficos real vs predicho y residuos.
- regresion lineal interpretable complementaria con coeficientes, R2 ajustado,
  VIF, normalidad de residuos y homocedasticidad.

Se usa un modelo no lineal porque el monto presenta fuerte asimetria, outliers y
relaciones no lineales con variables como `SKU`, `LOCAL`, descuento y canal. La
transformacion `log1p` reduce el efecto de la cola larga de montos.

Esta etapa acepta `--workers`. Cuando se ejecuta desde `run_pipeline.py`, usa la
misma cantidad configurada para validacion y EDA. El paralelismo se aplica por
tareas completas:

- pruebas de hipotesis;
- modelo predictivo no lineal;
- diagnostico de regresion lineal.

Cada tarea escribe archivos distintos. Los PNG se generan con un bloqueo interno
para evitar conflictos de Matplotlib.

### `scripts/run_pipeline.py`

Orquesta todas las etapas en orden. Es el comando recomendado para ejecutar el
trabajo completo.

Si una etapa falla, el pipeline se detiene. Esto evita generar graficos o modelo
sobre archivos incompletos.

Al finalizar correctamente, abre automaticamente `scripts/dashboard.py` usando el
dataset limpio recien generado y `logs.txt`. Si se necesita ejecutar solo en
consola, se puede agregar `--no-dashboard`.

### `scripts/dashboard.py`

Abre una ventana local para revisar resultados sin volver a generar archivos.
Carga `data/ventas_limpias.parquet` y `logs.txt`, muestra resumen del periodo,
descartes globales y graficos interactivos actualizados por rango de fechas.

Incluye:

- filtro por fecha inicio y fecha fin usando solo fechas existentes dentro del
  rango real del dataset;
- resumen de transacciones, monto total, ticket promedio y outliers;
- ventas diarias;
- monto por canal;
- distribucion de montos;
- matriz de correlacion del periodo.
- pestanas principales alineadas con el enunciado: `Preprocesamiento`,
  `Descriptiva`, `Normalidad`, `Correlaciones`, `Asociaciones` y `Temporal`;
- graficos filtrables solicitados por la rubrica: histogramas, boxplot,
  matriz de correlacion, serie temporal, descomposicion y ACF/PACF;
- opciones adicionales disponibles aparte: ventas por canal, diagnosticos del
  modelo, hipotesis y visor de PNG;
- recalculo de los graficos al aplicar el filtro de fechas;
- selector de PNG generados en `plots/` y boton `Abrir PNG` para ver el grafico
  elegido dentro de la pestana `PNG seleccionado`.
- las pestanas informativas se cargan desde los JSON de `resultados/`.

Se implementa con `tkinter` y `matplotlib`, por lo que no requiere instalar
Streamlit, Dash ni otras dependencias adicionales.

## Documentacion interna del codigo

Los scripts estan documentados con docstrings en modulos, clases, funciones y
metodos. Cada funcion describe:

- para que existe dentro del pipeline;
- que entradas recibe;
- que salida produce o que archivo escribe;
- pasos principales de ejecucion;
- precauciones relevantes, como uso de locks, orden de escritura, muestras
  reproducibles o decisiones de formato.

Puntos importantes por archivo:

| Archivo | Documentacion clave |
| --- | --- |
| `validacion.py` | validacion de RUT, UUID, fechas, edad, genero, tipos numericos y motivos de descarte |
| `procesar_paralelo.py` | chunking, workers, `OrderedOutputWriter`, buffer ordenado, `flush_ready` y proteccion contra carreras |
| `preprocesamiento.py` | carga Parquet/CSV, reporte de faltantes, imputacion, variables derivadas, IQR, z-score y JSON de respaldo |
| `analisis_exploratorio.py` | descriptivas, normalidad, correlaciones, asociaciones, graficos, serie temporal y paralelismo tabular |
| `inferencia_modelado.py` | hipotesis, modelo no lineal, diagnostico OLS, VIF, residuos, metricas y paralelismo por tareas |
| `dashboard.py` | carga de resultados, filtros por fecha, actualizacion de graficos, cache del modelo filtrado y visor PNG |
| `run_pipeline.py` | orquestacion, rutas Parquet, semilla, workers por cantidad y apertura del dashboard |

Orden recomendado para revisar el flujo desde el codigo:

1. `run_pipeline.py`
2. `procesar_paralelo.py`
3. `validacion.py`
4. `preprocesamiento.py`
5. `analisis_exploratorio.py`
6. `inferencia_modelado.py`
7. `dashboard.py`

## Proteccion de datos compartidos

La etapa paralela esta disenada para evitar condiciones de carrera.

Problema potencial:

- varios workers podrian intentar escribir al mismo tiempo en el dataset de salida;
- varios workers podrian intentar escribir al mismo tiempo en `logs.txt`;
- los chunks podrian terminar en orden distinto al orden original del archivo.

Medidas aplicadas:

1. **Sin escritura concurrente desde workers**

   Los workers solo reciben datos y devuelven resultados en memoria. No abren
   el dataset validado ni `logs.txt`.

2. **Escritura centralizada**

   Solo el proceso principal escribe archivos. Esto evita corrupcion de salida,
   intercalado de lineas y problemas de sincronizacion.

3. **Buffer por `chunk_index`**

   Como los chunks se procesan en paralelo, pueden terminar desordenados. Para
   conservar una salida determinista, cada resultado se guarda temporalmente en
   un buffer:

   ```python
   buffer[result["chunk_index"]] = result
   ```

   Luego se escribe solo cuando corresponde el siguiente chunk esperado.

4. **Orden determinista de salida**

   La funcion `flush_ready` escribe chunks en orden creciente de `chunk_index`.
   Esto permite que dos ejecuciones con los mismos datos y parametros produzcan
   el mismo orden de filas validas.

5. **Limite de tareas pendientes**

   `procesar_paralelo.py` usa un limite de chunks pendientes proporcional a
   `workers`. Esto evita leer demasiado rapido y acumular muchos bloques en
   memoria mientras los workers aun estan procesando.

En resumen, el paralelismo se usa para acelerar la validacion, pero las salidas
compartidas se escriben de manera controlada por un unico proceso.

## Instalacion en Windows PowerShell

Ejecutar desde PowerShell:

1. Entrar a la carpeta del proyecto:

```powershell
cd "C:\Users\amaru\Desktop\analisis paralelo\trabajo analisis"
```

2. Verificar Python:

```powershell
python --version
```

3. Crear entorno virtual:

```powershell
python -m venv .venv
```

4. Activar entorno:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activacion:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

5. Actualizar `pip` e instalar dependencias:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

6. Definir semilla reproducible:

```powershell
$env:CPYD_SEED = "42"
```

## Ejecucion

### Prueba rapida

Procesa una muestra del CSV completo. Sirve para verificar dependencias y flujo
sin esperar la ejecucion total.

```powershell
python scripts\run_pipeline.py --input data\ventas_completas.csv --sample 100000 --chunksize 100000 --workers 4 --prefix prueba
```

### Ejecucion completa

```powershell
python scripts\run_pipeline.py --input data\ventas_completas.csv --chunksize 100000 --workers 4 --executor process
```

Si `process` da problemas en Windows:

```powershell
python scripts\run_pipeline.py --input data\ventas_completas.csv --chunksize 100000 --workers 4 --executor thread
```

Si la terminal parece quedar pausada y continua al presionar Enter, normalmente
no es un bloqueo del codigo: suele ser QuickEdit/seleccion de texto de
PowerShell o ConHost. El pipeline y el preprocesamiento intentan desactivar ese
modo automaticamente al iniciar. Si vuelve a ocurrir, evita seleccionar texto en
la consola durante la ejecucion o usa Windows Terminal.

El dashboard se abre automaticamente al terminar. Para desactivarlo:

```powershell
python scripts\run_pipeline.py --input data\ventas_completas.csv --chunksize 100000 --workers 4 --executor process --no-dashboard
```

## Comandos individuales

Estos comandos permiten ejecutar cada etapa por separado:

```powershell
python scripts\explorar_csv.py

python scripts\procesar_paralelo.py --input data\ventas_completas.csv --output data\ventas_validas.parquet --chunksize 100000 --workers 4 --executor process

python scripts\preprocesamiento.py --input data\ventas_validas.parquet --output data\ventas_limpias.parquet

python scripts\analisis_exploratorio.py --input data\ventas_limpias.parquet --workers 4

python scripts\inferencia_modelado.py --input data\ventas_limpias.parquet --workers 4

python scripts\dashboard.py --input data\ventas_limpias.parquet --logs logs.txt --plots-dir plots --results-dir resultados
```

## Reproducibilidad

La variable `CPYD_SEED` controla las operaciones con aleatoriedad:

- muestras para graficos;
- particion train/test;
- cualquier proceso futuro que use aleatoriedad.

Si no se define, el codigo usa `42` por defecto.

Ademas de la semilla, el pipeline mantiene determinismo en la validacion
paralela porque:

- las reglas de validacion no usan aleatoriedad;
- cada chunk conserva su indice original;
- la escritura se realiza en orden de chunk;
- los archivos de salida se regeneran desde cero en cada ejecucion.

## Salidas esperadas

| Ruta | Contenido |
| --- | --- |
| `data/ventas_validas.parquet` | transacciones que pasan validacion; temporal por defecto |
| `data/ventas_limpias.parquet` | dataset limpio y enriquecido |
| `logs.txt` | filas descartadas y motivo |
| `resultados/resumen_preprocesamiento.json` | filas finales, faltantes y outliers |
| `resultados/reporte_faltantes.json` | faltantes por columna, tratamiento aplicado y pruebas simples de patron de ausencia |
| `resultados/parametros_normalizacion.json` | medias, desviaciones y columnas z-score generadas |
| `resultados/estadisticas_descriptivas.json` | media, mediana, dispersion, asimetria y curtosis |
| `resultados/normalidad.json` | pruebas Shapiro y Kolmogorov-Smirnov |
| `resultados/correlaciones.json` | matriz Spearman |
| `resultados/correlaciones_pvalues.json` | p-values de correlaciones Spearman |
| `resultados/pruebas_exploratorias.json` | Chi-cuadrado, Spearman y ANOVA/Kruskal |
| `resultados/analisis_temporal.json` | estado de descomposicion temporal y ACF/PACF |
| `resultados/pruebas_hipotesis.json` | hipotesis, p-values y decisiones |
| `resultados/metricas_modelo.json` | MAE, RMSE, R2 y predictores del modelo |
| `resultados/diagnostico_regresion_lineal.json` | coeficientes, R2 ajustado, VIF, diagnosticos de residuos y limitaciones |
| `plots/*.png` | visualizaciones del analisis y del modelo |

El archivo `data/ventas_validas*.parquet` es intermedio y el pipeline lo elimina
por defecto despues del preprocesamiento. El archivo
`data/ventas_limpias*.parquet` es el dataset analitico persistente para EDA,
inferencia y dashboard. Se regenera desde `ventas_completas.csv` y no debe
mantenerse en control de versiones.

El cambio desde CSV intermedio a Parquet se hizo porque el guardado y la lectura
de `ventas_limpias.csv` eran el principal cuello de botella. Parquet mantiene
tipos de datos, comprime mejor y acelera las lecturas posteriores, por lo que el
pipeline conserva trazabilidad sin cargar con archivos intermedios de mas de
1 GB.

## Resultados obtenidos con el CSV completo

Ultima ejecucion completa:

- tiempo total observado sin apertura de dashboard: `142.6` segundos;
- tiempo total esperado con apertura automatica del dashboard: aproximadamente
  `3` minutos;
- filas leidas: `3.242.878`;
- filas validas: `3.239.993`;
- filas descartadas en validacion: `2.885`;
- descartadas por edad fuera de rango en validacion: `2.876`;
- filas limpias: `3.239.993`;
- dataset limpio generado: `data/ventas_limpias.parquet`, aproximadamente
  `275 MB`;
- outliers de monto: `212.351`;
- outliers de unidades: `0`;
- dias observados en serie temporal: `241`;
- modelo: `HistGradientBoostingRegressor` sobre `log1p(MONTO APLICADO)`;
- modelo lineal interpretable: OLS sobre `log1p(MONTO APLICADO)`;
- muestra de entrenamiento/modelo: `500.000` filas reproducibles;
- MAE del modelo: `1683.57`;
- RMSE del modelo: `3651.64`;
- R2 del modelo: `0.9368`.

El modelo mejorado captura mejor la variabilidad del monto al incorporar `SKU`,
`LOCAL`, variables temporales, canal, descuento, edad y frecuencia de compra.

## Decisiones de diseno

- Se usa chunking para controlar memoria.
- Se usa paralelismo en validacion porque cada fila puede evaluarse de forma
  independiente.
- Se escribe el resultado en orden de chunk para mantener determinismo.
- Los workers no escriben directamente en archivos compartidos.
- El proceso principal concentra la escritura del dataset validado y logs.
- Se separan validacion, preprocesamiento, EDA y modelo para facilitar pruebas.
- Se evita mantener CSV intermedios pesados: el dataset validado temporal y el
  dataset limpio usan Parquet.
- Se marcan outliers en vez de eliminarlos automaticamente.
- Se genera un reporte JSON de faltantes antes de imputar, incluyendo evidencia
  estadistica simple sobre patrones de ausencia.
- Se guardan los parametros usados para normalizacion z-score.
- Se usa regresion no lineal porque mejora el ajuste en una variable objetivo
  altamente asimetrica.

## Limitaciones y manejo

1. **Edades no plausibles**

   Se detectaron edades superiores a 110 anos. Para evitar sesgos en estadisticas
   y modelo, la validacion paralela descarta esos registros antes de generar el
   dataset validado. Cada fila descartada queda registrada en `logs.txt` con
   motivo `EDAD:edad_fuera_de_rango`.

2. **`UNIDADES` sin variacion**

   En el dataset limpio `UNIDADES` queda constante en 1. Por eso no es posible
   evaluar de forma directa una hipotesis sobre descuento y unidades vendidas. Se
   reemplaza por una hipotesis evaluable: asociacion entre descuento y
   `MONTO APLICADO`.

3. **P-values muy pequenos por gran tamano muestral**

   Con mas de tres millones de filas, diferencias pequenas pueden resultar
   estadisticamente significativas. Por eso los resultados deben interpretarse
   junto con magnitud del efecto, graficos y contexto de negocio.

4. **Modelo entrenado sobre muestra reproducible**

   Para controlar tiempo y memoria, el modelo usa una muestra reproducible de
   hasta 500.000 filas del dataset limpio. La muestra se fija con `CPYD_SEED`.
   Esto mantiene el entrenamiento manejable sin perder representatividad general.

5. **Outliers de monto**

   Los outliers de monto no se eliminan automaticamente. Se marcan con
   `OUTLIER_MONTO`, porque ventas extremas pueden ser transacciones reales y
   relevantes para el negocio.

## Por que no se paraleliza todo

Paralelizar todas las fases no siempre mejora el resultado. En este proyecto se
paraleliza la validacion porque es una tarea naturalmente independiente: cada
fila puede evaluarse sin conocer las demas. Tambien se paraleliza parcialmente
el analisis exploratorio para ejecutar en paralelo calculos tabulares
independientes.

En cambio, el preprocesamiento calcula elementos globales:

- frecuencia de compra por cliente;
- cuantiles para outliers IQR;
- medias y desviaciones para z-score.

Estas operaciones requieren mirar el conjunto completo ya validado. Se podrian
paralelizar con herramientas como Dask o Spark, pero para este proyecto se
priorizo una solucion mas simple, reproducible y facil de explicar.

En el analisis exploratorio, los graficos se mantienen secuenciales porque
Matplotlib no es seguro para multiples hilos de dibujo. En inferencia/modelado,
los calculos se ejecutan como tareas paralelas y los graficos se serializan con
un bloqueo interno. Asi se mejora el tiempo de ejecucion sin arriesgar archivos
PNG corruptos ni escrituras simultaneas sobre el mismo resultado.
