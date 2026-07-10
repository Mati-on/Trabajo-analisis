# Informe tecnico

## Analisis estadistico de ventas e inferencia de modelo para Cruz Morada

Curso: Computacion Paralela y Distribuida  
Fecha de ejecucion de resultados: 2026-07-08  
Dataset principal: `data/ventas_completas.csv`

## 1. Introduccion

Cruz Morada dispone de un archivo CSV consolidado con transacciones de venta y
datos asociados a clientes. El volumen del archivo exige una estrategia que
controle el uso de memoria y permita procesar los registros de forma eficiente.

La solucion implementada consiste en un pipeline reproducible que valida,
limpia, transforma, analiza y modela los datos. El paralelismo se aplica con
distintas estrategias segun la naturaleza de cada fase: validacion por chunks en
procesos o hilos, calculos tabulares del analisis exploratorio en hilos, e
inferencia/modelado por tareas independientes. Las fases que requieren calculos
globales o escritura controlada se mantienen centralizadas para conservar
determinismo y evitar condiciones de carrera.

## 2. Datos de entrada

El archivo de entrada principal es:

```text
data/ventas_completas.csv
```

El CSV usa separador `;` y contiene las columnas del enunciado:

- `FECHA`
- `CANAL`
- `SKU`
- `PRODUCTO`
- `UNIDADES`
- `PORCENTAJE DESCUENTO`
- `MONTO APLICADO`
- `BOLETA`
- `LOCAL`
- `CODIGO CLIENTE`
- `RUN CLIENTE`
- `NOMBRES`
- `APELLIDOS`
- `FECHA NACIMIENTO`
- `GENERO`

Tambien se conserva `data/transacciones_prueba.csv` para pruebas pequenas del
flujo.

## 3. Arquitectura del pipeline

El flujo completo se ejecuta con:

```powershell
python scripts\run_pipeline.py --input data\ventas_completas.csv --chunksize 100000 --workers 4 --executor process
```

Etapas:

1. `procesar_paralelo.py`: validacion paralela por chunks.
2. `preprocesamiento.py`: limpieza, variables derivadas, outliers y z-score.
3. `analisis_exploratorio.py`: estadisticas y pruebas exploratorias en paralelo parcial; graficos en modo secuencial.
4. `inferencia_modelado.py`: hipotesis y modelo predictivo en paralelo parcial
   por tareas independientes.

El pipeline genera salidas regenerables para trazabilidad:

- `data/ventas_validas.parquet` temporal, eliminado por defecto tras el preprocesamiento
- `data/ventas_limpias.parquet`
- `logs.txt`
- `resultados/*.json`
- `plots/*.png`

Se reemplazaron los CSV intermedios por Parquet porque la escritura y lectura de
`ventas_limpias.csv` era el principal cuello de botella del flujo. Parquet
reduce el peso en disco, conserva tipos de datos y acelera la lectura de las
etapas posteriores.

## 4. Paralelismo y memoria

La paralelizacion principal procesa el archivo por bloques de `100000` filas.
Cada chunk se envia a un worker, que valida sus registros y devuelve:

- filas validas;
- motivos de descarte;
- conteos de filas leidas, validas y descartadas.

Los workers no escriben directamente en archivos compartidos. Solo el proceso
principal escribe el dataset validado en Parquet y `logs.txt`. Esto evita
condiciones de carrera, corrupcion de archivo e intercalado de lineas.

Como los chunks pueden terminar en distinto orden, el proceso principal usa un
buffer por `chunk_index` y escribe solo cuando corresponde el siguiente bloque.
Con esto, la salida se mantiene determinista.

Ademas, el analisis exploratorio paraleliza calculos tabulares independientes
con la misma cantidad de workers configurada en el pipeline. La inferencia y el
modelado tambien usan esa cantidad como limite y ejecutan en paralelo las
pruebas de hipotesis, el modelo predictivo no lineal y el diagnostico lineal.
Los graficos se generan de forma controlada para evitar conflictos de
Matplotlib.

La ejecucion completa leyo `3.242.878` filas en 33 chunks. La validacion paralela
demoro `65.18` segundos. El flujo completo sin apertura del dashboard demoro
`142.6` segundos; con apertura automatica del dashboard, el tiempo observado se
mantiene alrededor de `3` minutos en el equipo de prueba. El dataset limpio
final quedo en `data/ventas_limpias.parquet`, con un tamano aproximado de
`275 MB`.

## 5. Reproducibilidad

Las operaciones con aleatoriedad usan la variable de entorno:

```text
CPYD_SEED
```

Si no existe, el valor por defecto es `42`. Esta semilla se usa en:

- muestreos para graficos;
- particion train/test;
- muestra reproducible del modelo.

## 6. Validacion de registros

La validacion revisa:

- cantidad correcta de columnas;
- campos obligatorios no vacios;
- tipos numericos;
- valores positivos para unidades, boleta, local y monto;
- porcentaje de descuento entre `0` y `1`;
- RUT chileno con digito verificador;
- UUID valido para `CODIGO CLIENTE`;
- formato de fechas;
- edad plausible entre `0` y `110` anos;
- genero con valores `1` o `2`.

Resultados de validacion:

| Metrica | Valor |
| --- | ---: |
| Filas leidas | 3.242.878 |
| Filas validas | 3.239.993 |
| Filas descartadas | 2.885 |
| Descartadas por edad fuera de rango | 2.876 |
| Descartadas por otras reglas | 9 |

Cada fila descartada queda registrada en `logs.txt` con numero de linea y motivo.
Ejemplo:

```text
2327:EDAD:edad_fuera_de_rango:-1.80
19871:UNIDADES o BOLETA o LOCAL:valor_numerico_no_positivo:1/0/9069
```

## 7. Preprocesamiento

El dataset limpio final contiene `3.239.993` filas.

Transformaciones aplicadas:

- conversion de fechas;
- conversion de numericos;
- reporte de valores faltantes antes de imputacion;
- calculo de `MONTO POR UNIDAD`;
- calculo de `EDAD`;
- calculo de `FRECUENCIA COMPRA`;
- deteccion de outliers por IQR;
- normalizacion z-score con parametros guardados.

Resumen:

| Metrica | Valor |
| --- | ---: |
| Filas limpias | 3.239.993 |
| Outliers de monto | 212.351 |
| Outliers de unidades | 0 |
| Descartadas por edad en preprocesamiento | 0 |

El descarte de edad ocurre en validacion; el filtro en preprocesamiento se
mantiene solo como respaldo defensivo.

Adicionalmente, el pipeline genera `resultados/reporte_faltantes.json` con el
conteo de ausencias por columna, tratamiento aplicado y pruebas simples de
asociacion entre ausencia y variables observadas. Tambien genera
`resultados/parametros_normalizacion.json` con media y desviacion usadas para
cada columna estandarizada.

## 8. Analisis exploratorio

### Estadisticas descriptivas

Resultados principales:

| Variable | Media | Mediana | Desv. estandar | Minimo | Maximo |
| --- | ---: | ---: | ---: | ---: | ---: |
| `UNIDADES` | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| `PORCENTAJE DESCUENTO` | 0.392 | 0.400 | 0.108 | 0.00 | 1.00 |
| `MONTO APLICADO` | 10.180,54 | 7.662,00 | 14.451,43 | 15,00 | 226.476,00 |
| `EDAD` | 49,57 | 48,61 | 16,73 | 0,01 | 109,85 |
| `FRECUENCIA COMPRA` | 5,66 | 4,00 | 5,46 | 1,00 | 110,00 |

Interpretacion:

- `UNIDADES` no presenta variacion: todas las transacciones limpias tienen una
  unidad registrada.
- `MONTO APLICADO` presenta fuerte asimetria positiva y alta curtosis, lo que
  confirma presencia de ventas extremas.
- La edad queda en un rango plausible despues de validar y descartar registros
  imposibles.

### Normalidad

Las pruebas Shapiro-Wilk y Kolmogorov-Smirnov se aplican sobre las variables
numericas del analisis. Para variables sin variacion, como `UNIDADES`, la prueba
queda marcada como no evaluable. En monto, descuento y edad se rechaza
normalidad, lo que justifica complementar pruebas parametricas con pruebas no
parametricas, como Spearman, Kruskal-Wallis y Mann-Whitney.

### Correlaciones

Se uso correlacion de Spearman por la no normalidad y presencia de outliers.
Resultados relevantes:

- Descuento vs monto: `rho = 0.4827`.
- Edad vs monto: `rho = 0.0992`.
- Frecuencia de compra vs monto: `rho = 0.2245`.

Tambien se genero `correlaciones_pvalues.json` con significancia asociada.
`UNIDADES` no presenta variacion en los datos limpios, por lo que sus
correlaciones directas no son interpretables y se reportan como no evaluables
cuando corresponde.

### Asociaciones y comparaciones

Se ejecutaron:

- Chi-cuadrado entre `CANAL` y `LOCAL`;
- Chi-cuadrado complementario entre `CANAL` y `GENERO`;
- Spearman entre `UNIDADES`, `MONTO APLICADO` y `PORCENTAJE DESCUENTO`;
- ANOVA para `MONTO APLICADO` por `CANAL`;
- ANOVA para `MONTO APLICADO` por `LOCAL`;
- Kruskal-Wallis para `MONTO APLICADO` por `CANAL`.
- Kruskal-Wallis para `MONTO APLICADO` por `LOCAL`.

Las comparaciones por canal y local permiten evaluar diferencias estadisticas en
los montos segun canal de compra y sucursal. Para `LOCAL`, cuando existen muchos
grupos, el analisis se acota a los 200 locales con mas registros para controlar
costo computacional sin perder los grupos mas representativos.

### Patrones temporales

La serie temporal diaria contiene `241` dias observados. Se generaron:

- `serie_temporal_ventas.png`
- `descomposicion_temporal.png`
- `acf_pacf_temporal.png`

Esto cumple el requerimiento de analizar tendencia, estacionalidad y
autocorrelacion.

## 9. Hipotesis estadisticas

### Hipotesis del enunciado: APP vs WEB

- H0: ticket promedio APP <= ticket promedio WEB.
- H1: ticket promedio APP > ticket promedio WEB.
- Welch t-test unilateral: no se rechaza H0.
- Mann-Whitney unilateral: no se rechaza H0.

Decision: no se encontro evidencia suficiente para afirmar que APP tenga ticket
promedio mayor que WEB en los datos evaluados.

### Hipotesis del enunciado: descuento y unidades

- H0: el coeficiente de descuento sobre `UNIDADES` es igual a 0.
- H1: el coeficiente de descuento sobre `UNIDADES` es distinto de 0.

Decision: no evaluable. `UNIDADES` no presenta variacion en los datos limpios
(`UNIDADES = 1`), por lo que no es posible ajustar una regresion lineal
informativa ni ANCOVA sobre unidades. Se documenta esta limitacion y se usa
descuento vs monto aplicado como hipotesis evaluable alternativa.

### H1: monto promedio segun canal

- H0: `MONTO APLICADO` no difiere entre canales.
- H1: al menos un canal presenta diferencias.
- ANOVA: `F = 150.363162`, `p = 1.923011e-97`.
- Kruskal-Wallis: `H = 5937.899329`, `p = 0`.

Decision: se rechaza H0. Existen diferencias estadisticamente significativas en
el monto segun canal.

### H2: descuento y monto comprado

- H0: no existe asociacion monotonica entre descuento y monto.
- H1: existe asociacion.
- Spearman: `rho = 0.482738`, `p = 0`.

Decision: se rechaza H0. Existe asociacion positiva moderada entre descuento y
monto aplicado.

Se usa monto y no unidades porque `UNIDADES` no tiene variacion en los datos
limpios.

### H3: edad y monto comprado

- H0: no existe asociacion monotonica entre edad y monto.
- H1: existe asociacion.
- Spearman: `rho = 0.099179`, `p = 0`.

Decision: se rechaza H0. La asociacion existe pero es debil; por el gran tamano
muestral, significancia estadistica no implica necesariamente relevancia
practica alta.

### H4: monto promedio segun genero

- Welch t-test: `t = 45.019240`, `p = 0`.
- Mann-Whitney: `U = 1207448024126`, `p = 0`.

Decision: se rechaza H0. Hay diferencias estadisticas en monto segun genero,
pero deben interpretarse con cautela por el volumen de datos y posibles factores
de confusion.

## 10. Modelado predictivo

Se entreno un modelo `HistGradientBoostingRegressor` para predecir
`MONTO APLICADO`. La variable objetivo se transformo con `log1p` para reducir el
efecto de la cola larga de montos.

Predictores usados:

- `UNIDADES`
- `PORCENTAJE DESCUENTO`
- `EDAD`
- `FRECUENCIA COMPRA`
- `SKU`
- `GENERO`
- `LOCAL`
- `FECHA_MES`
- `FECHA_DIA_SEMANA`
- `FECHA_HORA`
- dummies de `CANAL`

Para controlar tiempo y memoria, el modelo uso una muestra reproducible de
`500.000` filas del dataset limpio.

Metricas:

| Metrica | Valor |
| --- | ---: |
| Train | 350.000 |
| Test | 150.000 |
| MAE | 1.683,57 |
| RMSE | 3.651,64 |
| R2 | 0,9368 |

Interpretacion:

El modelo explica una alta proporcion de la variabilidad del monto. La mejora se
debe a que el modelo no lineal captura diferencias por producto, local, canal y
patrones temporales. Aun asi, debe considerarse como modelo predictivo base y no
como causal: las variables explican patrones observados, no necesariamente
efectos causales.

### Regresion lineal interpretable

Para cubrir la opcion de regresion interpretativa del enunciado, se agrego una
regresion OLS complementaria sobre `log1p(MONTO APLICADO)` usando `CANAL`,
`LOCAL`, `UNIDADES` y `PORCENTAJE DESCUENTO`. El diagnostico se guarda en
`resultados/diagnostico_regresion_lineal.json`.

El reporte incluye:

- coeficientes e intervalos de confianza;
- R2 y R2 ajustado;
- VIF para multicolinealidad;
- normalidad de residuos con Jarque-Bera y Shapiro-Wilk;
- homocedasticidad con Breusch-Pagan;
- metricas MAE, RMSE y R2 en test;
- limitaciones de extrapolabilidad.

`UNIDADES` se excluye automaticamente del modelo lineal cuando no presenta
variacion. El diagnostico muestra que el modelo lineal es interpretable, pero
menos predictivo que el modelo no lineal, por lo que se conserva
`HistGradientBoostingRegressor` como modelo predictivo principal.

## 11. Limitaciones y manejo

- `UNIDADES` no tiene variacion, por lo que no permite evaluar directamente la
  hipotesis descuento vs unidades. Se reemplazo por descuento vs monto.
- Los p-values tienden a ser extremadamente pequenos debido al gran tamano
  muestral. Se interpretan junto a magnitud de asociacion y contexto.
- Los outliers de monto se marcan, no se eliminan automaticamente, porque pueden
  representar ventas reales.
- El modelo usa una muestra reproducible de 500.000 filas para controlar tiempo
  y memoria.
- Las pruebas estadisticas detectan asociacion, pero no prueban causalidad.

## 12. Cumplimiento del enunciado

| Requisito | Estado |
| --- | --- |
| Carga desde linea de comandos | Cumplido |
| Uso de `data/ventas_completas.csv` | Cumplido |
| Procesamiento por chunks | Cumplido |
| Paralelismo | Cumplido |
| Semilla `CPYD_SEED` | Cumplido |
| Limpieza y faltantes | Cumplido |
| Outliers robustos | Cumplido |
| Variables derivadas | Cumplido |
| Estadisticas descriptivas | Cumplido |
| Histogramas con densidad | Cumplido |
| Normalidad | Cumplido |
| Boxplot por categoria | Cumplido |
| Correlacion con significancia | Cumplido |
| Chi-cuadrado | Cumplido |
| Pearson/Spearman | Cumplido con Spearman |
| ANOVA | Cumplido |
| Series temporales, descomposicion, ACF/PACF | Cumplido |
| Al menos 3 hipotesis | Cumplido |
| Modelo predictivo | Cumplido |
| Train/test y metricas | Cumplido |
| Discusion de limitaciones | Cumplido |

## 13. Conclusiones

El proyecto implementa una solucion eficiente, reproducible y auditable para el
analisis de ventas de Cruz Morada. La validacion paralela permite procesar el
archivo grande por chunks, manteniendo control de memoria y registros de
descartes. El analisis exploratorio muestra montos fuertemente asimetricos,
diferencias por canal y patrones temporales. Las hipotesis planteadas muestran
asociaciones estadisticamente significativas, aunque su interpretacion debe
considerar el gran tamano muestral.

El modelo predictivo no lineal obtiene buen rendimiento para `MONTO APLICADO` y
cumple el requerimiento de modelado. Como trabajo futuro, se podria explorar
validacion temporal, comparacion con otros modelos y analisis mas profundo por
familia de productos o segmentos de clientes.
