# Enunciado del Trabajo Práctico

## Universidad Tecnológica Metropolitana
Departamento de Computación e Informática
Curso: Computación Paralela y Distribuida
Profesor: Sebastián Salazar Molina

### Título
Análisis Estadístico de Ventas e Inferencia de Modelo para Cruz Morada

### Fecha de entrega
10 de julio de 2026

## 1. Contexto

Cruz Morada, una de las cadenas de farmacias líderes en el mercado chileno, enfrenta desafíos importantes para analizar sus datos de ventas. Los registros de transacciones y la información asociada a los clientes solían estar fragmentados en múltiples archivos CSV, dificultando un análisis integral y la extracción de insights estratégicos.

Para este trabajo práctico se dispone de un archivo consolidado que contiene todas las transacciones y datos de clientes. Esta consolidación permite concentrarse en análisis avanzado, limpieza de datos y el diseño de un flujo de procesamiento escalable y reproducible.

El volumen de datos es considerable, por lo que se espera aplicar metodologías de procesamiento paralelo o por fragmentos para aprovechar mejor los recursos y reducir tiempos de cómputo.

## 2. Objetivo general

Diseñar e implementar una solución computacional que permita a Cruz Morada procesar y analizar sus datos de ventas de forma eficiente, reproducible y escalable.

La solución debe:

- procesar archivos grandes de forma eficiente,
- aplicar un algoritmo paralelo o basado en procesamiento por bloques,
- garantizar eficiencia, escalabilidad y bajo consumo de recursos,
- documentar el proceso y entregar resultados interpretables.

## 3. Objetivos específicos

Al finalizar el trabajo, se espera que el equipo sea capaz de:

1. aplicar técnicas estadísticas para caracterizar el comportamiento de las ventas,
   - describir tendencias,
   - modelar y validar hipótesis sobre patrones de consumo,
   - identificar relaciones significativas y sesgos en los datos.
2. diseñar e implementar soluciones de procesamiento paralelo o por fragmentos para grandes volúmenes de datos.
3. documentar y presentar los resultados de forma clara y profesional.

## 4. Lenguaje y herramientas recomendadas

- Lenguaje de programación: Python, Java, Scala, R, Rust u otro que soporte paralelismo nativo.
- Bibliotecas recomendadas:
  - Python: Pandas, Dask, NumPy, PySpark,
  - Java/Scala: Apache Spark, Akka,
  - Otros: cualquier framework que permita procesamiento paralelo o distribuido.

## 5. Especificaciones técnicas

### 5.1 Datos de entrada

- Se dispone de un archivo CSV consolidado con los datos de ventas.
- El archivo debe cargarse desde la línea de comandos al ejecutar el programa.
- El proyecto debe documentar cómo se maneja la memoria y el tamaño del archivo.

### 5.2 Estructura del archivo CSV

Cada registro contiene las siguientes columnas (separadas por `;`):

- `FECHA`: String. Fecha de operación en formato ISO 8601, por ejemplo `2026-05-08T00:02:53`.
- `CANAL`: String. Canal de compra, por ejemplo `POS`, `online`.
- `SKU`: Integer. Identificador único del producto.
- `PRODUCTO`: String. Nombre del producto.
- `UNIDADES`: Integer. Cantidad de unidades compradas.
- `PORCENTAJE DESCUENTO`: Float. Descuento aplicado, en rango `0.0` a `1.0`.
- `MONTO APLICADO`: Float. Monto total pagado en pesos chilenos.
- `BOLETA`: Integer. Número de boleta de la transacción.
- `LOCAL`: Integer. Identificador del local donde se realizó la compra.
- `CODIGO CLIENTE`: String (UUID). Identificador único del cliente en formato UUID.
- `RUN CLIENTE`: String. RUT del cliente en formato chileno.
- `NOMBRES`: String. Nombre del cliente.
- `APELLIDOS`: String. Apellidos del cliente.
- `FECHA NACIMIENTO`: String. Fecha de nacimiento del cliente en formato `AAAA-MM-DD`.
- `GENERO`: Integer. Género del cliente: `1` para masculino, `2` para femenino.

## 6. Requisitos del desarrollo

### 6.1 Carga y exploración de datos

- Implementar un módulo que cargue el archivo desde `data/ventas_completas.csv`.
- Dado el gran tamaño del archivo, debe usarse lectura por fragmentos (`chunking`) o herramientas que eviten cargar todo en memoria.
- Documentar el manejo de memoria y la estrategia de lectura.

### 6.2 Procesamiento paralelo o por bloques

- Implementar un algoritmo que procese los datos en paralelo o por particiones lógicas.
- Se puede dividir el procesamiento por `LOCAL`, por fecha, por bloques de filas o utilizando un framework paralelo.
- Optimizar el uso de CPU y memoria.

### 6.3 Reproducibilidad y determinismo

- Todas las operaciones que incluyan aleatoriedad deben fijar una semilla explícita.
- La semilla debe leerse desde la variable de entorno `CPYD_SEED`.
- Si se respeta este requisito, el resultado debe ser determinista.

## 7. Componentes del trabajo evaluado

### 7.1 Datos y preprocesamiento (20%)

- Consolidación y limpieza: documentar la carga y el manejo de memoria.
- Tratamiento de valores faltantes: identificar y justificar la estrategia de imputación o eliminación.
- Detección de outliers: emplear métodos robustos como IQR, Z-score u otros.
- Semilla fija: mantener determinismo si se usa aleatoriedad.
- Transformación de variables: crear derivadas relevantes.

Variables derivadas sugeridas:

- `MONTO POR UNIDAD` = `MONTO APLICADO` / `UNIDADES`.
- `EDAD` a partir de `FECHA NACIMIENTO` y la fecha de la transacción.
- `FRECUENCIA COMPRA` por `CODIGO CLIENTE`.
- Normalización o estandarización de variables numéricas.

### 7.2 Análisis exploratorio estadístico (30%)
a central, dispersión, asimetría y curtosis.
- Visualizac
- Describir distribuciones: tendenciiones obligatorias:
  - histogramas con curvas de densidad y pruebas de normalidad,
  - boxplots por categoría (por ejemplo `MONTO APLICADO` vs `CANAL`),
  - matriz de correlación con pruebas de significancia.
- Análisis de asociación:
  - pruebas Chi-cuadrado para variables categóricas,
  - correlación de Pearson o Spearman para variables numéricas,
  - ANOVA para comparar `MONTO APLICADO` entre `CANAL` o `LOCAL`.
- Patrones temporales:
  - descomposición de series de tiempo,
  - autocorrelación (ACF/PACF).

### 7.3 Inferencia estadística y modelado (30%)

- Pruebas de hipótesis:
  - Ejemplo 1: comparar ticket promedio entre `APP` y `WEB`.
  - Ejemplo 2: evaluar si el descuento afecta las unidades vendidas.
- Requerimiento: plantear al menos 3 hipótesis propias y validarlas.
- Modelado predictivo/descriptivo:
  - opción A: regresión para predecir `MONTO APLICADO`,
  - opción B: clustering de clientes o productos,
  - opción C: reglas de asociación entre productos.
- Validación de modelos:
  - dividir los datos en entrenamiento/prueba,
  - evaluar métricas como RMSE, MAE o silhouette score,
  - discutir extrapolabilidad y limitaciones.

## 8. Entregables

Los estudiantes deben entregar:

1. código fuente en un repositorio Git (GitHub/GitLab), con el profesor como colaborador (`sebasalazar`).
2. informe técnico en PDF que incluya:
   - descripción de la solución implementada,
   - justificación de las librerías utilizadas,
   - interpretación de resultados,
   - explicación de las técnicas aplicadas,
   - dificultades encontradas y cómo se resolvieron.

### 8.1 Fecha límite

10/07/2026 hasta las 23:59:59.999999 hora continental de Chile.

## 9. Rúbrica de evaluación

- Preprocesamiento y limpieza: 20%
- Análisis exploratorio: 30%
- Inferencia y modelado: 30%
- Rigor metodológico: 10%
- Claridad y estructura: 10%

> Se valorará el uso de múltiples pruebas estadísticas, la discusión de supuestos y la capacidad de interpretar resultados en contexto de negocio.
