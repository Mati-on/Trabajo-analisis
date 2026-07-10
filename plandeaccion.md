# Plan de accion y estado final

Fecha de actualizacion: 2026-07-08

Este documento resume el estado final del proyecto `trabajo analisis` y los
pasos restantes para preparar la entrega.

## 1. Objetivo del proyecto

Implementar una solucion reproducible para procesar y analizar
`data/ventas_completas.csv` aplicando procesamiento por chunks, paralelismo,
limpieza de datos, analisis estadistico, visualizaciones, pruebas de hipotesis y
modelado predictivo.

## 2. Estado actual

### Completado

- [x] Carga del CSV real desde linea de comandos.
- [x] Exploracion del volumen del archivo con `scripts/explorar_csv.py`.
- [x] Validacion de registros con `scripts/validacion.py`.
- [x] Validacion paralela por chunks con `scripts/procesar_paralelo.py`.
- [x] Registro de filas descartadas en `logs.txt`.
- [x] Descarte de RUT, UUID, fechas, genero, montos, descuentos y edades invalidas.
- [x] Conteo por chunk de filas descartadas por edad.
- [x] Preprocesamiento con `scripts/preprocesamiento.py`.
- [x] Variables derivadas: `MONTO POR UNIDAD`, `EDAD`, `FRECUENCIA COMPRA`.
- [x] Outliers por IQR y normalizacion z-score.
- [x] Analisis exploratorio con paralelismo parcial en `scripts/analisis_exploratorio.py`.
- [x] Histogramas con densidad, boxplot, matriz de correlacion y p-values.
- [x] Serie temporal, descomposicion temporal y ACF/PACF.
- [x] Pruebas Chi-cuadrado, Spearman, ANOVA y Kruskal-Wallis.
- [x] Pruebas de hipotesis en `scripts/inferencia_modelado.py`.
- [x] Modelo predictivo no lineal para `MONTO APLICADO`.
- [x] Metricas MAE, RMSE y R2.
- [x] Dashboard local para explorar resumenes y graficos por rango de fechas.
- [x] README tecnico actualizado.
- [x] Informe tecnico base actualizado.

## 3. Estructura vigente

```text
trabajo analisis/
|-- data/
|   |-- ventas_completas.csv
|   |-- ventas_limpias.parquet
|   `-- transacciones_prueba.csv
|-- plots/
|-- resultados/
|-- scripts/
|   |-- explorar_csv.py
|   |-- validacion.py
|   |-- procesar_paralelo.py
|   |-- preprocesamiento.py
|   |-- analisis_exploratorio.py
|   |-- inferencia_modelado.py
|   |-- dashboard.py
|   `-- run_pipeline.py
|-- README.md
|-- INFORME_TECNICO.md
|-- requirements.txt
|-- logs.txt
`-- plandeaccion.md
```

## 4. Resultados actuales

- Filas leidas: `3.242.878`.
- Filas validas: `3.239.993`.
- Filas descartadas en validacion: `2.885`.
- Descartadas por edad fuera de rango: `2.876`.
- Descartadas por otras reglas: `9`.
- Filas limpias: `3.239.993`.
- Outliers de monto: `212.351`.
- Outliers de unidades: `0`.
- Dias observados en serie temporal: `241`.
- Modelo: `HistGradientBoostingRegressor` sobre `log1p(MONTO APLICADO)`.
- MAE: `1683.57`.
- RMSE: `3651.64`.
- R2: `0.9368`.
- Tiempo total observado sin apertura de dashboard: `142.6` segundos.
- Tiempo total esperado con apertura automatica del dashboard: aproximadamente
  `3` minutos.

El pipeline ya no mantiene CSV intermedios. `ventas_validas*.parquet` se usa
como salida temporal de validacion y se elimina por defecto; el dataset limpio
persistente es `ventas_limpias*.parquet`. Para mantener liviana la carpeta
`data/`, solo deben quedar el CSV base, datasets Parquet necesarios y
`transacciones_prueba.csv`.

## 5. Pendientes reales de entrega

- [ ] Completar interpretacion narrativa en `INFORME_TECNICO.md`.
- [ ] Revisar visualmente graficos finales en `plots/`.
- [ ] Exportar `INFORME_TECNICO.md` a PDF.
- [ ] Subir codigo a repositorio GitHub/GitLab.
- [ ] Agregar al profesor `sebasalazar` como colaborador.
- [ ] Verificar que no se suba `.venv/` ni archivos pesados si el repositorio no
  los requiere.

## 6. Comando principal

```powershell
python scripts\run_pipeline.py --input data\ventas_completas.csv --chunksize 100000 --workers 4 --executor process
```

Si `process` falla en Windows:

```powershell
python scripts\run_pipeline.py --input data\ventas_completas.csv --chunksize 100000 --workers 4 --executor thread
```

## 7. Dashboard local

```powershell
python scripts\dashboard.py --input data\ventas_limpias.parquet --logs logs.txt
```

El pipeline completo abre este dashboard automaticamente al terminar. Para una
ejecucion solo por consola se usa `--no-dashboard`.
