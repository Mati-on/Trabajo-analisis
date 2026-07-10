"""Preprocesamiento de ventas validadas.

Convierte tipos, limpia datos esenciales, crea variables derivadas, marca
outliers por IQR, normaliza variables numericas y guarda un resumen auditable.
Por defecto lee y escribe Parquet para evitar CSV intermedios pesados.
"""

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ventas_validas.parquet"
OUTPUT_PATH = ROOT / "data" / "ventas_limpias.parquet"
SUMMARY_PATH = ROOT / "resultados" / "resumen_preprocesamiento.json"
MISSING_REPORT_PATH = ROOT / "resultados" / "reporte_faltantes.json"
NORMALIZATION_PATH = ROOT / "resultados" / "parametros_normalizacion.json"
MAX_EDAD_VALIDA = 110

# La semilla queda disponible para mantener el mismo criterio reproducible que
# el resto del pipeline. En este modulo no se muestrea aleatoriamente, pero se
# imprime para que las ejecuciones sean trazables.
SEED = int(os.environ.get("CPYD_SEED", "42"))


def disable_quick_edit_mode() -> None:
    """Evita que PowerShell/ConHost pause el script por seleccion accidental.

    Pasos:
    1. Verifica si el sistema es Windows.
    2. Obtiene el handle de entrada estandar.
    3. Lee el modo actual de consola.
    4. Desactiva QuickEdit y conserva flags extendidos.
    5. Ignora errores para mantener compatibilidad con otras terminales.

    Cuando QuickEdit esta activo, un click o seleccion en la terminal puede
    congelar la salida hasta presionar Enter. Esta proteccion solo aplica en
    Windows y se ignora silenciosamente si la consola no permite cambiar el modo.
    """
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        stdin_handle = kernel32.GetStdHandle(-10)
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(stdin_handle, ctypes.byref(mode)):
            quick_edit = 0x0040
            extended_flags = 0x0080
            new_mode = (mode.value | extended_flags) & ~quick_edit
            kernel32.SetConsoleMode(stdin_handle, new_mode)
    except Exception:
        return


def log_step(message: str, start: float | None = None) -> float:
    """Imprime avance con flush para que PowerShell muestre progreso real.

    Pasos:
    1. Si `start` es `None`, imprime el inicio de una etapa y retorna el tiempo
       actual.
    2. Si `start` trae un tiempo previo, imprime la duracion transcurrida.
    3. Siempre usa `flush=True` para que la consola muestre el progreso sin
       esperar a que termine el proceso.
    """
    now = time.perf_counter()
    if start is None:
        print(f"[preprocesamiento] {message}", flush=True)
    else:
        print(f"[preprocesamiento] {message} ({now - start:.2f}s)", flush=True)
    return now


def cargar_datos(path: Path) -> pd.DataFrame:
    """Carga el dataset validado desde CSV o Parquet.

    Pasos:
    1. Registra inicio con `log_step`.
    2. Si la extension es `.parquet`, usa `pd.read_parquet`.
    3. En otro caso, asume CSV separado por `;`.
    4. Registra tiempo total de carga y devuelve el `DataFrame`.

    Esta etapa ya recibe datos filtrados por la validacion inicial, por eso se
    puede usar pandas y trabajar con columnas completas para calcular derivadas,
    outliers y estadisticos.
    """
    start = log_step("cargando dataset validado")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    log_step("dataset validado cargado", start)
    return df


def guardar_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda el dataset limpio en Parquet o CSV segun la extension.

    Pasos:
    1. Crea la carpeta de salida si no existe.
    2. Si la extension es `.parquet`, usa `DataFrame.to_parquet`.
    3. En otro caso, guarda CSV con separador `;`.

    Parquet es el formato recomendado porque reduce peso en disco y acelera la
    lectura posterior de EDA, inferencia y dashboard. CSV queda disponible solo
    como alternativa de compatibilidad.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, sep=";")


def validar_columnas(df: pd.DataFrame) -> None:
    """Verifica que existan las columnas minimas para el preprocesamiento.

    Pasos:
    1. Define el conjunto de columnas requeridas por las transformaciones.
    2. Compara ese conjunto contra las columnas reales del `DataFrame`.
    3. Si falta alguna, lanza `ValueError` con la lista exacta.

    Si falta una columna critica, se detiene la ejecucion con un error claro. Es
    mejor fallar temprano que generar un dataset limpio incompleto o silencioso.
    """
    requeridas = {
        "FECHA",
        "CANAL",
        "SKU",
        "UNIDADES",
        "PORCENTAJE DESCUENTO",
        "MONTO APLICADO",
        "BOLETA",
        "LOCAL",
        "CODIGO CLIENTE",
        "RUN CLIENTE",
        "FECHA NACIMIENTO",
        "GENERO",
    }
    faltantes = sorted(requeridas - set(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")


def marcar_outlier_iqr(df: pd.DataFrame, columna: str, salida: str) -> None:
    """Crea una columna binaria de outlier usando el criterio IQR.

    Pasos:
    1. Calcula Q1 y Q3 de la columna.
    2. Calcula `IQR = Q3 - Q1`.
    3. Define limites `Q1 - 1.5 * IQR` y `Q3 + 1.5 * IQR`.
    4. Crea la columna `salida` con `1` si el registro cae fuera del rango y
       `0` en caso contrario.

    IQR es robusto frente a distribuciones sesgadas. No elimina registros:
    agrega una marca para que el informe pueda discutir ventas extremas sin
    perder informacion.
    """
    q1 = df[columna].quantile(0.25)
    q3 = df[columna].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[salida] = ((df[columna] < lower) | (df[columna] > upper)).astype(int)


def probar_patron_faltantes(df: pd.DataFrame, columna: str) -> list[dict]:
    """Evalua si los faltantes parecen asociarse con variables observadas.

    Pasos:
    1. Construye un indicador booleano que marca si `columna` esta ausente.
    2. Si no hay faltantes o todo falta, no ejecuta pruebas.
    3. Para variables numericas observadas, compara registros con y sin faltante
       mediante Mann-Whitney.
    4. Para variables categoricas observadas, usa Chi-cuadrado.
    5. Devuelve una lista de resultados serializable a JSON.

    No reemplaza una prueba MCAR completa de Little, pero deja evidencia
    cuantitativa simple: Mann-Whitney para variables numericas y Chi-cuadrado
    para variables categoricas. Si los p-values son altos, no se observa una
    asociacion fuerte entre ausencia y variables disponibles.
    """
    indicador = df[columna].isna()
    if indicador.sum() == 0 or indicador.sum() == len(df):
        return []

    pruebas = []
    numericas = ["UNIDADES", "PORCENTAJE DESCUENTO", "MONTO APLICADO", "LOCAL", "GENERO"]
    categoricas = ["CANAL"]

    for variable in [col for col in numericas if col in df.columns and col != columna]:
        con_faltante = df.loc[indicador, variable].dropna()
        sin_faltante = df.loc[~indicador, variable].dropna()
        if len(con_faltante) < 3 or len(sin_faltante) < 3:
            continue
        if con_faltante.nunique() < 2 and sin_faltante.nunique() < 2:
            continue
        try:
            stat, p_value = stats.mannwhitneyu(con_faltante, sin_faltante, alternative="two-sided")
            pruebas.append(
                {
                    "variable_observada": variable,
                    "prueba": "Mann-Whitney",
                    "estadistico": float(stat),
                    "p_value": float(p_value),
                }
            )
        except ValueError:
            continue

    for variable in [col for col in categoricas if col in df.columns and col != columna]:
        tabla = pd.crosstab(indicador, df[variable])
        if tabla.shape[0] < 2 or tabla.shape[1] < 2:
            continue
        chi2, p_value, dof, _ = stats.chi2_contingency(tabla)
        pruebas.append(
            {
                "variable_observada": variable,
                "prueba": "Chi-cuadrado",
                "estadistico": float(chi2),
                "grados_libertad": int(dof),
                "p_value": float(p_value),
            }
        )

    return pruebas


def construir_reporte_faltantes(df: pd.DataFrame) -> dict:
    """Construye reporte de faltantes, metodos de imputacion y evidencia MCAR.

    Pasos:
    1. Define tratamientos documentados para columnas con imputacion conocida.
    2. Recorre todas las columnas del `DataFrame`.
    3. Cuenta faltantes absolutos y porcentuales.
    4. Adjunta el tratamiento aplicado o marca que no corresponde imputacion.
    5. Si hay faltantes, llama `probar_patron_faltantes` para dejar evidencia
       estadistica simple sobre el patron de ausencia.
    6. Devuelve un diccionario listo para `reporte_faltantes.json`.
    """
    metodos = {
        "PORCENTAJE DESCUENTO": {
            "accion": "imputacion",
            "metodo": "mediana",
            "justificacion": "La mediana es robusta frente a descuentos extremos.",
        },
        "FECHA NACIMIENTO": {
            "accion": "imputacion",
            "metodo": "fecha fija 2000-01-01",
            "justificacion": "Respaldo trazable para permitir calcular EDAD cuando la fecha viene ausente o invalida.",
            "limitacion": "Puede sesgar la edad; se declara como imputacion simple y debe revisarse si aparecen faltantes relevantes.",
        },
    }

    columnas = []
    for col in df.columns:
        faltantes = int(df[col].isna().sum())
        porcentaje = float((faltantes / len(df)) * 100) if len(df) else 0.0
        item = {
            "columna": col,
            "faltantes": faltantes,
            "porcentaje": porcentaje,
            "tratamiento": metodos.get(col, {"accion": "sin imputacion", "metodo": "no aplica"}),
            "pruebas_patron_faltantes": probar_patron_faltantes(df, col) if faltantes else [],
        }
        columnas.append(item)

    total_faltantes = sum(item["faltantes"] for item in columnas)
    return {
        "filas_evaluadas": int(len(df)),
        "total_faltantes": int(total_faltantes),
        "nota_mcar": (
            "Se reportan pruebas simples de asociacion entre indicadores de ausencia y variables observadas. "
            "No corresponde a Little MCAR completo, pero entrega evidencia objetiva para justificar imputaciones."
        ),
        "columnas": columnas,
    }


def limpiar_y_transformar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza, derivadas, outliers y normalizacion.

    Pasos:
    1. Copia el `DataFrame` para no modificar el original.
    2. Limpia nombres de columnas y valida estructura minima.
    3. Convierte fechas y numericos con `errors="coerce"`.
    4. Elimina filas sin campos esenciales para analisis.
    5. Imputa descuento con mediana y fecha de nacimiento con una fecha fija
       declarada.
    6. Calcula `MONTO POR UNIDAD` y `EDAD`.
    7. Aplica respaldo defensivo para edad mayor a 110.
    8. Calcula `FRECUENCIA COMPRA` por `CODIGO CLIENTE`.
    9. Marca outliers IQR de monto y unidades.
    10. Calcula columnas z-score y guarda sus parametros en `df.attrs`.

    El resultado es el dataset `ventas_limpias.parquet`, usado por EDA,
    hipotesis y modelado. Todas las transformaciones quedan como columnas nuevas
    para que sean auditables.
    """
    # Trabajamos sobre una copia para no modificar el DataFrame original que se
    # haya pasado desde tests u otros scripts.
    start = log_step("copiando dataframe y validando columnas")
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    validar_columnas(df)
    log_step("columnas validadas", start)

    start = log_step("convirtiendo fechas")
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    df["FECHA NACIMIENTO"] = pd.to_datetime(df["FECHA NACIMIENTO"], errors="coerce")
    log_step("fechas convertidas", start)

    # Convertimos las columnas numericas con `errors="coerce"`: si un valor no
    # se puede convertir, queda como NaN y luego se decide como tratarlo.
    start = log_step("convirtiendo columnas numericas")
    numeric_cols = [
        "SKU",
        "UNIDADES",
        "PORCENTAJE DESCUENTO",
        "MONTO APLICADO",
        "BOLETA",
        "LOCAL",
        "GENERO",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    log_step("columnas numericas convertidas", start)

    # Estas columnas son indispensables para analisis y modelo. Si alguna viene
    # vacia despues de la conversion, la fila no es confiable para esta etapa.
    start = log_step("eliminando filas sin campos esenciales")
    esenciales = [
        "FECHA",
        "CANAL",
        "UNIDADES",
        "MONTO APLICADO",
        "LOCAL",
        "CODIGO CLIENTE",
        "RUN CLIENTE",
    ]
    df = df.dropna(subset=esenciales)
    log_step(f"filas esenciales filtradas: {len(df)}", start)

    # La imputacion de descuento con mediana evita sesgar por valores extremos.
    start = log_step("imputando faltantes simples")
    if df["PORCENTAJE DESCUENTO"].isna().any():
        df["PORCENTAJE DESCUENTO"] = df["PORCENTAJE DESCUENTO"].fillna(
            df["PORCENTAJE DESCUENTO"].median()
        )

    # Si falta fecha de nacimiento se usa una fecha fija y trazable. En el
    # informe se debe declarar esta decision como una imputacion simple.
    if df["FECHA NACIMIENTO"].isna().any():
        df["FECHA NACIMIENTO"] = df["FECHA NACIMIENTO"].fillna(pd.Timestamp("2000-01-01"))
    log_step("faltantes simples imputados", start)

    # Variables derivadas solicitadas por el enunciado.
    start = log_step("calculando monto por unidad y edad")
    df["MONTO POR UNIDAD"] = df["MONTO APLICADO"] / df["UNIDADES"].replace(0, np.nan)
    df["EDAD"] = ((df["FECHA"] - df["FECHA NACIMIENTO"]).dt.days / 365.25).astype(float)
    df["EDAD"] = df["EDAD"].clip(lower=0)
    log_step("monto por unidad y edad calculados", start)

    # Respaldo defensivo: la validacion paralela ya descarta edades fuera de
    # rango y las registra en logs.txt. Esta regla permanece para proteger el
    # preprocesamiento si se ejecuta con un CSV validado antiguo o externo.
    start = log_step("aplicando respaldo de rango de edad")
    filas_antes_edad = len(df)
    df = df[df["EDAD"] <= MAX_EDAD_VALIDA].copy()
    df.attrs["descartadas_edad_mayor_110"] = filas_antes_edad - len(df)
    log_step("respaldo de edad aplicado", start)

    start = log_step("calculando frecuencia de compra por cliente")
    df["FRECUENCIA COMPRA"] = df["CODIGO CLIENTE"].map(df["CODIGO CLIENTE"].value_counts())
    log_step("frecuencia de compra calculada", start)

    start = log_step("marcando outliers IQR")
    marcar_outlier_iqr(df, "MONTO APLICADO", "OUTLIER_MONTO")
    marcar_outlier_iqr(df, "UNIDADES", "OUTLIER_UNIDADES")
    log_step("outliers marcados", start)

    # Estandarizacion z-score para comparar variables en escalas distintas y
    # dejarlas disponibles para modelos posteriores.
    start = log_step("calculando columnas z-score")
    parametros_normalizacion = {}
    for col in [
        "MONTO APLICADO",
        "UNIDADES",
        "PORCENTAJE DESCUENTO",
        "MONTO POR UNIDAD",
        "EDAD",
        "FRECUENCIA COMPRA",
    ]:
        media = df[col].mean()
        desviacion = df[col].std(ddof=0)
        parametros_normalizacion[col] = {
            "media": float(media) if pd.notna(media) else None,
            "desviacion": float(desviacion) if pd.notna(desviacion) else None,
            "columna_generada": f"{col}_z",
            "formula": "(valor - media) / desviacion",
        }
        df[f"{col}_z"] = (df[col] - media) / desviacion if desviacion and desviacion != 0 else 0.0
    df.attrs["parametros_normalizacion"] = parametros_normalizacion
    log_step("columnas z-score calculadas", start)

    return df


def resumir_preprocesamiento(df: pd.DataFrame) -> dict:
    """Genera un resumen del dataset limpio.

    Pasos:
    1. Cuenta filas finales.
    2. Cuenta faltantes restantes solo en columnas donde existan.
    3. Suma marcas `OUTLIER_MONTO` y `OUTLIER_UNIDADES`.
    4. Devuelve un diccionario serializable a JSON.

    El resumen se usa como evidencia en el informe: total final, faltantes
    restantes y cantidad de outliers detectados.
    """
    start = log_step("generando resumen")
    resumen = {
        "filas_totales": int(len(df)),
        "valores_faltantes": {
            col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0
        },
        "outliers_monto": int(df["OUTLIER_MONTO"].sum()),
        "outliers_unidades": int(df["OUTLIER_UNIDADES"].sum()),
    }
    log_step("resumen generado", start)
    return resumen


def guardar_resumen(resumen: dict, output_path: Path) -> None:
    """Guarda el resumen de preprocesamiento en JSON para el informe.

    Pasos:
    1. Crea la carpeta destino.
    2. Convierte el diccionario a JSON indentado.
    3. Escribe el archivo en UTF-8.

    Crea la carpeta destino si falta y escribe JSON indentado para que pueda
    revisarse manualmente o consumirse desde el dashboard.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")


def guardar_json(payload: dict, output_path: Path) -> None:
    """Guarda un diccionario como JSON legible.

    Pasos:
    1. Crea la carpeta destino.
    2. Serializa el payload con indentacion.
    3. Conserva caracteres no ASCII con `ensure_ascii=False`.
    4. Escribe el archivo en UTF-8.

    Se usa para reportes auxiliares (`reporte_faltantes.json` y
    `parametros_normalizacion.json`) manteniendo una unica forma de escritura.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Define argumentos CLI para correr la etapa de preprocesamiento.

    Pasos:
    1. Registra ruta de entrada validada.
    2. Registra ruta de salida limpia.
    3. Registra rutas de JSON auxiliares.
    4. Registra `--sample` para pruebas.
    5. Devuelve argumentos parseados.

    Permite cambiar entrada validada, salida limpia, rutas de JSON auxiliares y
    una muestra opcional para pruebas rapidas.
    """
    parser = argparse.ArgumentParser(description="Preprocesa ventas validadas.")
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--missing-report-output", type=Path, default=MISSING_REPORT_PATH)
    parser.add_argument("--normalization-output", type=Path, default=NORMALIZATION_PATH)
    parser.add_argument("--sample", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Ejecuta carga, transformacion y guardado del dataset limpio.

    Pasos:
    1. Desactiva QuickEdit para evitar pausas accidentales en Windows.
    2. Lee argumentos y muestra rutas de entrada/salida.
    3. Carga el dataset validado.
    4. Construye reporte de faltantes antes de imputar.
    5. Limpia y transforma los datos.
    6. Genera resumen final.
    7. Guarda dataset limpio y archivos JSON de evidencia.
    """
    disable_quick_edit_mode()
    args = parse_args()
    print(f"CPYD_SEED={SEED}", flush=True)
    print("Archivo de entrada:", args.input, flush=True)
    print("Archivo de salida:", args.output, flush=True)

    df = cargar_datos(args.input)
    if args.sample and args.sample > 0:
        df = df.head(args.sample)

    print(f"Filas cargadas: {len(df)}", flush=True)

    start = log_step("analizando valores faltantes antes de imputar")
    df_reporte = df.copy()
    df_reporte.columns = [col.strip() for col in df_reporte.columns]
    if "FECHA NACIMIENTO" in df_reporte.columns:
        df_reporte["FECHA NACIMIENTO"] = pd.to_datetime(df_reporte["FECHA NACIMIENTO"], errors="coerce")
    for col in ["UNIDADES", "PORCENTAJE DESCUENTO", "MONTO APLICADO", "LOCAL", "GENERO"]:
        if col in df_reporte.columns:
            df_reporte[col] = pd.to_numeric(df_reporte[col], errors="coerce")
    reporte_faltantes = construir_reporte_faltantes(df_reporte)
    log_step("reporte de faltantes construido", start)

    clean = limpiar_y_transformar(df)
    print(f"Filas despues de limpieza: {len(clean)}", flush=True)

    resumen = resumir_preprocesamiento(clean)
    for key, value in resumen.items():
        print(f" - {key}: {value}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = log_step(f"guardando dataset limpio en {args.output}")
    try:
        guardar_dataset(clean, args.output)
    except PermissionError as exc:
        raise SystemExit(
            f"No se pudo guardar {args.output}. Cierra el archivo si esta abierto "
            "en el IDE, Excel u otro visor y vuelve a ejecutar."
        ) from exc
    log_step("dataset limpio guardado", start)
    guardar_resumen(resumen, args.summary_output)
    guardar_json(reporte_faltantes, args.missing_report_output)
    guardar_json(clean.attrs.get("parametros_normalizacion", {}), args.normalization_output)
    print(f"Datos guardados en: {args.output}", flush=True)
    print(f"Resumen guardado en: {args.summary_output}", flush=True)
    print(f"Reporte de faltantes guardado en: {args.missing_report_output}", flush=True)
    print(f"Parametros de normalizacion guardados en: {args.normalization_output}", flush=True)


if __name__ == "__main__":
    main()
