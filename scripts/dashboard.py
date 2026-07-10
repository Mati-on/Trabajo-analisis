"""Dashboard local para explorar resultados de Cruz Morada.

El dashboard abre una ventana de escritorio con resumenes y graficos del dataset
limpio. Permite filtrar por rango de fechas y recalcular las visualizaciones
para el periodo seleccionado.

No usa frameworks externos como Streamlit o Dash. Se apoya en librerias ya
presentes en el proyecto: pandas, matplotlib y tkinter.
"""

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "ventas_limpias.parquet"
DEFAULT_LOGS = ROOT / "logs.txt"
DEFAULT_PLOTS_DIR = ROOT / "plots"
DEFAULT_RESULTS_DIR = ROOT / "resultados"
MAX_PLOT_ROWS = 80_000
MAX_MODEL_ROWS = 100_000

USE_COLUMNS = [
    "FECHA",
    "CANAL",
    "SKU",
    "UNIDADES",
    "PORCENTAJE DESCUENTO",
    "MONTO APLICADO",
    "LOCAL",
    "GENERO",
    "EDAD",
    "FRECUENCIA COMPRA",
    "OUTLIER_MONTO",
    "OUTLIER_UNIDADES",
]


def load_sales(path: Path) -> pd.DataFrame:
    """Carga ventas limpias desde Parquet o CSV usando solo columnas necesarias.

    Pasos:
    1. Lee solo `USE_COLUMNS` para reducir memoria.
    2. Usa `read_parquet` si la extension es `.parquet`.
    3. Usa `read_csv` con separador `;` para compatibilidad.
    4. Convierte `FECHA` a datetime.
    5. Convierte columnas numericas disponibles.
    6. Elimina filas sin fecha o monto.

    Parquet es el formato normal del pipeline. La rama CSV se mantiene para
    compatibilidad con archivos externos o pruebas manuales.
    """
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, columns=USE_COLUMNS)
    else:
        df = pd.read_csv(path, sep=";", usecols=lambda col: col in USE_COLUMNS)
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")

    numeric_cols = [col for col in USE_COLUMNS if col != "FECHA" and col != "CANAL"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["FECHA", "MONTO APLICADO"])


def summarize_logs(path: Path) -> dict:
    """Resume los descartes registrados en logs.txt por tipo de motivo.

    Pasos:
    1. Si el log no existe, devuelve conteo cero.
    2. Recorre lineas del archivo.
    3. Ignora encabezados y separadores.
    4. Extrae el motivo principal antes del segundo `:`.
    5. Acumula total general y total por motivo.
    """
    summary = {"total": 0}
    if not path.exists():
        return summary

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if ":" not in line or line.startswith("=") or line.startswith("REGISTRO"):
                continue
            summary["total"] += 1
            parts = line.strip().split(":")
            reason = parts[1] if len(parts) > 1 else "SIN_MOTIVO"
            summary[reason] = summary.get(reason, 0) + 1
    return summary


def load_json(path: Path, default):
    """Carga un JSON de resultados y devuelve `default` si no existe.

    Pasos:
    1. Verifica si el archivo existe.
    2. Si existe, intenta parsearlo con `json.loads`.
    3. Si falta o esta corrupto, devuelve `default` para que el dashboard abra
       sin fallar.
    """
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def sample_for_plot(df: pd.DataFrame, max_rows: int = MAX_PLOT_ROWS) -> pd.DataFrame:
    """Reduce datos para graficos cuando el rango filtrado es muy grande.

    Pasos:
    1. Si el rango filtrado tiene hasta `max_rows`, devuelve todo.
    2. Si lo supera, toma una muestra reproducible.
    3. La muestra solo afecta visualizaciones, no los JSON ya calculados.
    """
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=42)


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara un subconjunto modelable para diagnosticos filtrados.

    Pasos:
    1. Copia los datos filtrados.
    2. Deriva variables temporales desde `FECHA`.
    3. Selecciona predictores disponibles.
    4. Elimina filas incompletas.
    5. Codifica `CANAL` con dummies.
    6. Devuelve una tabla lista para entrenar el modelo del dashboard.
    """
    model_df = df.copy()
    model_df["FECHA_MES"] = model_df["FECHA"].dt.month
    model_df["FECHA_DIA_SEMANA"] = model_df["FECHA"].dt.dayofweek
    model_df["FECHA_HORA"] = model_df["FECHA"].dt.hour

    required = ["MONTO APLICADO", "UNIDADES", "PORCENTAJE DESCUENTO", "EDAD", "FRECUENCIA COMPRA"]
    optional = ["SKU", "CANAL", "GENERO", "LOCAL", "FECHA_MES", "FECHA_DIA_SEMANA", "FECHA_HORA"]
    cols = [col for col in required + optional if col in model_df.columns]
    model_df = model_df[cols].dropna().copy()
    if "CANAL" in model_df.columns:
        model_df = pd.get_dummies(model_df, columns=["CANAL"], drop_first=True)
    return model_df


class DashboardApp:
    """Ventana principal del dashboard.

    Pasos:
    1. Se instancia desde `main` con Tk, datos, logs y carpetas.
    2. Construye controles y pestanas.
    3. Mantiene `self.filtered` como dataset activo.
    4. Redibuja resumen/graficos cuando cambia el filtro.

    Responsabilidades:
    1. Mantener el dataset limpio y el rango filtrado.
    2. Cargar resultados JSON ya calculados por el pipeline.
    3. Construir controles, resumen, pestanas y graficos.
    4. Recalcular visualizaciones al cambiar el filtro de fechas.
    """

    def __init__(
        self,
        root: tk.Tk,
        data: pd.DataFrame,
        log_summary: dict,
        plots_dir: Path,
        results_dir: Path,
    ):
        """Inicializa estado, controles y primera vista del dashboard.

        Pasos:
        1. Guarda referencias a ventana, datos, logs y carpetas.
        2. Calcula fechas validas disponibles.
        3. Inicializa estado del filtro y cache del modelo.
        4. Carga JSON de resultados.
        5. Configura titulo/tamano de ventana.
        6. Construye controles, resumen y pestanas.
        7. Aplica el filtro inicial con todo el rango disponible.
        """
        self.root = root
        self.data = data
        self.log_summary = log_summary
        self.plots_dir = plots_dir
        self.results_dir = results_dir
        self.available_dates = self._valid_date_values()
        self.filtered = data
        self.model_cache_key = None
        self.model_cache = None
        self.selected_png_path = None
        self.png_var = tk.StringVar()
        self.result_payloads = self.load_result_payloads()

        self.root.title("Dashboard Cruz Morada")
        self.root.geometry("1200x780")

        self.start_var = tk.StringVar(value=self.available_dates[0])
        self.end_var = tk.StringVar(value=self.available_dates[-1])

        self._build_controls()
        self._build_summary()
        self._build_tabs()
        self.apply_filter()

    def load_result_payloads(self) -> dict:
        """Carga resultados JSON usados por las pestanas informativas.

        Pasos:
        1. Lee cada archivo esperado en `resultados/`.
        2. Usa un valor por defecto si falta algun JSON.
        3. Devuelve un diccionario central para formatear pestanas de texto.
        """
        return {
            "resumen": load_json(self.results_dir / "resumen_preprocesamiento.json", {}),
            "faltantes": load_json(self.results_dir / "reporte_faltantes.json", {}),
            "normalizacion": load_json(self.results_dir / "parametros_normalizacion.json", {}),
            "descriptiva": load_json(self.results_dir / "estadisticas_descriptivas.json", []),
            "correlaciones": load_json(self.results_dir / "correlaciones.json", {}),
            "correlaciones_pvalues": load_json(self.results_dir / "correlaciones_pvalues.json", {}),
            "exploratorias": load_json(self.results_dir / "pruebas_exploratorias.json", {}),
            "temporal": load_json(self.results_dir / "analisis_temporal.json", {}),
            "modelo": load_json(self.results_dir / "metricas_modelo.json", {}),
            "regresion_lineal": load_json(self.results_dir / "diagnostico_regresion_lineal.json", {}),
            "hipotesis": load_json(self.results_dir / "pruebas_hipotesis.json", {}),
            "normalidad": load_json(self.results_dir / "normalidad.json", []),
        }

    def _valid_date_values(self) -> list[str]:
        """Devuelve fechas disponibles dentro del rango real del dataset.

        Pasos:
        1. Extrae `FECHA` sin nulos.
        2. Redondea cada fecha al dia.
        3. Ordena y elimina duplicados.
        4. Convierte a texto `YYYY-MM-DD`.
        5. Lanza error si no hay fechas validas.

        Usar una lista cerrada evita que el usuario escriba fechas inexistentes
        o fuera del periodo observado.
        """
        dates = self.data["FECHA"].dropna().dt.floor("D").sort_values().unique()
        if len(dates) == 0:
            raise ValueError("El dataset no contiene fechas validas para el dashboard.")
        return [pd.Timestamp(date).strftime("%Y-%m-%d") for date in dates]

    def _build_controls(self) -> None:
        """Crea la barra superior con filtros de fecha.

        Pasos:
        1. Muestra rango minimo/maximo disponible.
        2. Crea dos combobox de fechas validas.
        3. Agrega botones para aplicar/restablecer filtro.
        4. Agrega selector de PNG y boton para abrir el plot seleccionado.
        """
        frame = ttk.Frame(self.root, padding=8)
        frame.pack(fill="x")

        min_date = self.available_dates[0]
        max_date = self.available_dates[-1]

        ttk.Label(frame, text=f"Fecha inicio ({min_date} a {max_date}):").pack(side="left")
        ttk.Combobox(
            frame,
            textvariable=self.start_var,
            values=self.available_dates,
            width=14,
            state="readonly",
        ).pack(side="left", padx=(4, 16))

        ttk.Label(frame, text="Fecha fin:").pack(side="left")
        ttk.Combobox(
            frame,
            textvariable=self.end_var,
            values=self.available_dates,
            width=14,
            state="readonly",
        ).pack(side="left", padx=(4, 16))

        ttk.Button(frame, text="Aplicar filtro", command=self.apply_filter).pack(side="left")
        ttk.Button(frame, text="Restablecer", command=self.reset_filter).pack(side="left", padx=8)
        self.png_combo = ttk.Combobox(
            frame,
            textvariable=self.png_var,
            values=self.available_png_names(),
            width=30,
            state="readonly",
        )
        self.png_combo.pack(side="left", padx=(8, 4))
        ttk.Button(frame, text="Abrir PNG", command=self.open_selected_png).pack(side="left")

    def _build_summary(self) -> None:
        """Crea el panel textual de resumen.

        Pasos:
        1. Crea widget `Text`.
        2. Lo posiciona arriba de las pestanas.
        3. Lo deja en modo solo lectura.
        4. El contenido se carga despues desde `update_summary`.

        El panel se crea como `Text` de solo lectura y luego se actualiza desde
        `update_summary` cada vez que cambia el filtro de fechas.
        """
        self.summary_text = tk.Text(self.root, height=8, wrap="word")
        self.summary_text.pack(fill="x", padx=8, pady=(0, 8))
        self.summary_text.configure(state="disabled")

    def _build_tabs(self) -> None:
        """Crea pestanas para agrupar graficos.

        Pasos:
        1. Crea notebook principal.
        2. Agrega pestanas textuales alineadas con la rubrica.
        3. Agrega graficos principales que se actualizan por fecha.
        4. Agrega pestanas extra para modelo, hipotesis y PNG seleccionado.
        """
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=8)

        self.figures = {}
        self.axes = {}
        self.info_texts = {}

        for name, content in [
            ("Preprocesamiento", self.format_preprocessing_results()),
            ("Descriptiva", self.format_descriptive_results()),
            ("Normalidad", self.format_normality_results()),
            ("Correlaciones", self.format_correlation_results()),
            ("Asociaciones", self.format_association_results()),
            ("Temporal", self.format_temporal_results()),
        ]:
            self._build_text_tab(name, content)

        # Graficos principales solicitados por la rubrica. Todos se actualizan
        # al aplicar el filtro de fechas.
        for name in [
            "Hist monto",
            "Hist descuento",
            "Boxplot canal",
            "Matriz correlacion",
            "Serie temporal",
            "Descomposicion",
            "ACF/PACF",
        ]:
            self._build_figure_tab(name)

        # Opciones adicionales disponibles para exploracion y diagnostico.
        for name, content in [
            ("Modelo", self.format_model_results()),
            ("Hipotesis", self.format_hypothesis_results()),
        ]:
            self._build_text_tab(name, content)

        for name in [
            "Ventas por canal",
            "Modelo real vs predicho",
            "Residuos modelo",
        ]:
            self._build_figure_tab(name)

        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="PNG seleccionado")
        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.figures["PNG seleccionado"] = (fig, canvas)
        self.axes["PNG seleccionado"] = ax
        self.show_selected_png()

    def _build_figure_tab(self, name: str) -> None:
        """Crea una pestana con figura Matplotlib embebida.

        Pasos:
        1. Crea frame dentro del notebook.
        2. Crea `Figure` y `Axes`.
        3. Inserta `FigureCanvasTkAgg`.
        4. Guarda referencias para redibujar luego.

        Guarda referencias a `Figure`, `Canvas` y `Axes` para poder redibujar
        sin reconstruir toda la interfaz.
        """
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=name)
        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.figures[name] = (fig, canvas)
        self.axes[name] = ax

    def _build_text_tab(self, name: str, content: str) -> None:
        """Crea una pestana de texto solo lectura para resultados JSON.

        Pasos:
        1. Crea frame y scrollbar.
        2. Inserta contenido ya formateado.
        3. Bloquea edicion para evitar cambios accidentales.
        """
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text=name)
        text = tk.Text(tab, wrap="word")
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        text.insert(tk.END, content)
        text.configure(state="disabled")
        self.info_texts[name] = text

    def format_number(self, value) -> str:
        """Formatea numeros para lectura humana.

        Pasos:
        1. Convierte `None` a `N/A`.
        2. Formatea floats con precision compacta.
        3. Formatea enteros con separador de miles.
        4. Convierte otros valores a texto.

        Convierte `None` a `N/A`, floats a formato compacto e ints con separador
        de miles. Se usa en todos los paneles textuales.
        """
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:,.6g}"
        return f"{value:,}" if isinstance(value, int) else str(value)

    def format_preprocessing_results(self) -> str:
        """Resume faltantes, outliers y parametros de normalizacion.

        Pasos:
        1. Lee resumen, reporte de faltantes y normalizacion desde payloads.
        2. Agrega metricas principales de filas y outliers.
        3. Lista columnas con faltantes y su metodo de tratamiento.
        4. Lista parametros z-score por variable.
        5. Devuelve un texto listo para una pestana de solo lectura.
        """
        resumen = self.result_payloads["resumen"]
        faltantes = self.result_payloads["faltantes"]
        normalizacion = self.result_payloads["normalizacion"]

        lines = [
            "PREPROCESAMIENTO",
            "",
            f"Filas limpias: {self.format_number(resumen.get('filas_totales'))}",
            f"Outliers de monto: {self.format_number(resumen.get('outliers_monto'))}",
            f"Outliers de unidades: {self.format_number(resumen.get('outliers_unidades'))}",
            "",
            "FALTANTES",
            f"Filas evaluadas: {self.format_number(faltantes.get('filas_evaluadas'))}",
            f"Total faltantes: {self.format_number(faltantes.get('total_faltantes'))}",
            faltantes.get("nota_mcar", ""),
            "",
        ]

        for item in faltantes.get("columnas", []):
            if item.get("faltantes", 0) == 0:
                continue
            tratamiento = item.get("tratamiento", {})
            lines.extend(
                [
                    f"- {item.get('columna')}: {item.get('faltantes')} faltantes ({item.get('porcentaje'):.4f}%)",
                    f"  Metodo: {tratamiento.get('metodo', 'no aplica')}",
                    f"  Justificacion: {tratamiento.get('justificacion', 'no aplica')}",
                ]
            )
        if not any(item.get("faltantes", 0) for item in faltantes.get("columnas", [])):
            lines.append("No se detectaron faltantes en el dataset evaluado.")

        lines.extend(["", "NORMALIZACION Z-SCORE"])
        for variable, params in normalizacion.items():
            lines.extend(
                [
                    f"- {variable}",
                    f"  Columna generada: {params.get('columna_generada')}",
                    f"  Media: {self.format_number(params.get('media'))}",
                    f"  Desviacion: {self.format_number(params.get('desviacion'))}",
                    f"  Formula: {params.get('formula')}",
                ]
            )
        return "\n".join(lines)

    def format_model_results(self) -> str:
        """Resume metricas y configuracion del modelo guardado.

        Pasos:
        1. Lee metricas del modelo no lineal.
        2. Muestra filas de train/test, MAE, RMSE y R2.
        3. Lista predictores usados.
        4. Agrega diagnostico OLS si existe.
        5. Lista VIF, diagnosticos de residuos y limitaciones.
        """
        modelo = self.result_payloads["modelo"]
        metricas = modelo.get("metricas", {})
        lines = [
            "MODELO",
            "",
            f"Modelo: {modelo.get('modelo', 'N/A')}",
            f"Objetivo: {modelo.get('objetivo', 'N/A')}",
            f"Transformacion objetivo: {modelo.get('transformacion_objetivo', 'N/A')}",
            f"Muestra maxima: {self.format_number(modelo.get('muestra_maxima'))}",
            f"Filas usadas: {self.format_number(modelo.get('filas_usadas'))}",
            f"Train: {self.format_number(modelo.get('train'))}",
            f"Test: {self.format_number(modelo.get('test'))}",
            "",
            "METRICAS",
            f"MAE: {self.format_number(metricas.get('MAE'))}",
            f"RMSE: {self.format_number(metricas.get('RMSE'))}",
            f"R2: {self.format_number(metricas.get('R2'))}",
            "",
            "PREDICTORES",
        ]
        lines.extend(f"- {predictor}" for predictor in modelo.get("predictores", []))
        lines.extend(["", "REGRESION LINEAL INTERPRETABLE"])
        lineal = self.result_payloads["regresion_lineal"]
        if lineal:
            lines.extend(
                [
                    f"Modelo: {lineal.get('modelo', 'N/A')}",
                    f"Objetivo: {lineal.get('objetivo', 'N/A')}",
                    f"Filas usadas: {self.format_number(lineal.get('filas_usadas'))}",
                    f"Train: {self.format_number(lineal.get('train'))}",
                    f"Test: {self.format_number(lineal.get('test'))}",
                    f"R2 train: {self.format_number(lineal.get('r2_train'))}",
                    f"R2 ajustado train: {self.format_number(lineal.get('r2_ajustado_train'))}",
                    "",
                    "Metricas test escala original:",
                ]
            )
            for key, value in lineal.get("metricas_test_escala_original", {}).items():
                lines.append(f"- {key}: {self.format_number(value)}")
            lines.extend(["", "VIF:"])
            for item in lineal.get("vif", []):
                lines.append(f"- {item.get('variable')}: {self.format_number(item.get('vif'))}")
            lines.extend(["", "Diagnosticos residuos:"])
            for name, values in lineal.get("diagnosticos_residuos", {}).items():
                lines.append(f"- {name}")
                for key, value in values.items():
                    lines.append(f"  {key}: {self.format_number(value)}")
            lines.extend(["", "Limitaciones de extrapolabilidad:"])
            for item in lineal.get("extrapolabilidad", {}).get("limitaciones", []):
                lines.append(f"- {item}")
        else:
            lines.append("No hay diagnostico de regresion lineal disponible.")
        return "\n".join(lines)

    def format_descriptive_results(self) -> str:
        """Resume estadisticas descriptivas solicitadas por rubrica.

        Pasos:
        1. Lee `estadisticas_descriptivas.json`.
        2. Recorre cada variable numerica.
        3. Formatea tendencia central, dispersion, asimetria y curtosis.
        4. Devuelve texto para la pestana `Descriptiva`.
        """
        rows = self.result_payloads["descriptiva"]
        lines = [
            "ESTADISTICA DESCRIPTIVA",
            "",
            "Incluye tendencia central, dispersion, asimetria y curtosis para variables numericas.",
            "",
        ]
        for row in rows:
            lines.extend(
                [
                    f"- {row.get('variable')}",
                    f"  n: {self.format_number(row.get('n'))}",
                    f"  media: {self.format_number(row.get('media'))}",
                    f"  mediana: {self.format_number(row.get('mediana'))}",
                    f"  desviacion: {self.format_number(row.get('desviacion'))}",
                    f"  varianza: {self.format_number(row.get('varianza'))}",
                    f"  minimo: {self.format_number(row.get('minimo'))}",
                    f"  maximo: {self.format_number(row.get('maximo'))}",
                    f"  asimetria: {self.format_number(row.get('asimetria'))}",
                    f"  curtosis: {self.format_number(row.get('curtosis'))}",
                    "",
                ]
            )
        if not rows:
            lines.append("No hay estadisticas descriptivas disponibles.")
        return "\n".join(lines)

    def format_correlation_results(self) -> str:
        """Resume matriz Spearman y p-values asociados.

        Pasos:
        1. Lee matriz de correlaciones y matriz de p-values.
        2. Recorre cada par de variables.
        3. Formatea rho y p-value.
        4. Devuelve texto para auditoria de significancia.
        """
        corr = self.result_payloads["correlaciones"]
        pvalues = self.result_payloads["correlaciones_pvalues"]
        lines = [
            "CORRELACIONES",
            "",
            "Matriz Spearman con p-values asociados. Valores N/A indican variables sin variacion suficiente.",
            "",
        ]
        if not corr:
            lines.append("No hay correlaciones disponibles.")
            return "\n".join(lines)

        variables = list(corr.keys())
        for var_a in variables:
            lines.append(f"- {var_a}")
            for var_b, value in corr.get(var_a, {}).items():
                p_value = pvalues.get(var_a, {}).get(var_b) if isinstance(pvalues, dict) else None
                lines.append(
                    f"  {var_b}: rho={self.format_number(value)}, p={self.format_number(p_value)}"
                )
            lines.append("")
        return "\n".join(lines)

    def format_association_results(self) -> str:
        """Resume Chi-cuadrado, Spearman especifico y ANOVA/Kruskal.

        Pasos:
        1. Lee `pruebas_exploratorias.json`.
        2. Recorre cada prueba ejecutada.
        3. Muestra variables, estadisticos, p-values y notas.
        4. Devuelve texto para la pestana `Asociaciones`.
        """
        payload = self.result_payloads["exploratorias"]
        lines = [
            "ASOCIACIONES Y COMPARACIONES",
            "",
            "Incluye las pruebas exigidas por el enunciado: Chi-cuadrado, Spearman y ANOVA/Kruskal.",
            "",
        ]
        for result in payload.get("pruebas", []):
            lines.append(f"- {result.get('prueba')} | {result.get('variables')}")
            for key, value in result.items():
                if key in {"prueba", "variables"}:
                    continue
                lines.append(f"  {key}: {self.format_number(value)}")
            lines.append("")
        if not payload.get("pruebas"):
            lines.append("No hay pruebas exploratorias disponibles.")
        return "\n".join(lines)

    def format_temporal_results(self) -> str:
        """Resume analisis temporal solicitado por rubrica.

        Pasos:
        1. Lee `analisis_temporal.json`.
        2. Muestra dias observados.
        3. Informa estado de descomposicion y ACF/PACF.
        4. Aclara que las pestanas graficas se recalculan con el filtro.
        """
        payload = self.result_payloads["temporal"]
        lines = [
            "PATRONES TEMPORALES",
            "",
            f"Dias observados: {self.format_number(payload.get('dias_observados'))}",
            f"Descomposicion temporal: {payload.get('descomposicion_temporal', 'N/A')}",
            f"ACF/PACF: {payload.get('acf_pacf_temporal', 'N/A')}",
            "",
            "Las pestanas Serie temporal, Descomposicion y ACF/PACF se recalculan con el filtro de fechas.",
        ]
        return "\n".join(lines)

    def format_hypothesis_results(self) -> str:
        """Resume pruebas de hipotesis guardadas.

        Pasos:
        1. Lee `pruebas_hipotesis.json`.
        2. Recorre hipotesis.
        3. Muestra pregunta, H0 y H1.
        4. Agrega notas de no evaluable cuando existan.
        5. Lista pruebas, estadisticos y decision.
        """
        payload = self.result_payloads["hipotesis"]
        lines = ["HIPOTESIS", ""]
        for hipotesis in payload.get("hipotesis", []):
            lines.extend(
                [
                    f"{hipotesis.get('nombre')}: {hipotesis.get('pregunta')}",
                    f"H0: {hipotesis.get('h0')}",
                    f"H1: {hipotesis.get('h1')}",
                ]
            )
            if hipotesis.get("nota"):
                lines.append(f"Nota: {hipotesis.get('nota')}")
            for resultado in hipotesis.get("resultados", []):
                detalle = ", ".join(
                    f"{key}={self.format_number(value)}"
                    for key, value in resultado.items()
                    if key not in {"prueba", "decision"}
                )
                lines.append(f"  - {resultado.get('prueba')}: {detalle}; {resultado.get('decision')}")
            lines.append("")
        return "\n".join(lines)

    def format_normality_results(self) -> str:
        """Resume pruebas de normalidad guardadas.

        Pasos:
        1. Lee `normalidad.json`.
        2. Recorre cada variable evaluada.
        3. Muestra estadisticos y p-values de Shapiro y KS.
        4. Devuelve texto para la pestana `Normalidad`.
        """
        rows = self.result_payloads["normalidad"]
        lines = ["NORMALIDAD", ""]
        for row in rows:
            lines.extend(
                [
                    f"- {row.get('variable')}",
                    f"  Shapiro stat: {self.format_number(row.get('shapiro_stat'))}",
                    f"  Shapiro p-value: {self.format_number(row.get('shapiro_p'))}",
                    f"  KS stat: {self.format_number(row.get('ks_stat'))}",
                    f"  KS p-value: {self.format_number(row.get('ks_p'))}",
                ]
            )
        if not rows:
            lines.append("No hay resultados de normalidad disponibles.")
        return "\n".join(lines)

    def reset_filter(self) -> None:
        """Vuelve al rango completo del dataset y redibuja la interfaz.

        Pasos:
        1. Define fecha inicial como la primera disponible.
        2. Define fecha final como la ultima disponible.
        3. Llama `apply_filter` para recalcular resumen y graficos.
        """
        self.start_var.set(self.available_dates[0])
        self.end_var.set(self.available_dates[-1])
        self.apply_filter()

    def apply_filter(self) -> None:
        """Filtra el dataset por fechas y refresca resumen/graficos.

        Pasos:
        1. Convierte fechas seleccionadas en los combobox.
        2. Valida orden del rango.
        3. Valida que el rango este dentro de fechas reales del dataset.
        4. Filtra `self.data` y guarda en `self.filtered`.
        5. Actualiza resumen y graficos.
        """
        try:
            start = pd.to_datetime(self.start_var.get())
            end = pd.to_datetime(self.end_var.get()) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        except ValueError:
            messagebox.showerror("Fecha invalida", "Selecciona una fecha disponible del listado.")
            return

        if start > end:
            messagebox.showerror("Rango invalido", "La fecha inicial no puede ser mayor que la final.")
            return

        min_date = pd.to_datetime(self.available_dates[0])
        max_date = pd.to_datetime(self.available_dates[-1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        if start < min_date or end > max_date:
            messagebox.showerror(
                "Rango fuera de datos",
                f"Selecciona fechas entre {self.available_dates[0]} y {self.available_dates[-1]}.",
            )
            return

        self.filtered = self.data[(self.data["FECHA"] >= start) & (self.data["FECHA"] <= end)]
        if self.filtered.empty:
            messagebox.showwarning("Sin datos", "No hay registros para el rango seleccionado.")

        self.update_summary()
        self.update_plots()

    def update_summary(self) -> None:
        """Actualiza el resumen numerico del rango seleccionado.

        Pasos:
        1. Calcula cantidad de registros filtrados.
        2. Calcula monto total, promedio y mediana.
        3. Cuenta canales, locales y outliers.
        4. Agrega descartes globales desde `logs.txt`.
        5. Reemplaza el contenido del widget de resumen.

        Calcula transacciones, monto total, ticket promedio, mediana, canales,
        locales y outliers del periodo filtrado; ademas muestra descartes
        globales leidos desde `logs.txt`.
        """
        df = self.filtered
        total = len(df)
        total_monto = df["MONTO APLICADO"].sum()
        ticket_promedio = df["MONTO APLICADO"].mean() if total else 0
        mediana = df["MONTO APLICADO"].median() if total else 0
        outliers_monto = int(df.get("OUTLIER_MONTO", pd.Series(dtype=int)).sum()) if total else 0
        canales = df["CANAL"].nunique() if total and "CANAL" in df.columns else 0
        locales = df["LOCAL"].nunique() if total and "LOCAL" in df.columns else 0

        lines = [
            "RESUMEN DEL PERIODO",
            f"Rango: {self.start_var.get()} a {self.end_var.get()}",
            f"Transacciones limpias: {total:,}",
            f"Monto total: ${total_monto:,.0f}",
            f"Ticket promedio: ${ticket_promedio:,.2f}",
            f"Mediana de monto: ${mediana:,.2f}",
            f"Canales activos: {canales}",
            f"Locales activos: {locales}",
            f"Outliers de monto en periodo: {outliers_monto:,}",
            "",
            "DESCARTES GLOBALES DE VALIDACION",
            f"Total descartadas: {self.log_summary.get('total', 0):,}",
            f"Descartadas por edad: {self.log_summary.get('EDAD', 0):,}",
        ]

        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, "\n".join(lines))
        self.summary_text.configure(state="disabled")

    def update_plots(self) -> None:
        """Redibuja todos los graficos usando el rango filtrado.

        Pasos:
        1. Actualiza serie temporal.
        2. Actualiza graficos de canal, histogramas, boxplot y correlacion.
        3. Actualiza descomposicion y ACF/PACF.
        4. Actualiza diagnosticos de modelo filtrado.

        Centraliza las llamadas para que `apply_filter` solo tenga que llamar a
        una funcion despues de modificar `self.filtered`.
        """
        self.plot_time_series()
        self.plot_channels()
        self.plot_hist_amount()
        self.plot_hist_discount()
        self.plot_boxplot_channel()
        self.plot_correlation()
        self.plot_decomposition()
        self.plot_acf_pacf()
        self.plot_model_predictions()
        self.plot_model_residuals()

    def plot_time_series(self) -> None:
        """Grafica ventas agregadas por dia.

        Pasos:
        1. Limpia el eje actual.
        2. Agrupa monto por dia dentro del filtro activo.
        3. Dibuja linea temporal.
        4. Actualiza titulos/ejes y refresca canvas.
        """
        fig, canvas = self.figures["Serie temporal"]
        ax = self.axes["Serie temporal"]
        ax.clear()

        if not self.filtered.empty:
            daily = self.filtered.groupby(self.filtered["FECHA"].dt.floor("D"))["MONTO APLICADO"].sum()
            daily.plot(ax=ax, marker="o", linewidth=1.5)
        ax.set_title("Ventas diarias del periodo")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Monto aplicado")
        fig.tight_layout()
        canvas.draw()

    def plot_channels(self) -> None:
        """Grafica monto total por canal.

        Pasos:
        1. Limpia el eje.
        2. Agrupa monto por `CANAL`.
        3. Ordena de mayor a menor.
        4. Dibuja barras y refresca canvas.
        """
        fig, canvas = self.figures["Ventas por canal"]
        ax = self.axes["Ventas por canal"]
        ax.clear()

        if not self.filtered.empty and "CANAL" in self.filtered.columns:
            by_channel = self.filtered.groupby("CANAL")["MONTO APLICADO"].sum().sort_values(ascending=False)
            by_channel.plot(kind="bar", ax=ax)
        ax.set_title("Monto total por canal")
        ax.set_xlabel("Canal")
        ax.set_ylabel("Monto aplicado")
        fig.tight_layout()
        canvas.draw()

    def plot_hist_amount(self) -> None:
        """Grafica distribucion del monto aplicado.

        Pasos:
        1. Toma muestra si el filtro tiene muchas filas.
        2. Dibuja histograma normalizado de `MONTO APLICADO`.
        3. Configura etiquetas.
        4. Refresca canvas.
        """
        fig, canvas = self.figures["Hist monto"]
        ax = self.axes["Hist monto"]
        ax.clear()

        sample = sample_for_plot(self.filtered)
        if not sample.empty:
            ax.hist(sample["MONTO APLICADO"].dropna(), bins=50, density=True)
        ax.set_title("Distribucion de MONTO APLICADO")
        ax.set_xlabel("Monto aplicado")
        ax.set_ylabel("Densidad")
        fig.tight_layout()
        canvas.draw()

    def plot_hist_discount(self) -> None:
        """Grafica distribucion del porcentaje de descuento.

        Pasos:
        1. Toma muestra reproducible del filtro.
        2. Dibuja histograma normalizado de `PORCENTAJE DESCUENTO`.
        3. Configura etiquetas.
        4. Refresca canvas.
        """
        fig, canvas = self.figures["Hist descuento"]
        ax = self.axes["Hist descuento"]
        ax.clear()

        sample = sample_for_plot(self.filtered)
        if not sample.empty and "PORCENTAJE DESCUENTO" in sample.columns:
            ax.hist(sample["PORCENTAJE DESCUENTO"].dropna(), bins=40, density=True)
        ax.set_title("Distribucion de PORCENTAJE DESCUENTO")
        ax.set_xlabel("Porcentaje descuento")
        ax.set_ylabel("Densidad")
        fig.tight_layout()
        canvas.draw()

    def plot_boxplot_channel(self) -> None:
        """Grafica boxplot de monto aplicado por canal.

        Pasos:
        1. Filtra columnas necesarias.
        2. Toma muestra para no saturar Matplotlib.
        3. Agrupa montos por canal.
        4. Dibuja boxplot sin fliers extremos.
        5. Refresca canvas.
        """
        fig, canvas = self.figures["Boxplot canal"]
        ax = self.axes["Boxplot canal"]
        ax.clear()

        if not self.filtered.empty and {"CANAL", "MONTO APLICADO"}.issubset(self.filtered.columns):
            sample = sample_for_plot(self.filtered[["CANAL", "MONTO APLICADO"]].dropna())
            groups = [
                group["MONTO APLICADO"].to_numpy()
                for _, group in sample.groupby("CANAL")
                if len(group) > 0
            ]
            labels = [
                str(channel)
                for channel, group in sample.groupby("CANAL")
                if len(group) > 0
            ]
            if groups:
                try:
                    ax.boxplot(groups, tick_labels=labels, showfliers=False)
                except TypeError:
                    ax.boxplot(groups, labels=labels, showfliers=False)
        ax.set_title("MONTO APLICADO por CANAL")
        ax.set_xlabel("Canal")
        ax.set_ylabel("Monto aplicado")
        fig.tight_layout()
        canvas.draw()

    def plot_correlation(self) -> None:
        """Grafica una matriz de correlacion simple para variables numericas.

        Pasos:
        1. Selecciona variables numericas disponibles.
        2. Toma muestra filtrada sin nulos.
        3. Calcula Spearman para el periodo.
        4. Dibuja heatmap con anotaciones.
        5. Refresca canvas.
        """
        fig, canvas = self.figures["Matriz correlacion"]
        fig.clear()
        ax = fig.add_subplot(111)
        self.axes["Matriz correlacion"] = ax

        cols = [
            "PORCENTAJE DESCUENTO",
            "MONTO APLICADO",
            "EDAD",
            "FRECUENCIA COMPRA",
        ]
        cols = [col for col in cols if col in self.filtered.columns]
        sample = sample_for_plot(self.filtered[cols].dropna()) if cols else pd.DataFrame()

        if len(sample) > 1:
            corr = sample.corr(method="spearman")
            image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xticks(np.arange(len(cols)), labels=cols, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(cols)), labels=cols)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Correlacion Spearman del periodo")
        fig.tight_layout()
        canvas.draw()

    def daily_sales(self) -> pd.Series:
        """Devuelve ventas diarias ordenadas para graficos temporales.

        Pasos:
        1. Si no hay datos filtrados, devuelve serie vacia.
        2. Agrupa por fecha truncada al dia.
        3. Suma `MONTO APLICADO`.
        4. Ordena por fecha.

        Agrupa por fecha truncada al dia y suma `MONTO APLICADO`. Si el filtro
        no tiene datos, devuelve una serie vacia.
        """
        if self.filtered.empty:
            return pd.Series(dtype=float)
        daily = self.filtered.groupby(self.filtered["FECHA"].dt.floor("D"))["MONTO APLICADO"].sum()
        return daily.sort_index()

    def plot_decomposition(self) -> None:
        """Grafica descomposicion semanal para el periodo filtrado.

        Pasos:
        1. Obtiene serie diaria filtrada.
        2. Si hay al menos 14 dias, completa frecuencia diaria.
        3. Ejecuta `seasonal_decompose` con periodo 7.
        4. Dibuja observado, tendencia, estacionalidad y residuo.
        5. Si no hay datos suficientes, muestra mensaje explicativo.
        """
        fig, canvas = self.figures["Descomposicion"]
        fig.clear()

        daily = self.daily_sales()
        if len(daily) >= 14:
            try:
                daily_full = daily.asfreq("D", fill_value=0.0)
                decomposition = seasonal_decompose(daily_full, model="additive", period=7)
                axes = fig.subplots(4, 1, sharex=True)
                axes[0].plot(decomposition.observed)
                axes[0].set_ylabel("Observado")
                axes[1].plot(decomposition.trend)
                axes[1].set_ylabel("Tendencia")
                axes[2].plot(decomposition.seasonal)
                axes[2].set_ylabel("Estacional")
                axes[3].plot(decomposition.resid)
                axes[3].set_ylabel("Residuo")
                axes[0].set_title("Descomposicion temporal semanal")
            except ValueError as exc:
                ax = fig.add_subplot(111)
                ax.axis("off")
                ax.text(0.5, 0.5, f"No se pudo descomponer la serie: {exc}", ha="center", va="center")
        else:
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, "Se requieren al menos 14 dias para descomposicion temporal.", ha="center", va="center")
        fig.tight_layout()
        canvas.draw()

    def plot_acf_pacf(self) -> None:
        """Grafica autocorrelacion y autocorrelacion parcial filtradas.

        Pasos:
        1. Obtiene serie diaria filtrada.
        2. Verifica que existan al menos 10 dias.
        3. Calcula cantidad de rezagos segura.
        4. Dibuja ACF y PACF.
        5. Si no hay datos suficientes, muestra mensaje explicativo.
        """
        fig, canvas = self.figures["ACF/PACF"]
        fig.clear()

        daily = self.daily_sales()
        if len(daily) >= 10:
            lags = min(30, max(1, len(daily) // 2 - 1))
            axes = fig.subplots(1, 2)
            plot_acf(daily, ax=axes[0], lags=lags, title="Autocorrelacion (ACF)")
            plot_pacf(daily, ax=axes[1], lags=lags, title="Autocorrelacion parcial (PACF)")
        else:
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, "Se requieren al menos 10 dias para ACF/PACF.", ha="center", va="center")
        fig.tight_layout()
        canvas.draw()

    def get_model_diagnostics(self):
        """Entrena un modelo liviano para el periodo filtrado y cachea resultados.

        Pasos:
        1. Usa `(fecha_inicio, fecha_fin, filas)` como clave de cache.
        2. Si el filtro no cambio, reutiliza predicciones anteriores.
        3. Prepara variables con `build_model_frame`.
        4. Toma muestra si el periodo tiene demasiadas filas.
        5. Entrena `HistGradientBoostingRegressor` sobre `log1p`.
        6. Devuelve `y_test` y `y_pred` para graficos de modelo.
        """
        key = (self.start_var.get(), self.end_var.get(), len(self.filtered))
        if self.model_cache_key == key:
            return self.model_cache

        model_df = build_model_frame(self.filtered)
        if len(model_df) < 10:
            self.model_cache_key = key
            self.model_cache = None
            return None

        if len(model_df) > MAX_MODEL_ROWS:
            model_df = model_df.sample(MAX_MODEL_ROWS, random_state=42)

        X = model_df.drop(columns=["MONTO APLICADO"]).astype(float)
        y = model_df["MONTO APLICADO"].astype(float)
        if X.shape[1] == 0:
            self.model_cache_key = key
            self.model_cache = None
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=42,
        )
        model.fit(X_train, np.log1p(y_train))
        y_pred = np.clip(np.expm1(model.predict(X_test)), 0, None)

        if len(y_test) > MAX_PLOT_ROWS:
            sample_index = y_test.sample(MAX_PLOT_ROWS, random_state=42).index
            y_test = y_test.loc[sample_index]
            y_pred = pd.Series(y_pred, index=X_test.index).loc[sample_index].to_numpy()

        self.model_cache_key = key
        self.model_cache = (y_test, y_pred)
        return self.model_cache

    def plot_model_predictions(self) -> None:
        """Grafica real vs predicho del modelo entrenado para el filtro.

        Pasos:
        1. Obtiene diagnosticos/cache del modelo filtrado.
        2. Si hay datos, dibuja scatter real vs predicho.
        3. Agrega diagonal de referencia.
        4. Si no hay datos, muestra mensaje.
        5. Refresca canvas.
        """
        fig, canvas = self.figures["Modelo real vs predicho"]
        ax = self.axes["Modelo real vs predicho"]
        ax.clear()

        diagnostics = self.get_model_diagnostics()
        if diagnostics:
            y_test, y_pred = diagnostics
            ax.scatter(y_test, y_pred, alpha=0.25, s=8)
            min_v = min(y_test.min(), y_pred.min())
            max_v = max(y_test.max(), y_pred.max())
            ax.plot([min_v, max_v], [min_v, max_v], color="red", linestyle="--")
        else:
            ax.text(0.5, 0.5, "Datos insuficientes para entrenar modelo en este periodo.", ha="center", va="center")
        ax.set_title("Modelo filtrado: real vs predicho")
        ax.set_xlabel("Monto real")
        ax.set_ylabel("Monto predicho")
        fig.tight_layout()
        canvas.draw()

    def plot_model_residuals(self) -> None:
        """Grafica residuos del modelo entrenado para el filtro.

        Pasos:
        1. Obtiene diagnosticos/cache del modelo filtrado.
        2. Calcula residuo como real menos predicho.
        3. Dibuja residuo vs prediccion.
        4. Agrega linea horizontal en cero.
        5. Refresca canvas.
        """
        fig, canvas = self.figures["Residuos modelo"]
        ax = self.axes["Residuos modelo"]
        ax.clear()

        diagnostics = self.get_model_diagnostics()
        if diagnostics:
            y_test, y_pred = diagnostics
            residuals = y_test.to_numpy() - y_pred
            ax.scatter(y_pred, residuals, alpha=0.25, s=8)
            ax.axhline(0, color="red", linestyle="--")
        else:
            ax.text(0.5, 0.5, "Datos insuficientes para entrenar modelo en este periodo.", ha="center", va="center")
        ax.set_title("Residuos del modelo filtrado")
        ax.set_xlabel("Monto predicho")
        ax.set_ylabel("Residuo")
        fig.tight_layout()
        canvas.draw()

    def available_png_names(self) -> list[str]:
        """Lista los PNG disponibles en la carpeta de plots, ordenados por nombre.

        Pasos:
        1. Busca archivos `*.png` en `plots_dir`.
        2. Ordena por ruta/nombre.
        3. Devuelve solo el nombre de archivo para mostrarlo en el combobox.
        """
        return [path.name for path in sorted(self.plots_dir.glob("*.png"))]

    def open_selected_png(self) -> None:
        """Abre el PNG seleccionado desde el listado interno.

        Pasos:
        1. Refresca lista de PNG disponibles.
        2. Si no hay seleccion, usa el primer PNG disponible.
        3. Valida que el archivo exista.
        4. Guarda la ruta seleccionada.
        5. Redibuja la pestana `PNG seleccionado` y cambia a ella.
        """
        names = self.available_png_names()
        self.png_combo.configure(values=names)

        selected = self.png_var.get()
        if not selected:
            if names:
                selected = names[0]
                self.png_var.set(selected)
            else:
                messagebox.showwarning("Sin plots", f"No hay archivos PNG en {self.plots_dir}.")
                return

        path = self.plots_dir / selected
        if not path.exists():
            messagebox.showerror("PNG no encontrado", f"No existe el archivo seleccionado: {path}")
            return
        self.selected_png_path = path
        self.show_selected_png()
        self.tabs.select(len(self.tabs.tabs()) - 1)

    def show_selected_png(self) -> None:
        """Dibuja el PNG seleccionado en su pestana.

        Pasos:
        1. Limpia el eje de la pestana.
        2. Si hay PNG seleccionado, lo lee como imagen.
        3. Lo dibuja dentro del panel.
        4. Si no hay PNG, muestra un texto de ayuda.
        5. Refresca el canvas.

        Si no hay PNG seleccionado, muestra una instruccion breve dentro del
        panel. Si existe, lo lee con `matplotlib.image.imread` y lo muestra.
        """
        fig, canvas = self.figures["PNG seleccionado"]
        ax = self.axes["PNG seleccionado"]
        ax.clear()
        ax.axis("off")

        if self.selected_png_path and self.selected_png_path.exists():
            image = mpimg.imread(self.selected_png_path)
            ax.imshow(image)
            ax.set_title(self.selected_png_path.name)
        else:
            ax.text(
                0.5,
                0.5,
                "Usa el boton Abrir PNG para revisar un plot generado.",
                ha="center",
                va="center",
            )
        fig.tight_layout()
        canvas.draw()


def parse_args() -> argparse.Namespace:
    """Define rutas de entrada para ejecutar el dashboard.

    Pasos:
    1. Registra dataset limpio.
    2. Registra archivo de logs.
    3. Registra carpeta de plots.
    4. Registra carpeta de resultados JSON.
    5. Devuelve argumentos parseados.
    """
    parser = argparse.ArgumentParser(description="Abre dashboard local de resultados.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    """Carga datos y abre la ventana del dashboard.

    Pasos:
    1. Lee rutas por CLI.
    2. Carga ventas limpias.
    3. Resume logs de descartes.
    4. Crea la ventana Tk.
    5. Instancia `DashboardApp` y entra a `mainloop`.
    """
    args = parse_args()
    data = load_sales(args.input)
    log_summary = summarize_logs(args.logs)

    root = tk.Tk()
    DashboardApp(root, data, log_summary, args.plots_dir, args.results_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
