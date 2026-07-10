"""Analisis exploratorio estadistico para ventas limpias.

Este modulo toma `data/ventas_limpias.parquet` y genera resultados interpretables
para el informe: estadisticas descriptivas, pruebas exploratorias y graficos.
Cuando el dataset es muy grande, algunos graficos usan muestras reproducibles
para reducir tiempo y memoria sin cambiar los calculos tabulares principales.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Matplotlib necesita una carpeta de cache escribible. Se fija dentro del
# proyecto para evitar errores en entornos donde AppData/Temp esta bloqueado.
MPL_CACHE = ROOT / "resultados" / ".matplotlib_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

DEFAULT_INPUT = ROOT / "data" / "ventas_limpias.parquet"
DEFAULT_PLOTS_DIR = ROOT / "plots"
DEFAULT_RESULTS_DIR = ROOT / "resultados"
DEFAULT_WORKERS = 4
NUMERIC_COLUMNS = [
    "UNIDADES",
    "PORCENTAJE DESCUENTO",
    "MONTO APLICADO",
    "MONTO POR UNIDAD",
    "EDAD",
    "FRECUENCIA COMPRA",
]


def get_seed() -> int:
    """Lee la semilla reproducible del entorno.

    Pasos:
    1. Busca `CPYD_SEED` en variables de entorno.
    2. Si no existe, usa `42`.
    3. Convierte el valor a entero y lo devuelve.

    La semilla controla los muestreos usados en graficos para que dos corridas
    generen las mismas visualizaciones.
    """
    return int(os.environ.get("CPYD_SEED", "42"))


def load_data(path: Path) -> pd.DataFrame:
    """Carga ventas limpias desde Parquet o CSV y normaliza tipos del EDA.

    Pasos:
    1. Lee Parquet o CSV segun extension.
    2. Limpia espacios en nombres de columnas.
    3. Convierte `FECHA` a datetime.
    4. Convierte variables numericas usadas por estadisticas, pruebas y plots.

    Parquet es el formato recomendado porque evita el costo de leer CSV
    intermedios grandes. CSV queda soportado para compatibilidad.
    """
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [col.strip() for col in df.columns]
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    for col in NUMERIC_COLUMNS + ["LOCAL", "GENERO"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_descriptive_stats(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Guarda tendencia central, dispersion, asimetria y curtosis.

    Pasos:
    1. Recorre cada variable numerica disponible.
    2. Elimina nulos y fuerza tipo float.
    3. Calcula n, media, mediana, desviacion, varianza, minimo, maximo,
       asimetria y curtosis.
    4. Guarda `estadisticas_descriptivas.json`.
    5. Devuelve el resultado como `DataFrame`.

    Estas metricas cubren la parte descriptiva de la rubrica y quedan en JSON
    para poder citarlas directamente en el informe tecnico y en el dashboard.
    """
    rows = []
    for col in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        serie = df[col].dropna().astype(float)
        rows.append(
            {
                "variable": col,
                "n": int(serie.count()),
                "media": serie.mean(),
                "mediana": serie.median(),
                "desviacion": serie.std(),
                "varianza": serie.var(),
                "minimo": serie.min(),
                "maximo": serie.max(),
                "asimetria": serie.skew(),
                "curtosis": serie.kurtosis(),
            }
        )
    stats_df = pd.DataFrame(rows)
    (results_dir / "estadisticas_descriptivas.json").write_text(
        stats_df.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8"
    )
    return stats_df


def save_normality_tests(df: pd.DataFrame, results_dir: Path, seed: int) -> pd.DataFrame:
    """Aplica Shapiro y Kolmogorov-Smirnov sobre muestras reproducibles.

    Pasos:
    1. Recorre variables numericas disponibles.
    2. Toma hasta 5.000 observaciones reproducibles por variable.
    3. Si la muestra tiene variacion, aplica Shapiro-Wilk.
    4. Estandariza la muestra y aplica Kolmogorov-Smirnov contra normal.
    5. Si no hay variacion, marca la variable como no evaluable.
    6. Guarda `normalidad.json`.

    En datasets de millones de filas los tests de normalidad son muy sensibles y
    costosos. Por eso se limita a 5.000 observaciones reproducibles por variable.
    """
    rows = []
    for col in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        serie = df[col].dropna().astype(float)
        if len(serie) < 3:
            continue
        sample = serie.sample(min(len(serie), 5000), random_state=seed)
        std = sample.std()
        if std and std > 0:
            shapiro_stat, shapiro_p = stats.shapiro(sample)
            standardized = (sample - sample.mean()) / std
            ks_stat, ks_p = stats.kstest(standardized, "norm")
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
            ks_stat, ks_p = np.nan, np.nan
        rows.append(
            {
                "variable": col,
                "n_muestra": int(len(sample)),
                "shapiro_stat": shapiro_stat,
                "shapiro_p": shapiro_p,
                "ks_stat": ks_stat,
                "ks_p": ks_p,
                "nota": None if std and std > 0 else "No evaluable: variable sin variacion.",
            }
        )
    normality = pd.DataFrame(rows)
    (results_dir / "normalidad.json").write_text(
        normality.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8"
    )
    return normality


def save_correlations(df: pd.DataFrame, results_dir: Path, seed: int) -> pd.DataFrame:
    """Guarda matriz de correlacion de Spearman para variables numericas.

    Pasos:
    1. Selecciona variables numericas disponibles.
    2. Calcula matriz Spearman y guarda `correlaciones.json`.
    3. Construye una muestra sin nulos para p-values.
    4. Si hay mas de 200.000 filas, reduce a muestra reproducible.
    5. Calcula p-values por par cuando ambas variables tienen variacion.
    6. Guarda `correlaciones_pvalues.json`.

    Spearman es apropiado porque las ventas y descuentos no necesariamente
    siguen distribuciones normales y pueden tener outliers.
    """
    cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    corr = df[cols].corr(method="spearman")
    (results_dir / "correlaciones.json").write_text(
        corr.to_json(orient="index", indent=2), encoding="utf-8"
    )

    # Matriz de significancia asociada a la correlacion. Para mantener tiempo y
    # memoria acotados, se calcula sobre una muestra reproducible cuando el
    # dataset es muy grande.
    sample = df[cols].dropna()
    if len(sample) > 200_000:
        sample = sample.sample(200_000, random_state=seed)
    pvalues = pd.DataFrame(np.nan, index=cols, columns=cols)
    for col_a in cols:
        for col_b in cols:
            if col_a == col_b:
                pvalues.loc[col_a, col_b] = 0.0
            elif sample[col_a].std() > 0 and sample[col_b].std() > 0:
                _, p_value = stats.spearmanr(sample[col_a], sample[col_b])
                pvalues.loc[col_a, col_b] = p_value
    (results_dir / "correlaciones_pvalues.json").write_text(
        pvalues.to_json(orient="index", indent=2), encoding="utf-8"
    )
    return corr


def save_association_tests(df: pd.DataFrame, results_dir: Path) -> None:
    """Guarda pruebas Chi-cuadrado, Spearman y ANOVA/Kruskal.

    Pasos:
    1. Evalua `CANAL` vs `LOCAL` con Chi-cuadrado.
    2. Evalua asociaciones numericas solicitadas con Spearman.
    3. Evalua diferencias de `MONTO APLICADO` por `CANAL` y `LOCAL` con ANOVA.
    4. Agrega Kruskal-Wallis como alternativa no parametrica.
    5. Si hay demasiados locales, limita a los 200 con mas registros.
    6. Agrega Chi-cuadrado complementario `CANAL` vs `GENERO`.
    7. Guarda `pruebas_exploratorias.json`.

    Estas pruebas cubren asociaciones categoricas, numericas y diferencias de
    monto por grupo. Se guardan en JSON para mantener una salida estructurada y
    facil de consumir desde otros programas.
    """
    results = []

    if {"CANAL", "LOCAL"}.issubset(df.columns):
        table = pd.crosstab(df["CANAL"], df["LOCAL"])
        if table.shape[0] > 1 and table.shape[1] > 1:
            chi2, p_value, dof, _ = stats.chi2_contingency(table)
            results.append(
                {
                    "prueba": "Chi-cuadrado",
                    "variables": "CANAL vs LOCAL",
                    "chi2": chi2,
                    "grados_libertad": int(dof),
                    "p_value": p_value,
                }
            )

    # Correlaciones especificas solicitadas por rubrica entre UNIDADES, MONTO
    # APLICADO y PORCENTAJE DESCUENTO. Si una variable no tiene variacion, se
    # deja evidencia explicita en el JSON.
    target_corr = ["UNIDADES", "MONTO APLICADO", "PORCENTAJE DESCUENTO"]
    for idx, col_a in enumerate(target_corr):
        for col_b in target_corr[idx + 1 :]:
            if {col_a, col_b}.issubset(df.columns):
                pair = df[[col_a, col_b]].dropna()
                result = {
                    "prueba": "Spearman",
                    "variables": f"{col_a} vs {col_b}",
                    "rho": None,
                    "p_value": None,
                    "nota": None,
                }
                if len(pair) > 2 and pair[col_a].std() > 0 and pair[col_b].std() > 0:
                    rho, p_value = stats.spearmanr(pair[col_a], pair[col_b])
                    result["rho"] = rho
                    result["p_value"] = p_value
                else:
                    result["nota"] = "No evaluable: una o ambas variables no tienen variacion suficiente."
                results.append(result)

    for group_col in ["CANAL", "LOCAL"]:
        if {group_col, "MONTO APLICADO"}.issubset(df.columns):
            grouped = df[[group_col, "MONTO APLICADO"]].dropna()
            if group_col == "LOCAL" and grouped[group_col].nunique() > 200:
                top_groups = grouped[group_col].value_counts().head(200).index
                grouped = grouped[grouped[group_col].isin(top_groups)]
                local_note = "Evaluado sobre los 200 locales con mas registros para controlar costo computacional."
            else:
                local_note = None
            groups = [
                group["MONTO APLICADO"].dropna().to_numpy()
                for _, group in grouped.groupby(group_col)
                if len(group["MONTO APLICADO"].dropna()) > 1
            ]
            if len(groups) > 1:
                f_stat, p_anova = stats.f_oneway(*groups)
                h_stat, p_kruskal = stats.kruskal(*groups)
                results.append(
                    {
                        "prueba": "ANOVA",
                        "variables": f"MONTO APLICADO por {group_col}",
                        "estadistico": f_stat,
                        "p_value": p_anova,
                        "nota": local_note,
                    }
                )
                results.append(
                    {
                        "prueba": "Kruskal-Wallis",
                        "variables": f"MONTO APLICADO por {group_col}",
                        "estadistico": h_stat,
                        "p_value": p_kruskal,
                        "nota": local_note,
                    }
                )

    if {"CANAL", "GENERO"}.issubset(df.columns):
        table = pd.crosstab(df["CANAL"], df["GENERO"])
        if table.shape[0] > 1 and table.shape[1] > 1:
            chi2, p_value, dof, _ = stats.chi2_contingency(table)
            results.append(
                {
                    "prueba": "Chi-cuadrado",
                    "variables": "CANAL vs GENERO",
                    "chi2": chi2,
                    "grados_libertad": int(dof),
                    "p_value": p_value,
                    "nota": "Prueba complementaria no exigida directamente por la rubrica.",
                }
            )

    payload = {
        "pruebas": results,
        "nota": None if results else "No se ejecutaron pruebas exploratorias: faltan grupos o variacion suficiente.",
    }
    (results_dir / "pruebas_exploratorias.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )


def sample_df(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Toma muestra reproducible para graficos pesados.

    Pasos:
    1. Si el dataset cabe dentro de `max_rows`, lo devuelve completo.
    2. Si supera el limite, toma una muestra reproducible con `random_state`.
    3. Esta muestra solo se usa para graficos, no para reemplazar los calculos
       tabulares principales.

    Los graficos con millones de puntos pueden ser lentos y poco legibles. La
    muestra mantiene una representacion visual estable usando `CPYD_SEED`.
    """
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=seed)


def save_temporal_analysis(daily: pd.Series, plots_dir: Path, results_dir: Path) -> None:
    """Genera analisis temporal avanzado cuando la serie tiene suficientes dias.

    Pasos:
    1. Registra cuantos dias tiene la serie.
    2. Si hay al menos 14 dias, intenta descomposicion semanal.
    3. Si hay al menos 10 dias, genera ACF/PACF.
    4. Guarda PNG cuando corresponde.
    5. Guarda `analisis_temporal.json` indicando que se genero o por que no.

    Si hay al menos dos semanas se genera descomposicion semanal. Si hay al
    menos diez dias se generan ACF/PACF. Con menos datos se deja una nota en
    `analisis_temporal.json` en vez de producir graficos enganosos.
    """
    payload = {
        "dias_observados": int(len(daily)),
        "descomposicion_temporal": None,
        "acf_pacf_temporal": None,
    }

    if len(daily) >= 14:
        daily_full = daily.asfreq("D", fill_value=0.0)
        try:
            print(" - generando descomposicion_temporal.png", flush=True)
            decomposition = seasonal_decompose(daily_full, model="additive", period=7)
            fig = decomposition.plot()
            fig.set_size_inches(11, 8)
            plt.tight_layout()
            plt.savefig(plots_dir / "descomposicion_temporal.png")
            plt.close()
            payload["descomposicion_temporal"] = "generada"
        except ValueError as exc:
            payload["descomposicion_temporal"] = f"no_generada: {exc}"
    else:
        payload["descomposicion_temporal"] = "no_generada: se requieren al menos 14 dias"

    if len(daily) >= 10:
        print(" - generando acf_pacf_temporal.png", flush=True)
        lags = min(30, max(1, len(daily) // 2 - 1))
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        plot_acf(daily, ax=axes[0], lags=lags, title="Autocorrelacion (ACF)")
        plot_pacf(daily, ax=axes[1], lags=lags, title="Autocorrelacion parcial (PACF)")
        plt.tight_layout()
        plt.savefig(plots_dir / "acf_pacf_temporal.png")
        plt.close()
        payload["acf_pacf_temporal"] = "generada"
    else:
        payload["acf_pacf_temporal"] = "no_generada: se requieren al menos 10 dias"

    (results_dir / "analisis_temporal.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_plots(df: pd.DataFrame, plots_dir: Path, results_dir: Path, seed: int) -> None:
    """Genera los graficos requeridos por el enunciado.

    Pasos:
    1. Crea carpeta `plots/`.
    2. Toma muestra reproducible para graficos pesados.
    3. Genera histogramas con densidad de monto y descuento.
    4. Genera boxplot de monto por canal.
    5. Genera heatmap de correlacion Spearman.
    6. Agrupa ventas por dia y genera serie temporal.
    7. Llama `save_temporal_analysis` para descomposicion y ACF/PACF.
    8. Genera barras de transacciones por canal.

    Incluye histogramas con densidad, boxplot por canal, matriz de correlacion,
    serie temporal diaria y conteo por canal. Las figuras se guardan como PNG
    para poder incluirlas en el informe.
    """
    sns.set_theme(style="whitegrid")
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(" - preparando muestra para graficos", flush=True)
    plot_sample = sample_df(df, 100_000, seed)

    # Histogramas con KDE para observar forma, concentracion y colas.
    for col, filename in [
        ("MONTO APLICADO", "histograma_monto_aplicado.png"),
        ("PORCENTAJE DESCUENTO", "histograma_descuento.png"),
    ]:
        print(f" - generando {filename}", flush=True)
        plt.figure(figsize=(9, 5))
        sns.histplot(plot_sample[col].dropna(), kde=True, stat="density", bins=40)
        plt.title(f"Distribucion de {col}")
        plt.xlabel(col)
        plt.ylabel("Densidad")
        plt.tight_layout()
        plt.savefig(plots_dir / filename)
        plt.close()

    # Boxplot por canal: se ocultan fliers extremos para que la comparacion de
    # medianas y rango intercuartil sea visible.
    if {"CANAL", "MONTO APLICADO"}.issubset(df.columns):
        print(" - generando boxplot_monto_por_canal.png", flush=True)
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=sample_df(df[["CANAL", "MONTO APLICADO"]].dropna(), 80_000, seed), x="CANAL", y="MONTO APLICADO", showfliers=False)
        plt.title("MONTO APLICADO por CANAL")
        plt.tight_layout()
        plt.savefig(plots_dir / "boxplot_monto_por_canal.png")
        plt.close()

    # La misma correlacion guardada en CSV se representa como heatmap.
    print(" - generando matriz_correlacion.png", flush=True)
    corr = df[[c for c in NUMERIC_COLUMNS if c in df.columns]].corr(method="spearman")
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Matriz de correlacion Spearman")
    plt.tight_layout()
    plt.savefig(plots_dir / "matriz_correlacion.png")
    plt.close()

    # Serie diaria: se agregan todas las transacciones por dia para detectar
    # tendencia, estacionalidad y autocorrelacion.
    print(" - generando serie_temporal_ventas.png", flush=True)
    daily = df.dropna(subset=["FECHA"]).groupby(df["FECHA"].dt.floor("D"))["MONTO APLICADO"].sum()
    if not daily.empty:
        plt.figure(figsize=(11, 5))
        daily = daily.sort_index()
        if len(daily) <= 30:
            plt.bar(daily.index.astype(str), daily.values)
            plt.xticks(rotation=45, ha="right")
        else:
            daily.plot(marker="o", linewidth=1.5, markersize=3)
        plt.title("Ventas diarias")
        plt.xlabel("Fecha")
        plt.ylabel("Monto aplicado")
        plt.tight_layout()
        plt.savefig(plots_dir / "serie_temporal_ventas.png")
        plt.close()
        save_temporal_analysis(daily, plots_dir, results_dir)

    if "CANAL" in df.columns:
        print(" - generando ventas_por_canal.png", flush=True)
        plt.figure(figsize=(8, 5))
        df["CANAL"].value_counts().plot(kind="bar")
        plt.title("Transacciones por canal")
        plt.xlabel("Canal")
        plt.ylabel("Cantidad")
        plt.tight_layout()
        plt.savefig(plots_dir / "ventas_por_canal.png")
        plt.close()


def run_parallel_tables(df: pd.DataFrame, results_dir: Path, seed: int, workers: int) -> None:
    """Ejecuta calculos estadisticos independientes en paralelo.

    Pasos:
    1. Define tareas independientes: descriptivas, normalidad, correlaciones y
       pruebas exploratorias.
    2. Si `workers <= 1`, ejecuta secuencialmente.
    3. Si hay paralelismo, limita workers al numero de tareas disponibles.
    4. Envia cada tarea al `ThreadPoolExecutor`.
    5. Espera cada future, propaga errores y reporta tarea terminada.

    Se usa `ThreadPoolExecutor` para compartir el DataFrame en memoria. Usar
    procesos obligaria a copiar o serializar millones de filas para cada tarea,
    lo que puede ser mas lento y consumir mucha RAM. Los graficos se mantienen
    fuera de esta funcion porque Matplotlib no es seguro para dibujar desde
    varios hilos a la vez.
    """
    tasks = {
        "estadisticas_descriptivas": lambda: save_descriptive_stats(df, results_dir),
        "normalidad": lambda: save_normality_tests(df, results_dir, seed),
        "correlaciones": lambda: save_correlations(df, results_dir, seed),
        "pruebas_exploratorias": lambda: save_association_tests(df, results_dir),
    }

    if workers <= 1:
        for name, task in tasks.items():
            print(f" - ejecutando {name}", flush=True)
            task()
        return

    max_workers = min(workers, len(tasks))
    print(f"Ejecutando calculos EDA en paralelo con {max_workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for name, task in tasks.items():
            print(f" - enviando {name} al executor", flush=True)
            futures[executor.submit(task)] = name
        for future in as_completed(futures):
            name = futures[future]
            future.result()
            print(f" - {name} listo", flush=True)


def parse_args() -> argparse.Namespace:
    """Define argumentos CLI para entrada, carpetas de salida y workers.

    Pasos:
    1. Registra dataset limpio de entrada.
    2. Registra carpeta de plots.
    3. Registra carpeta de resultados JSON.
    4. Registra cantidad de workers para calculos tabulares.
    5. Devuelve argumentos parseados.
    """
    parser = argparse.ArgumentParser(description="Genera analisis exploratorio y graficos.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


def main() -> None:
    """Ejecuta todo el analisis exploratorio.

    Pasos:
    1. Lee argumentos y semilla reproducible.
    2. Crea carpetas de resultados y graficos.
    3. Carga el dataset limpio.
    4. Ejecuta calculos tabulares en paralelo parcial.
    5. Genera graficos en modo secuencial por seguridad de Matplotlib.
    6. Imprime rutas finales.
    """
    args = parse_args()
    seed = get_seed()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("Analisis exploratorio", flush=True)
    print(f"Entrada: {args.input}", flush=True)
    df = load_data(args.input)
    print(f"Filas cargadas: {len(df)}", flush=True)

    run_parallel_tables(df, args.results_dir, seed, args.workers)
    print("Generando graficos EDA en modo secuencial", flush=True)
    save_plots(df, args.plots_dir, args.results_dir, seed)
    print(f"Resultados guardados en: {args.results_dir}", flush=True)
    print(f"Graficos guardados en: {args.plots_dir}", flush=True)


if __name__ == "__main__":
    main()
