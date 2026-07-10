"""Inferencia estadistica y modelado predictivo.

Este modulo usa el dataset limpio para validar hipotesis del negocio y entrenar
un modelo predictivo para `MONTO APLICADO`. Las salidas se guardan en
`resultados/` y los graficos de diagnostico en `plots/`.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]

# Matplotlib necesita una carpeta escribible para cache/configuracion. La
# dejamos dentro del proyecto para que funcione en PowerShell y entornos
# restringidos.
MPL_CACHE = ROOT / "resultados" / ".matplotlib_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

DEFAULT_INPUT = ROOT / "data" / "ventas_limpias.parquet"
DEFAULT_PLOTS_DIR = ROOT / "plots"
DEFAULT_RESULTS_DIR = ROOT / "resultados"
DEFAULT_MODEL_SAMPLE = 500_000
DEFAULT_LINEAR_SAMPLE = 200_000
DEFAULT_WORKERS = 4
PLOT_LOCK = Lock()


def get_seed() -> int:
    """Lee la semilla reproducible desde `CPYD_SEED`.

    Pasos:
    1. Lee `CPYD_SEED` desde el entorno.
    2. Si no existe, usa `42`.
    3. Devuelve el valor como entero.

    Se usa para la particion train/test del modelo, asegurando que las metricas
    sean comparables entre ejecuciones.
    """
    return int(os.environ.get("CPYD_SEED", "42"))


def verdict(p_value: float, alpha: float = 0.05) -> str:
    """Traduce un p-value a una decision estadistica simple.

    Pasos:
    1. Si el p-value no es evaluable (`NaN`), retorna `No evaluable`.
    2. Si `p_value < alpha`, retorna `Se rechaza H0`.
    3. En caso contrario, retorna `No se rechaza H0`.

    Centralizar esta regla evita repetir el criterio de significancia en cada
    hipotesis. Por defecto se usa alpha = 0.05.
    """
    if pd.isna(p_value):
        return "No evaluable"
    return "Se rechaza H0" if p_value < alpha else "No se rechaza H0"


def load_data(path: Path) -> pd.DataFrame:
    """Carga ventas limpias desde Parquet o CSV y normaliza tipos principales.

    Pasos:
    1. Lee Parquet o CSV segun extension.
    2. Limpia nombres de columnas.
    3. Convierte `FECHA` a datetime.
    4. Convierte predictores numericos y objetivo con `to_numeric`.

    Aunque `preprocesamiento.py` ya convirtio tipos, al leer desde disco
    conviene volver a asegurar fechas y numeros para evitar comparaciones como
    texto si el origen es CSV.
    """
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [col.strip() for col in df.columns]
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    for col in [
        "SKU",
        "UNIDADES",
        "PORCENTAJE DESCUENTO",
        "MONTO APLICADO",
        "MONTO POR UNIDAD",
        "EDAD",
        "FRECUENCIA COMPRA",
        "LOCAL",
        "GENERO",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def run_hypothesis_tests(df: pd.DataFrame, results_dir: Path) -> None:
    """Ejecuta hipotesis requeridas y guarda interpretaciones.

    Pasos:
    1. Evalua APP vs WEB con Welch unilateral y Mann-Whitney unilateral cuando
       existen ambos canales.
    2. Evalua descuento vs unidades con regresion simple si `UNIDADES` tiene
       variacion.
    3. Evalua monto por canal con ANOVA y Kruskal-Wallis.
    4. Evalua descuento vs monto con Spearman.
    5. Evalua edad vs monto con Spearman.
    6. Evalua monto por genero con Welch y Mann-Whitney.
    7. Para cada hipotesis, agrega decision usando `verdict`.
    8. Guarda `pruebas_hipotesis.json`.

    Se incluyen hipotesis sobre canal, descuento, edad y genero. Cuando las
    variables no tienen variacion suficiente, la prueba se marca como no
    evaluable en vez de forzar un resultado invalido.
    """
    hypotheses = []

    h_app_web = {
        "nombre": "H0_APP_WEB",
        "pregunta": "El ticket promedio en APP es mayor que en WEB.",
        "h0": "Ticket promedio APP <= ticket promedio WEB.",
        "h1": "Ticket promedio APP > ticket promedio WEB.",
        "resultados": [],
        "tipo": "hipotesis del enunciado evaluada segun disponibilidad de datos",
    }
    if "CANAL" in df.columns:
        app = df.loc[df["CANAL"] == "APP", "MONTO APLICADO"].dropna()
        web = df.loc[df["CANAL"] == "WEB", "MONTO APLICADO"].dropna()
        if len(app) > 1 and len(web) > 1:
            t_stat, p_two = stats.ttest_ind(app, web, equal_var=False)
            p_one = p_two / 2 if t_stat > 0 else 1 - (p_two / 2)
            u_stat, p_u = stats.mannwhitneyu(app, web, alternative="greater")
            h_app_web["resultados"].append(
                {"prueba": "Welch t-test unilateral", "estadistico": t_stat, "p_value": p_one, "decision": verdict(p_one)}
            )
            h_app_web["resultados"].append(
                {"prueba": "Mann-Whitney unilateral", "estadistico": u_stat, "p_value": p_u, "decision": verdict(p_u)}
            )
        else:
            h_app_web["nota"] = "No evaluable: no existen suficientes registros en APP y WEB."
            h_app_web["canales_disponibles"] = sorted(str(x) for x in df["CANAL"].dropna().unique())
    else:
        h_app_web["nota"] = "No evaluable: falta la columna CANAL."
    hypotheses.append(h_app_web)

    h_discount_units = {
        "nombre": "H0_DESCUENTO_UNIDADES",
        "pregunta": "El porcentaje de descuento afecta significativamente las unidades vendidas.",
        "h0": "El coeficiente de descuento sobre UNIDADES es igual a 0.",
        "h1": "El coeficiente de descuento sobre UNIDADES es distinto de 0.",
        "resultados": [],
        "tipo": "hipotesis del enunciado evaluada segun variabilidad de datos",
    }
    if {"PORCENTAJE DESCUENTO", "UNIDADES"}.issubset(df.columns):
        pair = df[["PORCENTAJE DESCUENTO", "UNIDADES"]].dropna()
        if len(pair) > 2 and pair["PORCENTAJE DESCUENTO"].std() > 0 and pair["UNIDADES"].std() > 0:
            X = sm.add_constant(pair["PORCENTAJE DESCUENTO"].astype(float))
            y = pair["UNIDADES"].astype(float)
            model = sm.OLS(y, X).fit()
            h_discount_units["resultados"].append(
                {
                    "prueba": "Regresion lineal simple",
                    "coeficiente_descuento": model.params.get("PORCENTAJE DESCUENTO"),
                    "p_value": model.pvalues.get("PORCENTAJE DESCUENTO"),
                    "r2": model.rsquared,
                    "decision": verdict(model.pvalues.get("PORCENTAJE DESCUENTO")),
                }
            )
        else:
            h_discount_units["nota"] = "No evaluable: UNIDADES no presenta variacion suficiente en los datos limpios."
            h_discount_units["unidades_unicas"] = sorted(float(x) for x in pair["UNIDADES"].dropna().unique())
    else:
        h_discount_units["nota"] = "No evaluable: faltan columnas requeridas."
    hypotheses.append(h_discount_units)

    h1 = {
        "nombre": "H1",
        "pregunta": "El monto promedio difiere entre canales de compra.",
        "h0": "MONTO APLICADO no difiere entre canales.",
        "h1": "Al menos un canal presenta diferencias.",
        "resultados": [],
    }
    # Para comparar montos por canal se reporta ANOVA y Kruskal-Wallis. Kruskal
    # es una alternativa no parametrica util cuando la normalidad es dudosa.
    groups = [
        group["MONTO APLICADO"].dropna().to_numpy()
        for _, group in df.groupby("CANAL")
        if len(group["MONTO APLICADO"].dropna()) > 1
    ]
    if len(groups) > 1:
        f_stat, p_anova = stats.f_oneway(*groups)
        h_stat, p_kw = stats.kruskal(*groups)
        h1["resultados"].append({"prueba": "ANOVA", "estadistico": f_stat, "p_value": p_anova, "decision": verdict(p_anova)})
        h1["resultados"].append({"prueba": "Kruskal-Wallis", "estadistico": h_stat, "p_value": p_kw, "decision": verdict(p_kw)})
    else:
        h1["nota"] = "No evaluable: se requieren al menos dos canales con datos suficientes."
    hypotheses.append(h1)

    h2 = {
        "nombre": "H2",
        "pregunta": "El porcentaje de descuento se asocia con el monto comprado.",
        "h0": "No existe asociacion monotona entre descuento y monto.",
        "h1": "Existe asociacion monotona entre descuento y monto.",
        "nota": "Se usa MONTO APLICADO porque UNIDADES no presenta variacion en los datos limpios.",
        "resultados": [],
    }
    # Spearman mide asociacion monotonica sin asumir distribucion normal.
    pair = df[["PORCENTAJE DESCUENTO", "MONTO APLICADO"]].dropna()
    if len(pair) > 2 and pair["PORCENTAJE DESCUENTO"].std() > 0 and pair["MONTO APLICADO"].std() > 0:
        rho, p_value = stats.spearmanr(pair["PORCENTAJE DESCUENTO"], pair["MONTO APLICADO"])
        h2["resultados"].append({"prueba": "Spearman", "rho": rho, "p_value": p_value, "decision": verdict(p_value)})
    else:
        h2["nota"] = "No evaluable: variables sin variacion suficiente."
    hypotheses.append(h2)

    h3 = {
        "nombre": "H3",
        "pregunta": "La edad del cliente se asocia con el monto comprado.",
        "h0": "No existe asociacion monotona entre edad y monto.",
        "h1": "Existe asociacion monotona entre edad y monto.",
        "resultados": [],
    }
    pair = df[["EDAD", "MONTO APLICADO"]].dropna()
    if len(pair) > 2 and pair["EDAD"].std() > 0 and pair["MONTO APLICADO"].std() > 0:
        rho, p_value = stats.spearmanr(pair["EDAD"], pair["MONTO APLICADO"])
        h3["resultados"].append({"prueba": "Spearman", "rho": rho, "p_value": p_value, "decision": verdict(p_value)})
    else:
        h3["nota"] = "No evaluable: variables sin variacion suficiente."
    hypotheses.append(h3)

    h4 = {
        "nombre": "H4",
        "pregunta": "El monto promedio difiere segun genero.",
        "h0": "MONTO APLICADO no difiere segun genero.",
        "h1": "MONTO APLICADO difiere segun genero.",
        "resultados": [],
    }
    # Welch no asume varianzas iguales; Mann-Whitney complementa con una prueba
    # no parametrica para dos grupos.
    gender_values = sorted(df["GENERO"].dropna().unique()) if "GENERO" in df.columns else []
    if len(gender_values) >= 2:
        a = df.loc[df["GENERO"] == gender_values[0], "MONTO APLICADO"].dropna()
        b = df.loc[df["GENERO"] == gender_values[1], "MONTO APLICADO"].dropna()
        if len(a) > 1 and len(b) > 1:
            t_stat, p_t = stats.ttest_ind(a, b, equal_var=False)
            u_stat, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
            h4["resultados"].append({"prueba": "Welch t-test", "estadistico": t_stat, "p_value": p_t, "decision": verdict(p_t)})
            h4["resultados"].append({"prueba": "Mann-Whitney", "estadistico": u_stat, "p_value": p_u, "decision": verdict(p_u)})
        else:
            h4["nota"] = "No evaluable: se requieren dos grupos con datos suficientes."
    else:
        h4["nota"] = "No evaluable: se requieren al menos dos generos."
    hypotheses.append(h4)

    (results_dir / "pruebas_hipotesis.json").write_text(
        json.dumps({"hipotesis": hypotheses}, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara variables numericas y categoricas para el modelo.

    Pasos:
    1. Copia el dataset limpio para no modificar el original.
    2. Deriva mes, dia de semana y hora desde `FECHA`.
    3. Selecciona variables requeridas y opcionales disponibles.
    4. Elimina filas incompletas para entrenamiento.
    5. Codifica `CANAL` con dummies y `drop_first=True`.
    6. Devuelve una tabla lista para separar `X` e `y`.

    Selecciona predictores disponibles, elimina filas incompletas, agrega
    variables temporales y transforma `CANAL` en dummies. Se incluye `SKU` para
    capturar diferencias por producto y `LOCAL` para diferencias por sucursal.
    """
    model_df = df.copy()
    if "FECHA" in model_df.columns:
        model_df["FECHA_MES"] = model_df["FECHA"].dt.month
        model_df["FECHA_DIA_SEMANA"] = model_df["FECHA"].dt.dayofweek
        model_df["FECHA_HORA"] = model_df["FECHA"].dt.hour

    required = ["MONTO APLICADO", "UNIDADES", "PORCENTAJE DESCUENTO", "EDAD", "FRECUENCIA COMPRA"]
    optional = ["SKU", "CANAL", "GENERO", "LOCAL", "FECHA_MES", "FECHA_DIA_SEMANA", "FECHA_HORA"]
    cols = [c for c in required + optional if c in df.columns]
    cols = [c for c in required + optional if c in model_df.columns]
    model_df = model_df[cols].dropna().copy()
    if "CANAL" in model_df.columns:
        model_df = pd.get_dummies(model_df, columns=["CANAL"], drop_first=True)
    return model_df


def run_regression(
    df: pd.DataFrame,
    plots_dir: Path,
    results_dir: Path,
    seed: int,
    max_rows: int = DEFAULT_MODEL_SAMPLE,
) -> None:
    """Entrena un modelo no lineal y guarda metricas/graficos.

    Pasos:
    1. Construye matriz de modelado con `build_model_frame`.
    2. Si hay muy pocas filas o no hay predictores, guarda estado
       `no_entrenado`.
    3. Si el dataset supera `max_rows`, toma muestra reproducible.
    4. Separa predictores `X` y objetivo `MONTO APLICADO`.
    5. Divide train/test con semilla fija.
    6. Entrena `HistGradientBoostingRegressor` sobre `log1p(y)`.
    7. Invierte predicciones con `expm1` y recorta valores negativos.
    8. Calcula MAE, RMSE y R2.
    9. Guarda `metricas_modelo.json`.
    10. Genera graficos real vs predicho y residuos protegidos con `PLOT_LOCK`.

    Se usa `HistGradientBoostingRegressor` sobre `log1p(MONTO APLICADO)`. Esta
    alternativa captura relaciones no lineales y reduce el efecto de la cola
    larga de montos, manteniendo un entrenamiento razonable para datos grandes.
    """
    model_df = build_model_frame(df)
    metrics_path = results_dir / "metricas_modelo.json"

    if len(model_df) < 4:
        metrics_path.write_text(
            json.dumps(
                {
                    "modelo": "HistGradientBoostingRegressor",
                    "estado": "no_entrenado",
                    "motivo": "Se requieren al menos 4 filas limpias.",
                    "filas_disponibles": int(len(model_df)),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return

    if max_rows and len(model_df) > max_rows:
        model_df = model_df.sample(max_rows, random_state=seed)

    # `X` contiene predictores y `y` la variable objetivo exigida por la opcion
    # de regresion del enunciado.
    X = model_df.drop(columns=["MONTO APLICADO"]).astype(float)
    y = model_df["MONTO APLICADO"].astype(float)

    if X.shape[1] == 0:
        metrics_path.write_text(
            json.dumps(
                {
                    "modelo": "HistGradientBoostingRegressor",
                    "estado": "no_entrenado",
                    "motivo": "No hay predictores disponibles.",
                    "filas_disponibles": int(len(model_df)),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return

    # Con datasets normales usamos 70/30. Para datos de prueba muy pequenos se
    # usa 50/50 para asegurar observaciones en ambos conjuntos.
    test_size = 0.3 if len(model_df) >= 10 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    y_train_log = np.log1p(y_train)
    model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=seed,
    )
    model.fit(X_train, y_train_log)
    y_pred = np.expm1(model.predict(X_test))
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan

    payload = {
        "modelo": "HistGradientBoostingRegressor",
        "objetivo": "MONTO APLICADO",
        "transformacion_objetivo": "log1p",
        "muestra_maxima": max_rows,
        "filas_usadas": int(len(model_df)),
        "train": int(len(X_train)),
        "test": int(len(X_test)),
        "metricas": {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": None if pd.isna(r2) else r2,
        },
        "predictores": list(X.columns),
    }
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )

    # Grafico real vs predicho: si el modelo fuera perfecto, los puntos caerian
    # cerca de la diagonal roja.
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_n = min(50_000, len(y_test))
    plot_index = y_test.sample(plot_n, random_state=seed).index
    y_test_plot = y_test.loc[plot_index]
    y_pred_plot = pd.Series(y_pred, index=y_test.index).loc[plot_index]
    residuals = y_test_plot.to_numpy() - y_pred_plot.to_numpy()

    # Matplotlib usa estado global; el lock permite entrenar/calcular en
    # paralelo y serializar solo la escritura de figuras.
    with PLOT_LOCK:
        plt.figure(figsize=(7, 6))
        plt.scatter(y_test_plot, y_pred_plot, alpha=0.25, s=8)
        min_v = min(y_test_plot.min(), y_pred_plot.min())
        max_v = max(y_test_plot.max(), y_pred_plot.max())
        plt.plot([min_v, max_v], [min_v, max_v], color="red", linestyle="--")
        plt.title("Modelo: real vs predicho")
        plt.xlabel("Monto real")
        plt.ylabel("Monto predicho")
        plt.tight_layout()
        plt.savefig(plots_dir / "modelo_real_vs_predicho.png")
        plt.close()

        # Grafico de residuos: ayuda a detectar sesgos, heterocedasticidad o
        # patrones no capturados por el modelo lineal.
        plt.figure(figsize=(7, 5))
        plt.scatter(y_pred_plot, residuals, alpha=0.25, s=8)
        plt.axhline(0, color="red", linestyle="--")
        plt.title("Residuos del modelo")
        plt.xlabel("Monto predicho")
        plt.ylabel("Residuo")
        plt.tight_layout()
        plt.savefig(plots_dir / "modelo_residuos.png")
        plt.close()


def run_linear_diagnostics(
    df: pd.DataFrame,
    plots_dir: Path,
    results_dir: Path,
    seed: int,
    max_rows: int = DEFAULT_LINEAR_SAMPLE,
) -> None:
    """Ajusta una regresion lineal interpretable y guarda diagnosticos.

    Pasos:
    1. Selecciona variables de la opcion de regresion del enunciado.
    2. Elimina filas incompletas y toma muestra reproducible si corresponde.
    3. Transforma el objetivo con `log1p`.
    4. Codifica `CANAL` con dummies.
    5. Elimina predictores sin variacion.
    6. Divide train/test.
    7. Ajusta OLS con constante.
    8. Calcula predicciones en escala original para metricas.
    9. Calcula VIF, Jarque-Bera, Shapiro y Breusch-Pagan.
    10. Guarda coeficientes, diagnosticos y limitaciones de extrapolabilidad.
    11. Genera graficos de residuos y Q-Q con `PLOT_LOCK`.

    Complementa el modelo no lineal con una opcion tipo Opcion A de la rubrica:
    coeficientes, R2 ajustado, VIF, normalidad de residuos y homocedasticidad.
    Se modela `log1p(MONTO APLICADO)` para reducir asimetria extrema.
    """
    diag_path = results_dir / "diagnostico_regresion_lineal.json"
    cols = ["MONTO APLICADO", "CANAL", "LOCAL", "UNIDADES", "PORCENTAJE DESCUENTO"]
    available = [col for col in cols if col in df.columns]
    linear_df = df[available].dropna().copy()

    if len(linear_df) < 10:
        diag_path.write_text(
            json.dumps({"estado": "no_entrenado", "motivo": "Se requieren al menos 10 filas limpias."}, indent=2),
            encoding="utf-8",
        )
        return

    if max_rows and len(linear_df) > max_rows:
        linear_df = linear_df.sample(max_rows, random_state=seed)

    y = np.log1p(linear_df["MONTO APLICADO"].astype(float))
    feature_df = linear_df.drop(columns=["MONTO APLICADO"]).copy()
    if "CANAL" in feature_df.columns:
        feature_df = pd.get_dummies(feature_df, columns=["CANAL"], drop_first=True)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    feature_df = feature_df.loc[:, feature_df.nunique(dropna=True) > 1]
    feature_df = feature_df.astype(float)

    if feature_df.empty:
        diag_path.write_text(
            json.dumps({"estado": "no_entrenado", "motivo": "No hay predictores con variacion suficiente."}, indent=2),
            encoding="utf-8",
        )
        return

    X_train, X_test, y_train, y_test = train_test_split(feature_df, y, test_size=0.3, random_state=seed)
    X_train_const = sm.add_constant(X_train, has_constant="add")
    X_test_const = sm.add_constant(X_test, has_constant="add")
    model = sm.OLS(y_train, X_train_const).fit()
    y_pred_log = model.predict(X_test_const)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    y_test_original = np.expm1(y_test)
    residuals = model.resid

    vif_rows = []
    vif_frame = X_train_const.drop(columns=["const"], errors="ignore")
    for idx, col in enumerate(vif_frame.columns):
        try:
            vif = variance_inflation_factor(vif_frame.to_numpy(), idx)
        except Exception:
            vif = np.nan
        vif_rows.append({"variable": col, "vif": None if pd.isna(vif) else float(vif)})

    bp_stat, bp_pvalue, f_stat, f_pvalue = het_breuschpagan(residuals, model.model.exog)
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)
    shapiro_sample = pd.Series(residuals).sample(min(len(residuals), 5000), random_state=seed)
    shapiro_stat, shapiro_p = stats.shapiro(shapiro_sample)

    coefficients = []
    for name in model.params.index:
        coefficients.append(
            {
                "variable": name,
                "coeficiente": float(model.params[name]),
                "p_value": float(model.pvalues[name]),
                "intervalo_95": [float(model.conf_int().loc[name, 0]), float(model.conf_int().loc[name, 1])],
            }
        )

    payload = {
        "modelo": "OLS",
        "objetivo": "log1p(MONTO APLICADO)",
        "predictores_base": ["CANAL", "LOCAL", "UNIDADES", "PORCENTAJE DESCUENTO"],
        "nota": "UNIDADES se excluye automaticamente si no presenta variacion. CANAL se codifica con dummies.",
        "filas_usadas": int(len(linear_df)),
        "train": int(len(X_train)),
        "test": int(len(X_test)),
        "r2_train": float(model.rsquared),
        "r2_ajustado_train": float(model.rsquared_adj),
        "metricas_test_escala_original": {
            "MAE": float(mean_absolute_error(y_test_original, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test_original, y_pred))),
            "R2": float(r2_score(y_test_original, y_pred)),
        },
        "coeficientes": coefficients,
        "vif": vif_rows,
        "diagnosticos_residuos": {
            "normalidad_jarque_bera": {"estadistico": float(jb_stat), "p_value": float(jb_pvalue)},
            "normalidad_shapiro_muestra": {"estadistico": float(shapiro_stat), "p_value": float(shapiro_p)},
            "homocedasticidad_breusch_pagan": {
                "lm_stat": float(bp_stat),
                "lm_p_value": float(bp_pvalue),
                "f_stat": float(f_stat),
                "f_p_value": float(f_pvalue),
            },
        },
        "extrapolabilidad": {
            "conclusion": "Limitada fuera de los rangos observados.",
            "limitaciones": [
                "No extrapola de forma confiable a canales o locales no observados.",
                "UNIDADES no aporta informacion si permanece constante.",
                "Faltan variables externas como stock, campanas, feriados y promociones.",
                "La muestra usada controla tiempo de ejecucion; puede subrepresentar casos raros.",
            ],
        },
    }
    diag_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_n = min(50_000, len(y_test_original))
    plot_index = y_test_original.sample(plot_n, random_state=seed).index
    y_plot = y_test_original.loc[plot_index]
    pred_plot = pd.Series(y_pred, index=y_test.index).loc[plot_index]
    residual_plot = y_plot.to_numpy() - pred_plot.to_numpy()

    with PLOT_LOCK:
        plt.figure(figsize=(7, 5))
        plt.scatter(pred_plot, residual_plot, alpha=0.25, s=8)
        plt.axhline(0, color="red", linestyle="--")
        plt.title("Regresion lineal: residuos")
        plt.xlabel("Monto predicho")
        plt.ylabel("Residuo")
        plt.tight_layout()
        plt.savefig(plots_dir / "regresion_lineal_residuos.png")
        plt.close()

        plt.figure(figsize=(7, 5))
        stats.probplot(pd.Series(residuals).sample(min(len(residuals), 5000), random_state=seed), dist="norm", plot=plt)
        plt.title("Regresion lineal: Q-Q residuos")
        plt.tight_layout()
        plt.savefig(plots_dir / "regresion_lineal_qq_residuos.png")
        plt.close()


def parse_args() -> argparse.Namespace:
    """Define argumentos CLI de entrada, salidas, muestra y workers.

    Pasos:
    1. Registra dataset limpio de entrada.
    2. Registra carpetas de plots y resultados.
    3. Registra muestra maxima para modelo no lineal.
    4. Registra cantidad de workers para tareas de inferencia/modelado.
    5. Devuelve argumentos parseados.
    """
    parser = argparse.ArgumentParser(description="Ejecuta hipotesis y modelado predictivo.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model-sample", type=int, default=DEFAULT_MODEL_SAMPLE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


def run_parallel_inference(
    df: pd.DataFrame,
    plots_dir: Path,
    results_dir: Path,
    seed: int,
    model_sample: int,
    workers: int,
) -> None:
    """Ejecuta hipotesis, modelo predictivo y diagnostico lineal en paralelo.

    Pasos:
    1. Define tres tareas independientes: hipotesis, modelo no lineal y OLS.
    2. Si `workers <= 1`, ejecuta todo en secuencia.
    3. Si hay paralelismo, limita workers al numero de tareas.
    4. Envia cada tarea a un `ThreadPoolExecutor`.
    5. Espera cada future, propaga errores y reporta avance.

    El paralelismo se aplica por tarea completa, no por filas. Esto evita
    dividir calculos estadisticos que necesitan ver el dataset completo y evita
    copiar el DataFrame a procesos separados. Cada tarea escribe archivos
    distintos; los graficos se protegen internamente con `PLOT_LOCK`.
    """
    tasks = {
        "pruebas_hipotesis": lambda: run_hypothesis_tests(df, results_dir),
        "modelo_predictivo": lambda: run_regression(df, plots_dir, results_dir, seed, max_rows=model_sample),
        "diagnostico_lineal": lambda: run_linear_diagnostics(df, plots_dir, results_dir, seed),
    }

    if workers <= 1:
        for name, task in tasks.items():
            print(f" - ejecutando {name}", flush=True)
            task()
        return

    max_workers = min(workers, len(tasks))
    print(f"Ejecutando inferencia/modelado en paralelo con {max_workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for name, task in tasks.items():
            print(f" - enviando {name} al executor", flush=True)
            futures[executor.submit(task)] = name
        for future in as_completed(futures):
            name = futures[future]
            future.result()
            print(f" - {name} listo", flush=True)


def main() -> None:
    """Ejecuta pruebas de hipotesis y modelado predictivo.

    Pasos:
    1. Lee argumentos y semilla.
    2. Crea carpetas de salida.
    3. Carga dataset limpio.
    4. Ejecuta inferencia/modelado en paralelo parcial.
    5. Imprime ubicacion de JSON y graficos generados.
    """
    args = parse_args()
    seed = get_seed()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("Inferencia y modelado", flush=True)
    print(f"CPYD_SEED={seed}", flush=True)
    print(f"Entrada: {args.input}", flush=True)
    df = load_data(args.input)
    print(f"Filas cargadas: {len(df)}", flush=True)
    print(f"Cantidad de workers para inferencia/modelado: {args.workers}", flush=True)

    run_parallel_inference(
        df,
        args.plots_dir,
        args.results_dir,
        seed,
        args.model_sample,
        args.workers,
    )
    print(f"Resultados guardados en: {args.results_dir}", flush=True)
    print(f"Graficos guardados en: {args.plots_dir}", flush=True)


if __name__ == "__main__":
    main()
