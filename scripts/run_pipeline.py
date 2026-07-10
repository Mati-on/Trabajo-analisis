"""Orquesta el pipeline completo del proyecto Cruz Morada.

Flujo:
    1. validacion paralela por chunks hacia Parquet temporal;
    2. preprocesamiento y variables derivadas hacia Parquet limpio;
    3. analisis exploratorio y graficos;
    4. pruebas de hipotesis y modelado.
"""

import argparse
import ctypes
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_INPUT = ROOT / "data" / "ventas_completas.csv"

# Rutas de las etapas. El orquestador no implementa la logica de negocio; solo
# llama cada script en orden y detiene el proceso si alguna etapa falla.
PARALLEL = SCRIPTS / "procesar_paralelo.py"
PREPROC = SCRIPTS / "preprocesamiento.py"
EDA = SCRIPTS / "analisis_exploratorio.py"
MODEL = SCRIPTS / "inferencia_modelado.py"
DASHBOARD = SCRIPTS / "dashboard.py"


def disable_quick_edit_mode() -> None:
    """Evita pausas accidentales de PowerShell/ConHost por seleccion de texto.

    Pasos:
    1. Verifica que el sistema operativo sea Windows.
    2. Obtiene el handle de entrada de consola.
    3. Lee el modo actual.
    4. Activa `ENABLE_EXTENDED_FLAGS` y desactiva `ENABLE_QUICK_EDIT_MODE`.
    5. Si la consola no permite el cambio, continua sin interrumpir.

    En Windows, QuickEdit puede congelar una ejecucion de consola hasta que el
    usuario presiona Enter. No es un bloqueo del pipeline, sino del host de
    terminal. Si la llamada falla, se ignora para mantener compatibilidad con
    otros entornos.
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


def python_executable() -> str:
    """Usa el interprete actual para mantener el mismo entorno virtual.

    Pasos:
    1. Lee `sys.executable`.
    2. Devuelve esa ruta para ejecutar los scripts hijos con el mismo Python.

    Si el usuario activo `.venv`, `sys.executable` apunta a ese Python. Asi las
    etapas hijas usan las mismas dependencias instaladas.
    """
    return sys.executable


def run_step(label: str, command: list[str]) -> None:
    """Ejecuta una etapa del pipeline y detiene el flujo si falla.

    Pasos:
    1. Imprime separadores y nombre de etapa.
    2. Imprime el comando exacto que se ejecutara.
    3. Llama `subprocess.check_call` desde la raiz del proyecto.
    4. Si el script hijo falla, propaga la excepcion y detiene el pipeline.

    `subprocess.check_call` levanta una excepcion si el script devuelve error.
    Eso evita continuar con EDA o modelo cuando no se genero correctamente el
    dataset limpio.
    """
    print("", flush=True)
    print("=" * 80, flush=True)
    print(label, flush=True)
    print("=" * 80, flush=True)
    print(" ".join(command), flush=True)
    subprocess.check_call(command, cwd=ROOT)


def parse_args() -> argparse.Namespace:
    """Define parametros de ejecucion del pipeline completo.

    Pasos:
    1. Crea el parser de argumentos.
    2. Registra entrada, chunksize, workers, sample y executor.
    3. Registra banderas para saltar EDA/modelo, conservar validado temporal y
       no abrir dashboard.
    4. Devuelve el namespace parseado.

    Permite cambiar entrada, tamano de chunk, numero de workers y modo de
    ejecucion (`process` o `thread`) sin editar codigo.
    """
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline completo de ventas.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--executor", choices=["process", "thread"], default="process")
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefijo opcional para salidas generadas por una ejecucion de prueba.",
    )
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument(
        "--keep-valid-output",
        action="store_true",
        help="Conserva el dataset validado intermedio. Por defecto se elimina al terminar el preprocesamiento.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="No abre la ventana del dashboard al finalizar el pipeline.",
    )
    return parser.parse_args()


def output_paths(prefix: str, sample: int):
    """Define nombres de salida consistentes para ejecuciones reales o de prueba.

    Pasos:
    1. Si existe `prefix`, usa ese sufijo.
    2. Si no hay prefijo pero hay `sample`, usa `_prueba`.
    3. Si no hay ninguno, usa nombres finales sin sufijo.
    4. Construye rutas Parquet validada y limpia.

    - Sin prefijo ni muestra: genera `ventas_validas.parquet` y
      `ventas_limpias.parquet`.
    - Con `--sample`: usa sufijo `_prueba`.
    - Con `--prefix`: usa el sufijo indicado por el usuario.
    """
    if prefix:
        suffix = f"_{prefix}"
    elif sample and sample > 0:
        suffix = "_prueba"
    else:
        suffix = ""

    valid_path = ROOT / "data" / f"ventas_validas{suffix}.parquet"
    clean_path = ROOT / "data" / f"ventas_limpias{suffix}.parquet"
    return valid_path, clean_path


def main() -> None:
    """Orquesta validacion, preprocesamiento, EDA e inferencia/modelado.

    Pasos:
    1. Desactiva QuickEdit y lee argumentos.
    2. Fija `CPYD_SEED` y `PYTHONUNBUFFERED`.
    3. Calcula rutas de salida.
    4. Ejecuta validacion paralela.
    5. Ejecuta preprocesamiento y elimina el validado temporal si corresponde.
    6. Ejecuta EDA si no fue omitido.
    7. Ejecuta inferencia/modelado si no fue omitido.
    8. Abre dashboard automaticamente salvo que se use `--no-dashboard`.

    El flujo normal no conserva CSV intermedios: usa un Parquet validado
    temporal y deja como salida analitica `ventas_limpias.parquet`.
    """
    disable_quick_edit_mode()
    args = parse_args()

    # Si el usuario no definio semilla, fijamos 42 para cumplir el requisito de
    # determinismo del enunciado.
    seed = os.environ.get("CPYD_SEED", "42")
    os.environ["CPYD_SEED"] = seed
    os.environ["PYTHONUNBUFFERED"] = "1"

    py = python_executable()
    sample = args.sample if args.sample and args.sample > 0 else 0
    valid_path, clean_path = output_paths(args.prefix, sample)
    plots_dir = ROOT / "plots"
    results_dir = ROOT / "resultados"

    print("Pipeline Cruz Morada", flush=True)
    print(f"CPYD_SEED={seed}", flush=True)
    print(f"Entrada: {args.input}", flush=True)
    print(f"Dataset validado temporal: {valid_path}", flush=True)
    print(f"Dataset limpio: {clean_path}", flush=True)
    print(f"Cantidad de workers para validacion, EDA e inferencia/modelado: {args.workers}", flush=True)

    # Etapa 1: validacion paralela. Produce el dataset validado temporal que
    # alimenta el preprocesamiento.
    cmd_parallel = [
        py,
        str(PARALLEL),
        "--input",
        str(args.input),
        "--output",
        str(valid_path),
        "--chunksize",
        str(args.chunksize),
        "--workers",
        str(args.workers),
        "--executor",
        args.executor,
    ]
    if sample:
        cmd_parallel += ["--sample", str(sample)]
    run_step("1) Validacion paralela por chunks", cmd_parallel)

    # Etapa 2: limpieza y variables derivadas. Produce el dataset analitico.
    cmd_preproc = [
        py,
        str(PREPROC),
        "--input",
        str(valid_path),
        "--output",
        str(clean_path),
    ]
    run_step("2) Preprocesamiento", cmd_preproc)
    if not args.keep_valid_output:
        try:
            valid_path.unlink(missing_ok=True)
            print(f"Dataset validado temporal eliminado: {valid_path}", flush=True)
        except OSError as exc:
            print(f"No se pudo eliminar el dataset validado temporal: {exc}", flush=True)

    # Etapa 3: analisis exploratorio. Usa la misma cantidad de workers definida
    # por `--workers` en la validacion, aunque cada etapa crea sus propios
    # procesos o hilos. Se puede saltar durante depuracion si ya existen
    # graficos/resultados y solo se quiere probar el modelo.
    if not args.skip_eda:
        cmd_eda = [
            py,
            str(EDA),
            "--input",
            str(clean_path),
            "--plots-dir",
            str(plots_dir),
            "--results-dir",
            str(results_dir),
            "--workers",
            str(args.workers),
        ]
        run_step("3) Analisis exploratorio", cmd_eda)

    # Etapa 4: inferencia y modelo. Usa la misma cantidad configurada por
    # `--workers` para paralelizar pruebas de hipotesis, modelo predictivo y
    # diagnostico lineal por tareas independientes.
    if not args.skip_model:
        cmd_model = [
            py,
            str(MODEL),
            "--input",
            str(clean_path),
            "--plots-dir",
            str(plots_dir),
            "--results-dir",
            str(results_dir),
            "--workers",
            str(args.workers),
        ]
        run_step("4) Inferencia y modelado", cmd_model)

    print("", flush=True)
    print("Pipeline finalizado.", flush=True)
    print(f"Datos limpios: {clean_path}", flush=True)
    print(f"Graficos: {plots_dir}", flush=True)
    print(f"Resultados: {results_dir}", flush=True)

    if not args.no_dashboard:
        cmd_dashboard = [
            py,
            str(DASHBOARD),
            "--input",
            str(clean_path),
            "--logs",
            str(ROOT / "logs.txt"),
            "--plots-dir",
            str(plots_dir),
            "--results-dir",
            str(results_dir),
        ]
        try:
            subprocess.Popen(cmd_dashboard, cwd=ROOT)
            print("Dashboard abierto automaticamente.", flush=True)
        except OSError as exc:
            print(f"No se pudo abrir el dashboard automaticamente: {exc}", flush=True)
            print("Puedes abrirlo manualmente con:", flush=True)
            print(" ".join(cmd_dashboard), flush=True)


if __name__ == "__main__":
    main()
