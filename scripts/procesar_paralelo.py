"""Procesamiento paralelo por chunks para el CSV de ventas.

Lee `data/ventas_completas.csv` por bloques, valida cada bloque en paralelo y
escribe un dataset validado con las transacciones validas. Por defecto se usa
Parquet para evitar CSV intermedios pesados. El archivo completo no se carga en
memoria: solo se mantienen algunos chunks pendientes segun la cantidad de
workers.
"""

import argparse
import csv
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from validacion import validar_transaccion

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "ventas_completas.csv"
DEFAULT_OUTPUT = ROOT / "data" / "ventas_validas.parquet"
DEFAULT_LOG = ROOT / "logs.txt"

# Valores por defecto elegidos para equilibrar memoria y paralelismo. Un chunk
# de 100.000 filas evita cargar el CSV completo y mantiene bajo el overhead de
# enviar trabajo a los workers.
DEFAULT_CHUNKSIZE = 100_000
DEFAULT_WORKERS = 4


class OrderedOutputWriter:
    """Escribe filas validas en CSV o Parquet desde el proceso principal.

    Pasos:
    1. Recibe ruta de salida y encabezado.
    2. Abre CSV al entrar al contexto, si corresponde.
    3. Inicializa Parquet al recibir el primer lote, si corresponde.
    4. Escribe lotes ya ordenados desde el proceso principal.
    5. Cierra recursos al salir del contexto.

    Los workers nunca escriben al archivo compartido. Esta clase concentra la
    escritura y permite cambiar el formato sin modificar la validacion paralela.

    Uso dentro del flujo:
    1. El proceso principal crea una instancia con ruta de salida y encabezado.
    2. Si la salida es CSV, abre el archivo y escribe el encabezado al entrar al
       contexto.
    3. Si la salida es Parquet, espera hasta recibir el primer lote de filas
       para inferir el esquema.
    4. Cada llamada a `write_rows` agrega filas validas ya ordenadas.
    5. Al salir del contexto, cierra el writer correspondiente.
    """

    def __init__(self, output_path: Path, header: list[str]):
        """Guarda configuracion inicial del writer ordenado.

        Pasos:
        1. Recibe ruta de salida y encabezado del CSV original.
        2. Inicializa manejadores CSV en `None`.
        3. Inicializa writer y esquema Parquet en `None`.
        4. La apertura real ocurre en `__enter__` o en el primer lote Parquet.
        """
        self.output_path = output_path
        self.header = header
        self.csv_file = None
        self.csv_writer = None
        self.parquet_writer = None
        self.parquet_schema = None

    def __enter__(self):
        """Prepara el recurso de escritura segun la extension de salida.

        Pasos:
        1. Si la salida es Parquet, no abre nada todavia porque necesita el
           primer lote para inferir esquema.
        2. Si la salida es CSV, abre el archivo.
        3. Crea `csv.writer`.
        4. Escribe encabezado.
        5. Devuelve `self` para usarlo dentro del `with`.
        """
        if self.output_path.suffix.lower() == ".parquet":
            return self
        self.csv_file = self.output_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file, delimiter=";")
        self.csv_writer.writerow(self.header)
        return self

    def write_rows(self, rows: list[list[str]]) -> None:
        """Agrega filas al archivo de salida conservando el formato elegido.

        Pasos:
        1. Si el chunk no tiene filas validas, no escribe nada.
        2. Para Parquet, transforma las filas en un `DataFrame`, luego en
           `pyarrow.Table`.
        3. En el primer lote Parquet, fija el esquema para que todos los lotes
           posteriores tengan columnas consistentes.
        4. Para CSV, delega directamente en `csv.writer.writerows`.
        """
        if not rows:
            return

        if self.output_path.suffix.lower() == ".parquet":
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(pd.DataFrame(rows, columns=self.header), preserve_index=False)
            if self.parquet_writer is None:
                self.parquet_schema = table.schema
                self.parquet_writer = pq.ParquetWriter(self.output_path, self.parquet_schema)
            self.parquet_writer.write_table(table.cast(self.parquet_schema))
            return

        self.csv_writer.writerows(rows)

    def __exit__(self, exc_type, exc, tb):
        """Cierra el writer activo para asegurar que el archivo quede completo.

        Pasos:
        1. Si existe writer Parquet, lo cierra para escribir metadatos finales.
        2. Si existe archivo CSV abierto, lo cierra.
        3. No suprime excepciones del bloque `with`.
        """
        if self.parquet_writer is not None:
            self.parquet_writer.close()
        if self.csv_file is not None:
            self.csv_file.close()


def process_chunk(payload):
    """Valida un bloque de filas en un worker.

    Args:
        payload: tupla `(chunk_index, header, rows, first_line)`.

    Returns:
        Diccionario con conteos, filas validas y mensajes de log.

    Pasos:
    1. Recibe el indice del chunk, encabezado, filas y numero de linea inicial.
    2. Recorre las filas del bloque en memoria.
    3. Llama `validar_transaccion` para cada registro.
    4. Acumula filas validas en `valid_rows`.
    5. Acumula descartes en `log_rows` usando el numero de linea original.
    6. Cuenta motivos de descarte para el resumen por chunk.
    7. Devuelve todo al proceso principal.

    Esta funcion es deliberadamente independiente del archivo de salida. Cada
    worker solo procesa datos en memoria; el proceso principal escribe en disco
    para evitar escrituras concurrentes.
    """
    chunk_index, header, rows, first_line = payload
    valid_rows = []
    log_rows = []
    discard_reasons = Counter()

    for offset, row in enumerate(rows):
        line_number = first_line + offset
        is_valid, reason = validar_transaccion(row, header)
        if is_valid:
            valid_rows.append(row)
        else:
            log_rows.append(f"{line_number}:{reason}\n")
            reason_key = reason.split(":", 1)[0] if reason else "SIN_MOTIVO"
            discard_reasons[reason_key] += 1

    return {
        "chunk_index": chunk_index,
        "read": len(rows),
        "valid": len(valid_rows),
        "discarded": len(log_rows),
        "discard_reasons": dict(discard_reasons),
        "valid_rows": valid_rows,
        "log_rows": log_rows,
    }


def chunked_reader(reader, chunksize: int, sample: int | None = None):
    """Genera chunks consecutivos desde el lector CSV.

    La funcion produce bloques de hasta `chunksize` filas y conserva el numero
    de linea inicial del bloque. Ese numero se usa para que `logs.txt` indique
    la posicion real de cada transaccion descartada.

    Pasos:
    1. Lee filas una a una desde `csv.reader`.
    2. Guarda en `first_line` la linea real donde inicia cada chunk.
    3. Acumula filas hasta completar `chunksize`.
    4. Entrega `(chunk_index, chunk, first_line)` con `yield`.
    5. Si `sample` esta definido, se detiene al alcanzar esa cantidad de filas.
    6. Al final, entrega el ultimo chunk aunque venga incompleto.

    `sample` permite ejecutar pruebas rapidas con las primeras N filas del CSV.
    """
    chunk = []
    first_line = 1
    total = 0
    chunk_index = 0

    for row in reader:
        if not chunk:
            first_line = total + 1
        chunk.append(row)
        total += 1

        if len(chunk) == chunksize:
            yield chunk_index, chunk, first_line
            chunk_index += 1
            chunk = []

        if sample and total >= sample:
            break

    if chunk:
        yield chunk_index, chunk, first_line


def flush_ready(buffer, writer, log_file, next_to_write):
    """Escribe resultados que ya llegaron y respetan el orden original.

    Los chunks pueden terminar en distinto orden porque se procesan en paralelo.
    Para que el dataset resultante sea determinista, guardamos temporalmente los
    resultados en `buffer` y solo escribimos cuando corresponde el siguiente
    `chunk_index`.

    Pasos:
    1. Revisa si el chunk esperado (`next_to_write`) ya esta en `buffer`.
    2. Si esta, lo saca del buffer.
    3. Escribe filas validas mediante `OrderedOutputWriter`.
    4. Escribe logs de descarte en `logs.txt`.
    5. Imprime resumen del chunk.
    6. Avanza al siguiente indice esperado y repite hasta encontrar un hueco.
    """
    while next_to_write in buffer:
        result = buffer.pop(next_to_write)
        writer.write_rows(result["valid_rows"])
        log_file.writelines(result["log_rows"])
        edad_descartadas = result["discard_reasons"].get("EDAD", 0)
        print(
            f"Chunk {result['chunk_index'] + 1}: leidas={result['read']}, "
            f"validas={result['valid']}, descartadas={result['discarded']}, "
            f"descartadas_edad={edad_descartadas}",
            flush=True,
        )
        next_to_write += 1
    return next_to_write


def process_file(
    input_path: Path,
    output_path: Path,
    log_path: Path,
    chunksize: int,
    workers: int,
    sample: int | None = None,
    executor_type: str = "process",
):
    """Ejecuta la validacion completa por chunks y workers paralelos.

    Args:
        input_path: CSV original con todas las transacciones.
        output_path: salida con solo transacciones validas, CSV o Parquet segun extension.
        log_path: archivo donde se registran descartes.
        chunksize: tamano de cada bloque enviado a los workers.
        workers: numero de procesos o hilos.
        sample: limite opcional de filas para pruebas.
        executor_type: `process` para CPU/paralelismo real o `thread` como
            alternativa compatible si Windows presenta problemas con procesos.

    Returns:
        Resumen con filas leidas, validas, descartadas y tiempo total.

    Pasos:
    1. Inicializa contadores, buffer de chunks terminados y cola de futures.
    2. Crea o reinicia `logs.txt`.
    3. Abre el CSV original en streaming.
    4. Selecciona `ProcessPoolExecutor` o `ThreadPoolExecutor`.
    5. Envia chunks al executor sin superar `workers * 2` tareas pendientes.
    6. Cuando una tarea termina, acumula sus conteos y guarda su resultado en
       `buffer`.
    7. Llama `flush_ready` para escribir solo los chunks que ya pueden salir en
       orden.
    8. Drena las tareas pendientes al terminar la lectura.
    9. Devuelve un resumen auditable para consola e informe.
    """
    start = time.time()
    total_read = 0
    total_valid = 0
    total_discarded = 0
    next_to_write = 0
    buffer = {}
    pending = {}
    # Limitamos la cantidad de chunks pendientes para no acumular demasiadas
    # filas en memoria si los workers van mas lento que la lectura.
    max_pending = max(1, workers * 2)
    executor_cls = ThreadPoolExecutor if executor_type == "thread" else ProcessPoolExecutor

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "REGISTRO DE TRANSACCIONES DESCARTADAS\n"
        "==================================\n",
        encoding="utf-8",
    )

    with input_path.open("r", newline="", encoding="utf-8-sig") as fi, log_path.open(
        "a", encoding="utf-8"
    ) as log_file:
        reader = csv.reader(fi, delimiter=";")
        header = next(reader, None)
        if header is None:
            raise ValueError("El CSV de entrada esta vacio")
        print("Iniciando lectura y envio de chunks...", flush=True)

        with OrderedOutputWriter(output_path, header) as writer, executor_cls(max_workers=workers) as executor:
            for chunk_index, rows, first_line in chunked_reader(reader, chunksize, sample):
                future = executor.submit(process_chunk, (chunk_index, header, rows, first_line))
                pending[future] = chunk_index
                print(
                    f"Chunk {chunk_index + 1}: enviado al executor "
                    f"(filas={len(rows)}, pendientes={len(pending)})",
                    flush=True,
                )

                # Cuando hay suficientes tareas pendientes, esperamos al menos
                # una terminada antes de seguir leyendo mas CSV.
                if len(pending) >= max_pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for finished in done:
                        pending.pop(finished)
                        result = finished.result()
                        total_read += result["read"]
                        total_valid += result["valid"]
                        total_discarded += result["discarded"]
                        buffer[result["chunk_index"]] = result
                    next_to_write = flush_ready(buffer, writer, log_file, next_to_write)

            # Al terminar la lectura, drenamos los chunks que siguen corriendo.
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for finished in done:
                    pending.pop(finished)
                    result = finished.result()
                    total_read += result["read"]
                    total_valid += result["valid"]
                    total_discarded += result["discarded"]
                    buffer[result["chunk_index"]] = result
                next_to_write = flush_ready(buffer, writer, log_file, next_to_write)

    elapsed = time.time() - start
    return {
        "total_read": total_read,
        "total_valid": total_valid,
        "total_discarded": total_discarded,
        "elapsed_seconds": elapsed,
        "output": str(output_path),
        "log": str(log_path),
    }


def parse_args() -> argparse.Namespace:
    """Define la interfaz CLI de la etapa paralela.

    Pasos:
    1. Crea parser de linea de comandos.
    2. Registra entrada, salida y log.
    3. Registra parametros de chunks y workers.
    4. Registra muestra y tipo de executor.
    5. Devuelve argumentos parseados.

    Permite configurar:
    - archivo de entrada;
    - archivo de salida CSV o Parquet;
    - ruta de `logs.txt`;
    - tamano de chunk;
    - cantidad de workers;
    - muestra para pruebas;
    - tipo de executor (`process` o `thread`).
    """
    parser = argparse.ArgumentParser(description="Procesa ventas por chunks en paralelo.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--executor", choices=["process", "thread"], default="process")
    return parser.parse_args()


def main() -> None:
    """Lee argumentos, ejecuta la validacion paralela y muestra el resumen.

    Pasos:
    1. Interpreta parametros de linea de comandos.
    2. Normaliza `workers`, `chunksize` y `sample`.
    3. Imprime configuracion inicial.
    4. Ejecuta `process_file`.
    5. Imprime metricas finales de lectura, validacion, descarte y tiempo.
    """
    args = parse_args()
    sample = args.sample if args.sample and args.sample > 0 else None
    workers = max(1, args.workers)
    chunksize = max(1, args.chunksize)

    print("Procesamiento paralelo por chunks", flush=True)
    print(f"Entrada: {args.input}", flush=True)
    print(f"Salida: {args.output}", flush=True)
    print(f"Chunksize: {chunksize}", flush=True)
    print(f"Workers: {workers}", flush=True)
    print(f"Executor: {args.executor}", flush=True)
    if sample:
        print(f"Sample: {sample}", flush=True)

    summary = process_file(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log,
        chunksize=chunksize,
        workers=workers,
        sample=sample,
        executor_type=args.executor,
    )
    print("Resumen:", flush=True)
    print(f" - filas leidas: {summary['total_read']}", flush=True)
    print(f" - filas validas: {summary['total_valid']}", flush=True)
    print(f" - filas descartadas: {summary['total_discarded']}", flush=True)
    print(f" - tiempo segundos: {summary['elapsed_seconds']:.2f}", flush=True)


if __name__ == "__main__":
    main()
