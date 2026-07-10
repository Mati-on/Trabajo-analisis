"""Exploracion rapida del CSV de ventas.

Este script sirve para estimar el volumen del archivo antes de ejecutar el
pipeline completo. Cuenta las transacciones y calcula cuantos bloques se
formarian usando el mismo tamano de chunk que utiliza la etapa paralela.

No carga el archivo completo en memoria: recorre el CSV fila por fila.
"""

import csv
from pathlib import Path

# Archivo real definido por el enunciado. Se mantiene como valor por defecto
# para poder ejecutar el script sin argumentos desde la raiz del proyecto.
DATA_PATH = Path("data/ventas_completas.csv")

# Tamano de bloque usado como referencia en el pipeline paralelo.
CHUNK_SIZE = 100_000


def contar_registros(path: Path, chunk_size: int = CHUNK_SIZE):
    """Cuenta filas y agrupa el total en bloques logicos.

    Pasos:
    1. Abre el CSV en modo streaming.
    2. Lee el encabezado.
    3. Recorre filas una a una sin guardar el archivo completo.
    4. Cuenta filas acumuladas por chunk logico.
    5. Guarda el tamano de cada chunk completo.
    6. Agrega el ultimo chunk si queda parcial.
    7. Devuelve encabezado, total de filas y lista de tamanos de chunk.

    Args:
        path: ruta del CSV que se desea inspeccionar.
        chunk_size: cantidad de transacciones por bloque logico.

    Returns:
        Una tupla con:
        - encabezado del CSV;
        - total de transacciones leidas;
        - lista con el tamano de cada bloque.

    La lista `chunk_counts` permite saber cuantos chunks procesara el pipeline y
    si el ultimo bloque queda completo o parcial.
    """
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file, delimiter=";")
        header = next(reader)
        total_rows = 0
        chunk_counts = []
        current_chunk_rows = 0

        for _row in reader:
            total_rows += 1
            current_chunk_rows += 1

            if current_chunk_rows == chunk_size:
                chunk_counts.append(current_chunk_rows)
                current_chunk_rows = 0

        if current_chunk_rows:
            chunk_counts.append(current_chunk_rows)

    return header, total_rows, chunk_counts


if __name__ == "__main__":
    header, total_rows, chunk_counts = contar_registros(DATA_PATH)
    print("ARCHIVO:", DATA_PATH)
    print("TRANSACCIONES_TOTALES:", total_rows)
    print("NUMERO_DE_TROZOS:", len(chunk_counts))
    print("TAMANO_ULTIMO_TROZO:", chunk_counts[-1])
    print("COLUMNAS:", len(header))
    print("NOMBRES_COLUMNAS:", header)
