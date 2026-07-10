"""Reglas de validacion para las transacciones de ventas.

Este modulo concentra las validaciones compartidas por el pipeline y por la
extraccion de filas validas. Separar estas funciones deja la logica lista para
ser usada por workers paralelos.
"""

import uuid
from datetime import datetime

MAX_EDAD_VALIDA = 110


def validar_rut(rut: str) -> bool:
    """Valida un RUT chileno mediante el algoritmo del digito verificador.

    Pasos:
    1. Convierte el valor a texto, elimina puntos/guion y pasa el DV a
       mayuscula para aceptar formatos como `12.345.678-5` o `123456785`.
    2. Separa cuerpo numerico y digito verificador informado.
    3. Recorre el cuerpo desde derecha a izquierda multiplicando por la serie
       2, 3, 4, 5, 6, 7 y reiniciando en 2.
    4. Calcula el digito esperado con modulo 11.
    5. Retorna `True` solo si el digito esperado coincide con el informado.

    Se usa en la validacion inicial para descartar identificadores de cliente
    inconsistentes antes de generar el dataset analitico.
    """
    texto = str(rut).strip().upper().replace(".", "").replace("-", "")
    if len(texto) < 2 or not texto[:-1].isdigit():
        return False

    cuerpo = texto[:-1]
    dv = texto[-1]
    suma = 0
    multiplo = 2

    for digito in reversed(cuerpo):
        suma += int(digito) * multiplo
        multiplo += 1
        if multiplo == 8:
            multiplo = 2

    resto = 11 - (suma % 11)
    dv_esperado = "0" if resto == 11 else "K" if resto == 10 else str(resto)
    return dv == dv_esperado


def validar_uuid(valor: str) -> bool:
    """Valida que `CODIGO CLIENTE` tenga formato UUID.

    Pasos:
    1. Convierte el valor recibido a texto y elimina espacios externos.
    2. Intenta construir un objeto `uuid.UUID`.
    3. Si la libreria acepta el valor, retorna `True`; si levanta
       `ValueError`, retorna `False`.

    Se usa para asegurar que la frecuencia de compra por cliente se calcule
    sobre identificadores estructuralmente validos.
    """
    try:
        uuid.UUID(str(valor).strip())
        return True
    except ValueError:
        return False


def validar_fecha(valor: str, formato: str | None = None) -> bool:
    """Valida fechas ISO o con un formato explicito.

    Pasos:
    1. Convierte el valor a texto y descarta cadenas vacias.
    2. Si `formato` es `None`, usa `datetime.fromisoformat`, pensado para
       `FECHA` en formato ISO 8601, por ejemplo `2026-05-08T00:02:53`.
    3. Si `formato` viene definido, usa `datetime.strptime`; en el proyecto se
       usa para `FECHA NACIMIENTO` con `%Y-%m-%d`.
    4. Retorna `False` ante cualquier `ValueError`.
    """
    texto = str(valor).strip()
    if not texto:
        return False

    try:
        if formato is None:
            datetime.fromisoformat(texto)
        else:
            datetime.strptime(texto, formato)
        return True
    except ValueError:
        return False


def validar_transaccion(row: list[str], header: list[str]) -> tuple[bool, str | None]:
    """Evalua si una fila cumple las reglas minimas de integridad.

    Args:
        row: valores de una transaccion leida desde el CSV.
        header: nombres de columnas del archivo.

    Returns:
        `(True, None)` cuando la fila es valida. Si se descarta, retorna
        `(False, motivo)`. El motivo queda escrito en `logs.txt` para auditar
        que regla fallo.

    Pasos:
    1. Verifica que la cantidad de valores coincida con el encabezado.
    2. Construye un diccionario `columna -> valor` para validar por nombre.
    3. Rechaza campos criticos vacios.
    4. Convierte y valida enteros (`SKU`, `UNIDADES`, `BOLETA`, `LOCAL`).
    5. Convierte y valida decimales (`PORCENTAJE DESCUENTO`,
       `MONTO APLICADO`).
    6. Valida identificadores (`RUN CLIENTE`, `CODIGO CLIENTE`).
    7. Valida fechas y calcula edad al momento de la transaccion.
    8. Rechaza edades negativas o mayores a `MAX_EDAD_VALIDA`.
    9. Valida que `GENERO` pertenezca al dominio esperado.

    La funcion no modifica la fila: solo decide si avanza al preprocesamiento.
    Cuando descarta, devuelve un motivo estructurado que luego queda en
    `logs.txt`.
    """
    if len(row) != len(header):
        return False, "columna:cantidad_de_columnas_incorrecta:sin_valor"

    # Convertimos la fila en diccionario para poder validar por nombre de campo.
    datos = {nombre: valor.strip() for nombre, valor in zip(header, row)}
    campos_requeridos = [
        "FECHA",
        "CANAL",
        "SKU",
        "PRODUCTO",
        "UNIDADES",
        "PORCENTAJE DESCUENTO",
        "MONTO APLICADO",
        "BOLETA",
        "LOCAL",
        "CODIGO CLIENTE",
        "RUN CLIENTE",
        "NOMBRES",
        "APELLIDOS",
        "FECHA NACIMIENTO",
        "GENERO",
    ]

    # Ninguna columna critica puede venir vacia: el analisis posterior depende
    # de fechas, montos, unidades, local y datos de cliente.
    for campo in campos_requeridos:
        if not datos[campo]:
            return False, f"{campo}:campo_vacio:{datos[campo]}"

    # Primero validamos columnas enteras. Si alguna falla, no seguimos con las
    # reglas numericas porque el cast ya no seria confiable.
    try:
        int(datos["SKU"])
        int(datos["UNIDADES"])
        int(datos["BOLETA"])
        int(datos["LOCAL"])
    except ValueError:
        return False, "SKU o UNIDADES o BOLETA o LOCAL:entero_invalido:valor_no_entero"

    if int(datos["UNIDADES"]) <= 0 or int(datos["BOLETA"]) <= 0 or int(datos["LOCAL"]) <= 0:
        return (
            False,
            "UNIDADES o BOLETA o LOCAL:valor_numerico_no_positivo:"
            f"{datos['UNIDADES']}/{datos['BOLETA']}/{datos['LOCAL']}",
        )

    # Descuento y monto se validan como decimales porque pueden venir con punto.
    try:
        descuento = float(datos["PORCENTAJE DESCUENTO"])
        monto = float(datos["MONTO APLICADO"])
    except ValueError:
        return (
            False,
            "PORCENTAJE DESCUENTO o MONTO APLICADO:decimal_invalido:"
            f"{datos['PORCENTAJE DESCUENTO']}/{datos['MONTO APLICADO']}",
        )

    if not (0 <= descuento <= 1) or monto <= 0:
        return (
            False,
            "PORCENTAJE DESCUENTO o MONTO APLICADO:descuento_o_monto_invalido:"
            f"{datos['PORCENTAJE DESCUENTO']}/{datos['MONTO APLICADO']}",
        )

    # Validaciones de identificadores y fechas, necesarias para unir clientes y
    # calcular edad correctamente.
    if not validar_rut(datos["RUN CLIENTE"]):
        return False, f"RUN CLIENTE:rut_invalido:{datos['RUN CLIENTE']}"
    if not validar_uuid(datos["CODIGO CLIENTE"]):
        return False, f"CODIGO CLIENTE:uuid_invalido:{datos['CODIGO CLIENTE']}"
    if not validar_fecha(datos["FECHA"]):
        return False, f"FECHA:formato_fecha_invalido:{datos['FECHA']}"
    if not validar_fecha(datos["FECHA NACIMIENTO"], "%Y-%m-%d"):
        return False, f"FECHA NACIMIENTO:formato_fecha_invalido:{datos['FECHA NACIMIENTO']}"

    fecha_transaccion = datetime.fromisoformat(datos["FECHA"])
    fecha_nacimiento = datetime.strptime(datos["FECHA NACIMIENTO"], "%Y-%m-%d")
    edad = (fecha_transaccion - fecha_nacimiento).days / 365.25
    if edad < 0 or edad > MAX_EDAD_VALIDA:
        return False, f"EDAD:edad_fuera_de_rango:{edad:.2f}"

    if datos["GENERO"] not in {"1", "2"}:
        return False, f"GENERO:valor_invalido:{datos['GENERO']}"

    return True, None
