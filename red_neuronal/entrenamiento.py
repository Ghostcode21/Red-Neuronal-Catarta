"""
entrenamiento.py

Responsabilidad de este archivo:
---------------------------------
Este archivo organiza todo el proceso de entrenamiento:
    1) Carga las imagenes de las dos carpetas (normal y catarata) y las
       convierte en vectores numericos (usando procesamiento_imagenes.py).
    2) Separa el conjunto de datos en dos partes: entrenamiento (la red
       aprende con estas imagenes) y prueba (se guardan aparte, la red
       NUNCA las ve durante el entrenamiento, para poder evaluarla de
       forma justa despues, en evaluacion.py).
    3) Crea la red neuronal (red_neuronal.py) y la entrena durante varias
       "epocas" (una epoca = una pasada completa por todas las imagenes
       de entrenamiento), ajustando los pesos poco a poco.
    4) Al terminar, guarda los pesos entrenados y el conjunto de prueba
       en disco, para que main.py y evaluacion.py puedan usarlos despues
       sin tener que volver a entrenar.

Este archivo se puede ejecutar directamente (python entrenamiento.py) o
ser llamado desde main.py.
"""

import random
from procesamiento_imagenes import cargar_dataset
from red_neuronal import RedNeuronal

# ----------------------------------------------------------------------
# CONFIGURACION DEL PROYECTO
# ----------------------------------------------------------------------
# IMPORTANTE: cambia estas dos rutas por las carpetas donde tengas tus
# propias imagenes.
RUTA_NORMAL = r"C:\Users\Rodrigo Aguirre\Desktop\PROYECTO-IA\Datos\Normal"
RUTA_CATARATA = r"C:\Users\Rodrigo Aguirre\Desktop\PROYECTO-IA\Datos\Cataratas"

TAMAÑO_IMAGEN = 20          # las imagenes se redimensionan a 20x20 pixeles
N_ENTRADAS = TAMAÑO_IMAGEN * TAMAÑO_IMAGEN   # 400 entradas
N_OCULTAS = 8                # neuronas en la capa oculta
TASA_APRENDIZAJE = 0.3       # "lambda" (λ) visto en clase
EPOCAS = 60                  # numero de pasadas completas por el dataset
PORCENTAJE_ENTRENAMIENTO = 0.8   # 80% entrenamiento, 20% prueba
SEMILLA = 42                 # para que el orden aleatorio sea reproducible

RUTA_PESOS = "pesos_entrenados.json"
RUTA_CONJUNTO_PRUEBA = "conjunto_prueba.json"


def dividir_entrenamiento_prueba(dataset, porcentaje_entrenamiento, semilla):
    """
    Separa el dataset en entrenamiento y prueba de forma ESTRATIFICADA:
    es decir, separamos primero los ejemplos de cada clase (normal y
    catarata) y dividimos cada grupo por separado, para asegurarnos de
    que tanto el conjunto de entrenamiento como el de prueba mantengan
    una proporcion parecida de ambas clases (igual idea que la particion
    70/15/15 "estratificada" mencionada en el articulo cientifico).
    """
    random.seed(semilla)

    clase_0 = [muestra for muestra in dataset if muestra[1] == 0]
    clase_1 = [muestra for muestra in dataset if muestra[1] == 1]

    random.shuffle(clase_0)
    random.shuffle(clase_1)

    corte_0 = int(len(clase_0) * porcentaje_entrenamiento)
    corte_1 = int(len(clase_1) * porcentaje_entrenamiento)

    entrenamiento = clase_0[:corte_0] + clase_1[:corte_1]
    prueba = clase_0[corte_0:] + clase_1[corte_1:]

    random.shuffle(entrenamiento)
    random.shuffle(prueba)

    return entrenamiento, prueba


def entrenar_red(ruta_normal=RUTA_NORMAL, ruta_catarata=RUTA_CATARATA,
                  limite_por_clase=None, guardar_resultados=True,
                  callback_estado=None, callback_progreso=None):
    """
    Funcion principal de este archivo. Ejecuta todo el proceso de
    entrenamiento de principio a fin.

    Parametros:
        ruta_normal / ruta_catarata (str): carpetas con las imagenes.
        limite_por_clase (int | None): permite entrenar con menos
            imagenes por clase (util para pruebas rapidas).
        guardar_resultados (bool): si es True, guarda los pesos
            entrenados y el conjunto de prueba en disco.
        callback_estado (callable | None): funcion opcional que recibe
            un texto (str) cada vez que el entrenamiento pasa a una
            nueva etapa (por ejemplo, para mostrarlo en una interfaz
            grafica ademas de imprimirlo en la terminal). Si es None,
            simplemente no se llama a nada (el comportamiento por
            terminal no cambia en absoluto).
        callback_progreso (callable | None): funcion opcional que recibe
            (epoca_actual, epocas_totales, error_promedio) al terminar
            CADA epoca, pensada para actualizar una barra de progreso en
            una interfaz grafica. Si es None, no se llama a nada.

    Retorna:
        (red, conjunto_prueba): la red ya entrenada y la lista de
        imagenes de prueba (para poder evaluarla inmediatamente si se
        desea, sin tener que leer el archivo de disco).
    """
    def _avisar_estado(texto):
        if callback_estado is not None:
            callback_estado(texto)

    print("=" * 60)
    print("PASO 1: Cargando y procesando las imagenes...")
    print("=" * 60)
    _avisar_estado("Cargando y procesando las imagenes...")
    dataset = cargar_dataset(ruta_normal, ruta_catarata,
                              tamaño=TAMAÑO_IMAGEN,
                              limite_por_clase=limite_por_clase)

    if len(dataset) == 0:
        raise ValueError("No se cargo ninguna imagen. Revisa las rutas configuradas.")

    print(f"\nTotal de imagenes cargadas: {len(dataset)}")

    print("\n" + "=" * 60)
    print("PASO 2: Separando en conjunto de entrenamiento y de prueba...")
    print("=" * 60)
    _avisar_estado(f"{len(dataset)} imagenes cargadas. Separando entrenamiento/prueba...")
    entrenamiento, prueba = dividir_entrenamiento_prueba(
        dataset, PORCENTAJE_ENTRENAMIENTO, SEMILLA)
    print(f"Imagenes de entrenamiento: {len(entrenamiento)}")
    print(f"Imagenes de prueba (la red no las vera durante el entrenamiento): {len(prueba)}")

    print("\n" + "=" * 60)
    print("PASO 3: Entrenando la red neuronal...")
    print("=" * 60)
    _avisar_estado(
        f"Entrenando con {len(entrenamiento)} imagenes "
        f"(prueba: {len(prueba)})..."
    )
    red = RedNeuronal(n_entradas=N_ENTRADAS, n_ocultas=N_OCULTAS,
                       n_salidas=1, semilla=SEMILLA)

    for epoca in range(1, EPOCAS + 1):
        random.shuffle(entrenamiento)
        error_total = 0.0

        for vector_imagen, etiqueta in entrenamiento:
            error_muestra = red.entrenar_una_muestra(
                vector_imagen, etiqueta, TASA_APRENDIZAJE)
            error_total += error_muestra

        error_promedio = error_total / len(entrenamiento)

        # Mostramos el progreso cada pocas epocas para no saturar la pantalla.
        if epoca == 1 or epoca % 5 == 0 or epoca == EPOCAS:
            print(f"  Epoca {epoca:3d}/{EPOCAS} - Error promedio (MSE): {error_promedio:.5f}")

        # A diferencia del print (que solo se hace cada 5 epocas), este
        # callback se llama SIEMPRE, para que una barra de progreso en
        # una interfaz grafica se vea fluida y no "a saltos".
        if callback_progreso is not None:
            callback_progreso(epoca, EPOCAS, error_promedio)

    print("\nEntrenamiento finalizado.")
    _avisar_estado("Entrenamiento finalizado. Guardando resultados...")

    if guardar_resultados:
        print("\n" + "=" * 60)
        print("PASO 4: Guardando pesos entrenados y conjunto de prueba...")
        print("=" * 60)
        red.guardar_pesos(RUTA_PESOS)
        print(f"  Pesos guardados en: {RUTA_PESOS}")

        _guardar_conjunto_prueba(prueba, RUTA_CONJUNTO_PRUEBA)
        print(f"  Conjunto de prueba guardado en: {RUTA_CONJUNTO_PRUEBA}")

    _avisar_estado("¡Listo! Modelo entrenado y guardado correctamente.")
    return red, prueba


def _guardar_conjunto_prueba(prueba, ruta_archivo):
    """
    Guarda el conjunto de prueba (vectores + etiquetas) en un archivo
    JSON, para que evaluacion.py pueda usar EXACTAMENTE las mismas
    imagenes que la red nunca vio durante el entrenamiento.
    """
    import json
    datos = [{"vector": vector, "etiqueta": etiqueta} for vector, etiqueta in prueba]
    with open(ruta_archivo, "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo)


if __name__ == "__main__":
    entrenar_red()
