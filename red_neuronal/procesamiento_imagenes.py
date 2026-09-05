"""
procesamiento_imagenes.py

Responsabilidad de este archivo:
---------------------------------
Este archivo se encarga UNICAMENTE de leer las imagenes de ojos desde el
disco y transformarlas en algo que la red neuronal pueda entender: un
vector (lista) de numeros.

Una red neuronal, tal como la vimos en clase, no puede recibir una imagen
directamente. Necesita una lista de valores numericos (x1, x2, x3, ..., xn),
igual que en el ejemplo de la compuerta OR donde las entradas eran x1 y x2.

Para una imagen, cada numero de la lista representa el nivel de gris de
UN pixel. Por eso, antes de darle la imagen a la red, hacemos 3 pasos:

    1) Convertir la imagen a escala de grises (para no manejar 3 canales
       de color por pixel -rojo, verde, azul- sino solo 1 valor por pixel).
    2) Ajustar la imagen a un tamaño fijo y pequeno (por ejemplo,
       20 x 20 pixeles), CONSERVANDO su proporcion original y rellenando
       con gris el espacio sobrante (en vez de deformarla). Esto es
       obligatorio porque todas las imagenes de entrada a la red deben
       tener siempre la misma cantidad de numeros (la misma "cantidad
       de entradas").
    3) "Aplanar" la imagen: pasar de una matriz de 20 x 20 valores a una
       sola lista de 20*20 = 400 valores, uno detras de otro.

Sobre el uso de Pillow (PIL):
------------------------------
Esta es la UNICA libreria externa que usamos en todo el proyecto, y
solamente para abrir el archivo de imagen (jpg/png) y leer sus pixeles.
Programar un lector de imagenes JPEG desde cero no es razonable para
este curso (es un algoritmo de compresion complejo, un tema aparte de
redes neuronales). Todo lo que es la red neuronal en si (pesos, sesgos,
propagacion, error, retropropagacion, entrenamiento) esta hecho a mano
en el archivo red_neuronal.py, sin usar ninguna libreria de aprendizaje
automatico.
"""

import os
from PIL import Image, ImageOps


def _procesar_imagen_pil(ruta_imagen, tamaño=20):
    """
    Abre la imagen, la convierte a escala de grises y la ajusta a un
    cuadrado de (tamaño x tamaño) pixeles CONSERVANDO su proporcion
    original (no la deforma), rellenando el espacio sobrante con un
    gris neutro. Esta funcion es interna (de uso solo dentro de este
    archivo) y devuelve la imagen ya procesada como objeto de Pillow,
    todavia SIN convertir a lista de numeros.

    Por que rellenar en vez de simplemente estirar la imagen:
    -------------------------------------------------------------
    La primera version de este archivo usaba imagen.resize((tamaño,
    tamaño)), que fuerza CUALQUIER imagen a un cuadrado, sin importar
    su forma original. Si la foto original es rectangular (por ejemplo,
    una foto panoramica de un ojo, mucho mas ancha que alta), estirarla
    a la fuerza a un cuadrado deforma el ojo: lo aplasta o lo estira,
    cambiando su forma natural. Esto puede confundir a la red, porque
    las imagenes de entrenamiento y las imagenes nuevas terminarian
    deformadas de manera distinta segun su forma original.

    Para evitar esto, aqui usamos ImageOps.pad de Pillow, que:
        1) Redimensiona la imagen MANTENIENDO su proporcion original
           (ancho y alto se achican o agrandan en la misma medida).
        2) Centra esa imagen ya redimensionada dentro de un lienzo
           cuadrado de tamaño x tamaño.
        3) Rellena el espacio que sobra (arriba/abajo o a los lados,
           segun corresponda) con un color gris neutro (valor 128, a
           mitad de camino entre negro y blanco), en vez de dejar ese
           espacio con informacion "inventada" o deformando la imagen.
    """
    imagen = Image.open(ruta_imagen)

    # Convertir a escala de grises: 'L' significa "Luminancia", es
    # decir, un solo valor por pixel en vez de RGB (3 valores).
    imagen = imagen.convert("L")

    # Ajustar a un cuadrado manteniendo la proporcion, con relleno gris.
    imagen_cuadrada = ImageOps.pad(
        imagen, (tamaño, tamaño), method=Image.BICUBIC,
        color=128, centering=(0.5, 0.5)
    )

    return imagen_cuadrada


def cargar_y_procesar_imagen(ruta_imagen, tamaño=20):
    """
    Abre una imagen desde 'ruta_imagen', la convierte a escala de grises,
    la ajusta a (tamaño x tamaño) pixeles sin deformarla, y la transforma
    en una lista plana de numeros normalizados entre 0 y 1.

    Normalizar (dividir entre 255) es importante porque la funcion de
    activacion sigmoide que usa la red trabaja mejor con numeros pequenos;
    si le dieramos valores de pixel entre 0 y 255 directamente, la red
    tardaria mucho mas en aprender o directamente no aprenderia bien.

    Parametros:
        ruta_imagen (str): ruta completa del archivo de imagen.
        tamaño (int): la imagen se ajusta a tamaño x tamaño pixeles.

    Retorna:
        list[float]: lista de tamaño*tamaño numeros entre 0.0 y 1.0.
    """
    imagen_cuadrada = _procesar_imagen_pil(ruta_imagen, tamaño)

    # Obtener los valores de cada pixel (una lista de tamaño*tamaño
    # numeros enteros entre 0 y 255) y normalizarlos a [0, 1].
    pixeles = list(imagen_cuadrada.getdata())
    vector_normalizado = [valor / 255.0 for valor in pixeles]

    return vector_normalizado


def generar_vista_previa_procesada(ruta_imagen, tamaño=20, factor_ampliacion=10):
    """
    Genera una version AMPLIADA de exactamente lo que la red neuronal
    recibe como entrada para una imagen dada (en escala de grises,
    tamaño x tamaño pixeles, ya con el relleno para mantener la
    proporcion). Esta funcion NO se usa para entrenar ni predecir; es
    puramente para que la interfaz grafica pueda MOSTRAR al usuario
    "lo que ve la red" antes de analizar una imagen, ya que a simple
    vista una imagen de solo 20x20 pixeles es demasiado pequena para
    apreciarse bien.

    Se reutiliza _procesar_imagen_pil() (la misma funcion que usa
    cargar_y_procesar_imagen) para garantizar que la vista previa sea
    identica a lo que realmente procesa la red, y luego simplemente se
    agranda la imagen resultante para que se pueda ver con claridad.

    Parametros:
        ruta_imagen (str): ruta completa del archivo de imagen.
        tamaño (int): tamaño real que usa la red (20 por defecto).
        factor_ampliacion (int): cuantas veces se agranda la imagen
            resultante solo para poder visualizarla mejor.

    Retorna:
        PIL.Image.Image: imagen en escala de grises, ya ampliada.
    """
    imagen_cuadrada = _procesar_imagen_pil(ruta_imagen, tamaño)

    tamaño_ampliado = tamaño * factor_ampliacion
    # Image.NEAREST evita que Pillow "suavice" los pixeles al agrandar,
    # para que se vea claramente cada pixel individual tal cual lo
    # recibe la red (sin inventar detalle que no existe).
    return imagen_cuadrada.resize((tamaño_ampliado, tamaño_ampliado), Image.NEAREST)


def cargar_dataset(ruta_normal, ruta_catarata, tamaño=20, limite_por_clase=None):
    """
    Recorre las carpetas de imagenes 'normal' y 'catarata', procesa cada
    imagen con cargar_y_procesar_imagen(), y arma el conjunto de datos
    completo que usara la red para entrenar y evaluar.

    A cada imagen se le asigna una etiqueta (la respuesta "correcta" que
    la red debera aprender a predecir):
        0 -> ojo normal
        1 -> ojo con catarata

    Parametros:
        ruta_normal (str): carpeta con las imagenes de ojos normales.
        ruta_catarata (str): carpeta con las imagenes de ojos con catarata.
        tamaño (int): tamaño al que se redimensiona cada imagen.
        limite_por_clase (int | None): si se indica un numero, solo se
            cargan esa cantidad de imagenes por clase (util para hacer
            pruebas rapidas con menos datos). Si es None, se cargan todas.

    Retorna:
        list[tuple(list[float], int)]: lista de tuplas (vector_imagen, etiqueta)
    """
    dataset = []

    dataset += _procesar_carpeta(ruta_normal, etiqueta=0, tamaño=tamaño,
                                  limite=limite_por_clase)
    dataset += _procesar_carpeta(ruta_catarata, etiqueta=1, tamaño=tamaño,
                                  limite=limite_por_clase)

    return dataset


def _procesar_carpeta(ruta_carpeta, etiqueta, tamaño, limite):
    """
    Funcion auxiliar (de uso interno) que procesa todas las imagenes
    validas dentro de una carpeta y les asigna la misma etiqueta.
    """
    resultados = []

    if not os.path.isdir(ruta_carpeta):
        raise FileNotFoundError(
            f"No se encontro la carpeta de imagenes: {ruta_carpeta}\n"
            f"Verifica que la ruta este bien escrita en entrenamiento.py"
        )

    extensiones_validas = (".jpg", ".jpeg", ".png", ".bmp")
    nombres_archivos = sorted(os.listdir(ruta_carpeta))

    contador = 0
    for nombre in nombres_archivos:
        if not nombre.lower().endswith(extensiones_validas):
            continue

        if limite is not None and contador >= limite:
            break

        ruta_completa = os.path.join(ruta_carpeta, nombre)
        try:
            vector = cargar_y_procesar_imagen(ruta_completa, tamaño=tamaño)
            resultados.append((vector, etiqueta))
            contador += 1
        except Exception as error:
            # Si una imagen esta corrupta o no se puede leer, la omitimos
            # y avisamos, en vez de detener todo el programa.
            print(f"  [Aviso] No se pudo procesar '{nombre}': {error}")

    print(f"  Cargadas {contador} imagenes desde: {ruta_carpeta} (etiqueta={etiqueta})")
    return resultados
