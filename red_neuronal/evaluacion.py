"""
evaluacion.py

Responsabilidad de este archivo:
---------------------------------
Una vez entrenada la red (entrenamiento.py), este archivo responde a la
pregunta: "¿que tan bien funciona realmente el modelo?"

Para eso, carga los pesos ya entrenados y el conjunto de PRUEBA (las
imagenes que la red nunca vio durante el entrenamiento), le pide a la
red que prediga cada una, y compara la prediccion contra la etiqueta
real. Con esas comparaciones se arma la matriz de confusion y se
calculan las metricas clasicas de clasificacion binaria:

    - Exactitud (accuracy)
    - Precision
    - Sensibilidad / Recall
    - F1-score

Estas 4 metricas son las mismas que aparecen en el articulo cientifico
en el que se basa este proyecto (seccion 2.5), y ademas se pueden
calcular con operaciones aritmeticas simples (sumas, restas y
divisiones), sin necesidad de ninguna libreria estadistica.

Convencion usada (clase positiva = catarata = 1):
    VP (Verdadero Positivo): la red dijo "catarata" y en verdad lo era.
    VN (Verdadero Negativo): la red dijo "normal" y en verdad lo era.
    FP (Falso Positivo): la red dijo "catarata" pero en verdad era normal.
    FN (Falso Negativo): la red dijo "normal" pero en verdad era catarata.
"""

import json
from red_neuronal import RedNeuronal

RUTA_PESOS = "pesos_entrenados.json"
RUTA_CONJUNTO_PRUEBA = "conjunto_prueba.json"


def cargar_conjunto_prueba(ruta_archivo=RUTA_CONJUNTO_PRUEBA):
    """
    Lee el archivo JSON generado por entrenamiento.py y lo reconstruye
    como una lista de tuplas (vector, etiqueta), lista para usarse.
    """
    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        datos = json.load(archivo)
    return [(item["vector"], item["etiqueta"]) for item in datos]


def evaluar_modelo(red=None, conjunto_prueba=None):
    """
    Evalua el modelo sobre el conjunto de prueba y muestra en pantalla
    la matriz de confusion y las metricas de desempeno.

    Parametros:
        red (RedNeuronal | None): si no se indica, se carga desde
            RUTA_PESOS.
        conjunto_prueba (list | None): si no se indica, se carga desde
            RUTA_CONJUNTO_PRUEBA.

    Retorna:
        dict: diccionario con todas las metricas calculadas.
    """
    if red is None:
        red = RedNeuronal.cargar_pesos(RUTA_PESOS)

    if conjunto_prueba is None:
        conjunto_prueba = cargar_conjunto_prueba()

    # Contadores de la matriz de confusion
    vp = vn = fp = fn = 0

    for vector_imagen, etiqueta_real in conjunto_prueba:
        _, clase_predicha = red.predecir(vector_imagen)

        if clase_predicha == 1 and etiqueta_real == 1:
            vp += 1
        elif clase_predicha == 0 and etiqueta_real == 0:
            vn += 1
        elif clase_predicha == 1 and etiqueta_real == 0:
            fp += 1
        elif clase_predicha == 0 and etiqueta_real == 1:
            fn += 1

    total = vp + vn + fp + fn

    exactitud = (vp + vn) / total if total > 0 else 0.0
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    _imprimir_resultados(vp, vn, fp, fn, exactitud, precision, recall, f1)

    return {
        "matriz_confusion": {"VP": vp, "VN": vn, "FP": fp, "FN": fn},
        "exactitud": exactitud,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def _imprimir_resultados(vp, vn, fp, fn, exactitud, precision, recall, f1):
    print("=" * 60)
    print("MATRIZ DE CONFUSION")
    print("=" * 60)
    print("                     Predicho: Normal   Predicho: Catarata")
    print(f"Real: Normal              {vn:^6}              {fp:^6}")
    print(f"Real: Catarata            {fn:^6}              {vp:^6}")

    print("\n" + "=" * 60)
    print("METRICAS DE DESEMPENO")
    print("=" * 60)
    print(f"  Exactitud (Accuracy) : {exactitud:.4f}  ({exactitud * 100:.2f} %)")
    print(f"  Precision            : {precision:.4f}")
    print(f"  Sensibilidad (Recall): {recall:.4f}")
    print(f"  F1-score             : {f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluar_modelo()
