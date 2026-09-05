"""
main.py

Responsabilidad de este archivo:
---------------------------------
Este es el punto de entrada del proyecto: el archivo que se ejecuta para
usar el sistema. Muestra un menu sencillo con 3 opciones:

    1) Entrenar el modelo desde cero (usa entrenamiento.py)
    2) Evaluar el modelo ya entrenado sobre el conjunto de prueba
       (usa evaluacion.py)
    3) Predecir una imagen nueva (una imagen que no esta en el dataset
       de entrenamiento ni de prueba)
    4) Salir

Para usar el proyecto por primera vez, se debe ejecutar la opcion 1
(entrenar), ya que genera el archivo "pesos_entrenados.json" que las
otras opciones necesitan.
"""

import os
from entrenamiento import entrenar_red, RUTA_PESOS, TAMAÑO_IMAGEN
from evaluacion import evaluar_modelo
from procesamiento_imagenes import cargar_y_procesar_imagen
from red_neuronal import RedNeuronal


def opcion_entrenar():
    print("\nIniciando entrenamiento del modelo...\n")
    entrenar_red()
    print("\nListo. Ya puedes usar la opcion 2 (evaluar) o 3 (predecir).")


def opcion_evaluar():
    if not os.path.exists(RUTA_PESOS):
        print("\n[Error] Todavia no existe un modelo entrenado.")
        print("Primero ejecuta la opcion 1 (Entrenar el modelo).")
        return
    print("\nEvaluando el modelo sobre el conjunto de prueba...\n")
    evaluar_modelo()


def opcion_predecir():
    if not os.path.exists(RUTA_PESOS):
        print("\n[Error] Todavia no existe un modelo entrenado.")
        print("Primero ejecuta la opcion 1 (Entrenar el modelo).")
        return

    ruta_imagen = input("\nRuta completa de la imagen a analizar: ").strip().strip('"')

    if not os.path.exists(ruta_imagen):
        print(f"[Error] No se encontro el archivo: {ruta_imagen}")
        return

    red = RedNeuronal.cargar_pesos(RUTA_PESOS)
    vector_imagen = cargar_y_procesar_imagen(ruta_imagen, tamaño=TAMAÑO_IMAGEN)
    probabilidad, clase = red.predecir(vector_imagen)

    etiqueta_texto = "CATARATA" if clase == 1 else "NORMAL"

    print("\n" + "=" * 40)
    print("RESULTADO DE LA PREDICCION")
    print("=" * 40)
    print(f"  Imagen analizada : {ruta_imagen}")
    print(f"  Probabilidad de catarata: {probabilidad:.4f}")
    print(f"  Clasificacion final     : {etiqueta_texto}")
    print("=" * 40)


def mostrar_menu():
    print("\n" + "=" * 60)
    print(" SISTEMA DE DETECCION DE CATARATAS - RED NEURONAL DESDE CERO")
    print("=" * 60)
    print("  1) Entrenar el modelo desde cero")
    print("  2) Evaluar el modelo con el conjunto de prueba")
    print("  3) Predecir una imagen nueva")
    print("  4) Salir")
    print("=" * 60)


def main():
    while True:
        mostrar_menu()
        opcion = input("Elige una opcion (1-4): ").strip()

        if opcion == "1":
            opcion_entrenar()
        elif opcion == "2":
            opcion_evaluar()
        elif opcion == "3":
            opcion_predecir()
        elif opcion == "4":
            print("\nSaliendo del programa. ¡Hasta luego!")
            break
        else:
            print("\nOpcion invalida, intenta de nuevo.")


if __name__ == "__main__":
    main()
