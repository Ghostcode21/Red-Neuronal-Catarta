"""
red_neuronal.py

Responsabilidad de este archivo:
---------------------------------
Aqui esta el corazon del proyecto: la implementacion de la red neuronal
DESDE CERO, sin usar ninguna libreria de aprendizaje automatico.

Relacion con lo visto en clase (Semana 13-14, "Algoritmo de entrenamiento
del perceptron"):
------------------------------------------------------------------------
En clase se vio un perceptron de UNA sola neurona, con:
    - Entradas x1...xn, cada una con su peso w1...wn.
    - Un umbral (theta).
    - Salida: z = suma(wi * xi) - theta
    - Funcion de activacion escalon (0 o 1).
    - Ajuste de pesos con la regla: e = (d - z), Δw = λ * e * x

En este proyecto usamos exactamente esos mismos conceptos (pesos, sesgo,
suma ponderada, error, tasa de aprendizaje lambda), pero con UNA
extension minima necesaria para que la red pueda distinguir patrones
mas complejos que una imagen: agregamos UNA capa oculta con varias
neuronas, en vez de una sola neurona.

Dos diferencias importantes frente a la clase, y por que son necesarias:

    1) Funcion de activacion: en vez del escalon (que solo da 0 o 1 sin
       puntos intermedios), usamos la funcion SIGMOIDE. La sigmoide es
       la version "suave" del escalon: tambien va de 0 a 1, pero de
       forma continua. La necesitamos porque el algoritmo de
       retropropagacion (backpropagation) requiere poder calcular la
       DERIVADA de la funcion de activacion, y el escalon no se puede
       derivar (tiene un salto brusco). La sigmoide si se puede derivar
       facilmente: derivada = salida * (1 - salida).

    2) Retropropagacion (backpropagation): con una sola neurona (como en
       clase), el error se usa directamente para ajustar sus pesos. Pero
       ahora hay DOS capas de pesos (entrada->oculta y oculta->salida).
       Backpropagation es simplemente la generalizacion logica de la
       misma idea de clase: primero calculamos el error en la salida
       (igual que antes: e = d - z), ajustamos esos pesos, y LUEGO
       "propagamos" ese mismo error hacia atras, hacia la capa oculta,
       para saber cuanto le "toca de culpa" a cada neurona oculta y
       ajustar tambien sus pesos. Es la misma regla delta de clase,
       aplicada dos veces (una por cada capa de pesos).

Estructura de la red:
----------------------
    entrada (400 numeros)  -->  capa oculta (8 neuronas)  -->  salida (1 neurona)
"""

import math
import random
import json


class RedNeuronal:

    def __init__(self, n_entradas, n_ocultas, n_salidas=1, semilla=42):
        """
        Inicializa la red neuronal con pesos y sesgos aleatorios pequenos.

        Parametros:
            n_entradas (int): cantidad de valores de entrada (ej. 400
                para una imagen de 20x20 pixeles).
            n_ocultas (int): cantidad de neuronas en la capa oculta.
            n_salidas (int): cantidad de neuronas de salida (1, porque
                es un problema binario: normal o catarata).
            semilla (int): semilla para los numeros aleatorios, para que
                el entrenamiento sea reproducible (siempre parta de los
                mismos pesos iniciales al repetirlo).
        """
        random.seed(semilla)

        self.n_entradas = n_entradas
        self.n_ocultas = n_ocultas
        self.n_salidas = n_salidas

        # Pesos entre la capa de entrada y la capa oculta.
        # W1[j][i] = peso que conecta la entrada i con la neurona oculta j.
        self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(n_entradas)]
                   for _ in range(n_ocultas)]
        # Sesgo (equivalente al "theta" de clase, pero sumando en vez de
        # restar, que es la convencion mas comun) de cada neurona oculta.
        self.b1 = [random.uniform(-0.5, 0.5) for _ in range(n_ocultas)]

        # Pesos entre la capa oculta y la capa de salida.
        # W2[k][j] = peso que conecta la neurona oculta j con la salida k.
        self.W2 = [[random.uniform(-0.5, 0.5) for _ in range(n_ocultas)]
                   for _ in range(n_salidas)]
        self.b2 = [random.uniform(-0.5, 0.5) for _ in range(n_salidas)]

    # ------------------------------------------------------------------
    # Funcion de activacion
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoide(x):
        """
        Funcion de activacion sigmoide: comprime cualquier numero real
        a un valor entre 0 y 1. Es la version derivable de la funcion
        escalon vista en clase.
        """
        # Se limita el valor de x para evitar errores de "overflow"
        # cuando x es un numero muy grande o muy negativo.
        x = max(-60.0, min(60.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _derivada_sigmoide(salida_sigmoide):
        """
        Derivada de la sigmoide, expresada en funcion de su propia
        salida (esto es una propiedad matematica conveniente de la
        sigmoide que evita tener que recalcular todo de nuevo).
        """
        return salida_sigmoide * (1.0 - salida_sigmoide)

    # ------------------------------------------------------------------
    # Propagacion hacia adelante
    # ------------------------------------------------------------------
    def propagar_adelante(self, entrada):
        """
        Calcula la salida de la red para una entrada dada, pasando la
        informacion "hacia adelante": de la capa de entrada, a la capa
        oculta, y de ahi a la capa de salida.

        Es exactamente la misma formula vista en clase (z = suma(w*x) + b),
        aplicada primero para obtener la capa oculta, y luego usando esa
        capa oculta como "entrada" para obtener la salida final.

        Retorna:
            (salida_oculta, salida_final): ambas son listas de numeros
            entre 0 y 1 (ya con la sigmoide aplicada). Se devuelven las
            dos porque salida_oculta se necesita despues, en el paso de
            retropropagacion.
        """
        # --- Capa oculta ---
        salida_oculta = []
        for j in range(self.n_ocultas):
            suma = self.b1[j]
            for i in range(self.n_entradas):
                suma += self.W1[j][i] * entrada[i]
            salida_oculta.append(self._sigmoide(suma))

        # --- Capa de salida ---
        salida_final = []
        for k in range(self.n_salidas):
            suma = self.b2[k]
            for j in range(self.n_ocultas):
                suma += self.W2[k][j] * salida_oculta[j]
            salida_final.append(self._sigmoide(suma))

        return salida_oculta, salida_final

    # ------------------------------------------------------------------
    # Calculo del error
    # ------------------------------------------------------------------
    @staticmethod
    def _calcular_error_cuadratico(salida_esperada, salida_obtenida):
        """
        Calcula el error cuadratico medio entre lo que la red devolvio
        y lo que deberia haber devuelto.

        Es la generalizacion de e = (d - z) visto en clase: aqui se
        eleva al cuadrado (para que errores positivos y negativos no se
        cancelen entre si) y se promedia entre todas las salidas.
        """
        total = 0.0
        for esperado, obtenido in zip(salida_esperada, salida_obtenida):
            total += (esperado - obtenido) ** 2
        return total / len(salida_esperada)

    # ------------------------------------------------------------------
    # Retropropagacion + actualizacion de pesos
    # ------------------------------------------------------------------
    def entrenar_una_muestra(self, entrada, etiqueta_esperada, tasa_aprendizaje):
        """
        Realiza UN paso completo de entrenamiento para UNA imagen:
            1) Propagacion hacia adelante (obtener la prediccion actual).
            2) Calculo del error en la salida.
            3) Retropropagacion del error hacia la capa oculta.
            4) Actualizacion de todos los pesos y sesgos.

        Parametros:
            entrada (list[float]): vector de la imagen (400 numeros).
            etiqueta_esperada (int): 0 (normal) o 1 (catarata).
            tasa_aprendizaje (float): equivalente al "lambda" visto en
                clase; controla que tan grande es cada ajuste de pesos.

        Retorna:
            float: el error cuadratico de esta muestra (sirve solo para
            poder graficar/imprimir el progreso del entrenamiento).
        """
        salida_esperada = [float(etiqueta_esperada)]

        # 1) Propagacion hacia adelante
        salida_oculta, salida_final = self.propagar_adelante(entrada)

        # 2) Error en la capa de salida
        error_cuadratico = self._calcular_error_cuadratico(salida_esperada, salida_final)

        # "delta" de la capa de salida: combina el error (d - z) con la
        # derivada de la sigmoide, tal como exige la regla de la cadena
        # usada en backpropagation. Es la version derivable de la regla
        # delta de clase (Δw = λ * e * x).
        delta_salida = []
        for k in range(self.n_salidas):
            error_k = salida_esperada[k] - salida_final[k]
            delta_salida.append(error_k * self._derivada_sigmoide(salida_final[k]))

        # 3) Retropropagar el error hacia la capa oculta: cada neurona
        # oculta "hereda" una parte del error de salida, proporcional
        # al peso que la conecta con esa salida.
        delta_oculta = []
        for j in range(self.n_ocultas):
            error_propagado = 0.0
            for k in range(self.n_salidas):
                error_propagado += delta_salida[k] * self.W2[k][j]
            delta_oculta.append(error_propagado * self._derivada_sigmoide(salida_oculta[j]))

        # 4) Actualizar pesos y sesgos de la capa de salida (oculta -> salida)
        for k in range(self.n_salidas):
            for j in range(self.n_ocultas):
                self.W2[k][j] += tasa_aprendizaje * delta_salida[k] * salida_oculta[j]
            self.b2[k] += tasa_aprendizaje * delta_salida[k]

        # Actualizar pesos y sesgos de la capa oculta (entrada -> oculta)
        for j in range(self.n_ocultas):
            for i in range(self.n_entradas):
                self.W1[j][i] += tasa_aprendizaje * delta_oculta[j] * entrada[i]
            self.b1[j] += tasa_aprendizaje * delta_oculta[j]

        return error_cuadratico

    # ------------------------------------------------------------------
    # Prediccion
    # ------------------------------------------------------------------
    def predecir(self, entrada):
        """
        Dada una imagen (ya procesada como vector), devuelve:
            - probabilidad (float entre 0 y 1): que tan "segura" esta
              la red de que sea catarata (cerca de 1) o normal (cerca de 0).
            - clase (int): 0 (normal) o 1 (catarata), aplicando el umbral
              de decision 0.5, igual que el criterio visto en clase de
              "si supera cierto umbral, se dispara".
        """
        _, salida_final = self.propagar_adelante(entrada)
        probabilidad = salida_final[0]
        clase = 1 if probabilidad >= 0.5 else 0
        return probabilidad, clase

    # ------------------------------------------------------------------
    # Guardar y cargar los pesos entrenados
    # ------------------------------------------------------------------
    def guardar_pesos(self, ruta_archivo):
        """
        Guarda todos los pesos y sesgos de la red en un archivo JSON,
        junto con las dimensiones de la red, para poder reconstruirla
        exactamente igual al momento de cargarla despues.
        """
        datos = {
            "n_entradas": self.n_entradas,
            "n_ocultas": self.n_ocultas,
            "n_salidas": self.n_salidas,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }
        with open(ruta_archivo, "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo)

    @classmethod
    def cargar_pesos(cls, ruta_archivo):
        """
        Crea una nueva instancia de RedNeuronal y le carga los pesos
        previamente entrenados y guardados en un archivo JSON.
        """
        with open(ruta_archivo, "r", encoding="utf-8") as archivo:
            datos = json.load(archivo)

        red = cls(datos["n_entradas"], datos["n_ocultas"], datos["n_salidas"])
        red.W1 = datos["W1"]
        red.b1 = datos["b1"]
        red.W2 = datos["W2"]
        red.b2 = datos["b2"]
        return red
