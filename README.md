# Detector de Cataratas — Red Neuronal desde Cero en Python

Sistema de clasificación binaria (**ojo normal / ojo con catarata**) mediante un perceptrón multicapa (MLP) implementado **completamente desde cero en Python**, sin librerías de aprendizaje automático (sin TensorFlow, PyTorch, scikit-learn, etc.).

Proyecto desarrollado como **Producto Acreditable Final** del curso de Inteligencia Artificial — Universidad Nacional Pedro Ruiz Gallo, Escuela Profesional de Ingeniería en Computación e Informática.

> Basado en la problemática del síndrome del ojo rojo en el contexto peruano, y en el artículo científico *"Aplicación de redes neuronales artificiales para la detección binaria del síndrome del ojo rojo"* (Torres y Santos, 2026).

---

## Tabla de contenidos

- [Contexto y motivación](#-contexto-y-motivación)
- [Características](#-características)
- [Arquitectura de la red](#-arquitectura-de-la-red)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Dataset](#-dataset)
- [Resultados](#-resultados)
- [Limitaciones conocidas](#-limitaciones-conocidas)
- [Integrantes](#-integrantes)
- [Referencias](#-referencias)

---

## Contexto y motivación

El síndrome del ojo rojo es uno de los motivos de consulta más frecuentes en salud a nivel mundial. En el contexto peruano, especialmente en regiones como La Libertad, la falta de equipos y personal especializado dificulta el diagnóstico oportuno de condiciones como la **catarata**, una de las principales causas de discapacidad visual prevenible en el país.

Este proyecto implementa un sistema capaz de identificar automáticamente la presencia de cataratas a partir de una fotografía del ojo, pensado como apoyo al **cribado inicial** en zonas con escasa disponibilidad de especialistas — **sin pretender reemplazar el diagnóstico médico profesional.**

## Características

- Red neuronal (perceptrón multicapa) **implementada desde cero**: pesos, sesgos, propagación hacia adelante, retropropagación y actualización de pesos, sin librerías de ML.
- Preprocesamiento de imágenes (escala de grises, redimensionado, normalización) usando únicamente Pillow.
- Evaluación con métricas estándar: exactitud, precisión, sensibilidad (recall) y F1-score, más matriz de confusión.
- Interfaz gráfica de escritorio (CustomTkinter) con tres módulos: **Predecir**, **Entrenar** y **Evaluar**.
- Modo alternativo por terminal para entrenar, evaluar o predecir sin interfaz gráfica.

## Arquitectura de la red

Perceptrón multicapa con una única capa oculta:

```
Capa de entrada        Capa oculta          Capa de salida
 400 neuronas    ──►    8 neuronas    ──►     1 neurona
(imagen 20×20         (activación             (0 = normal
   aplanada)            sigmoide)              1 = catarata)
```

- **400 entradas**: cada píxel de la imagen de 20×20 en escala de grises.
- **8 neuronas ocultas**: suficientes para combinar información visual sin sobrecomplejizar el modelo.
- **1 neurona de salida**: probabilidad de catarata (umbral de decisión = 0.5).
- **Activación**: sigmoide (derivable, necesaria para retropropagación).
- **Función de error**: error cuadrático medio.

### Hiperparámetros de entrenamiento

| Parámetro | Valor |
|---|---|
| Épocas | 60 |
| Tasa de aprendizaje (λ) | 0.3 |
| Neuronas en capa oculta | 8 |
| Tamaño de imagen | 20 × 20 px |
| División entrenamiento / prueba | 80 % / 20 % (estratificada) |
| Semilla aleatoria | 42 |

## Estructura del proyecto

```
.
├── procesamiento_imagenes.py   # Carga y preprocesa imágenes (escala de grises, resize, normalización)
├── red_neuronal.py             # Clase RedNeuronal: forward prop, backprop, guardado/carga de pesos
├── entrenamiento.py            # Bucle de entrenamiento por épocas
├── evaluacion.py                # Cálculo de métricas y matriz de confusión
├── interfaz.py                  # Interfaz gráfica (CustomTkinter): Predecir / Entrenar / Evaluar
├── main.py                      # Punto de entrada alternativo por terminal
└── requisitos.txt                # Dependencias del proyecto
```

## Requisitos

- Python 3.10 o superior
- [Pillow](https://pypi.org/project/pillow/) — lectura y transformación de imágenes
- [CustomTkinter](https://pypi.org/project/customtkinter/) — interfaz gráfica

> Ninguna de las dos dependencias participa en el algoritmo de la red neuronal; toda la lógica de aprendizaje está programada en Python puro.

## ⚙️ Instalación

```bash
git clone https://github.com/tu-usuario/detector-cataratas.git
cd detector-cataratas

pip install pillow
pip install customtkinter
```

## Uso

### Opción 1: Interfaz gráfica (recomendada)

```bash
python interfaz.py
```

- **Predecir**: selecciona una imagen de ojo y presiona "Analizar imagen" para ver el diagnóstico.
- **Entrenar**: define las rutas de las carpetas "Normal" y "Cataratas" y presiona "Entrenar modelo" (se ejecuta en segundo plano con barra de progreso).
- **Evaluar**: presiona "Evaluar modelo" para ver la matriz de confusión y las métricas sobre el conjunto de prueba.

### Opción 2: Terminal

```bash
python main.py
```

Muestra un menú de texto con las mismas tres opciones: entrenar, evaluar y predecir.

## Dataset

Se utiliza un subconjunto del dataset público de Bitto (2024), *"Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation"*, disponible en Mendeley Data:

- **Ojos normales**: 649 imágenes
- **Ojos con cataratas**: 544 imágenes

📎 DOI: [10.17632/n9zp473wfw.2](https://doi.org/10.17632/n9zp473wfw.2)

Cada imagen pasa por un pipeline de preprocesamiento: conversión a escala de grises → redimensionado a 20×20 px (conservando proporción, relleno gris) → aplanado a vector de 400 valores → normalización a [0, 1].

## 📈 Resultados

Evaluado sobre 239 imágenes de prueba (20 % del total, nunca vistas en entrenamiento):

| Métrica | Valor |
|---|---|
| Exactitud | 97.1 % |
| Precisión | 96.4 % |
| Sensibilidad (Recall) | 97.2 % |
| F1-score | 96.8 % |

**Matriz de confusión:**

| | Predicho: Normal | Predicho: Catarata |
|---|---|---|
| **Real: Normal** | 126 | 4 |
| **Real: Catarata** | 3 | 106 |

## Limitaciones conocidas

El modelo tiene un **desempeño alto y confiable dentro del dominio del dataset de entrenamiento**, pero presenta errores sistemáticos al clasificar fotografías obtenidas de otras fuentes (por ejemplo, imágenes tipo "stock" con iluminación de estudio y reflejos de cámara), tendiendo a clasificarlas incorrectamente como "catarata" incluso cuando son ojos sanos.

Esto se debe principalmente a:
1. **Reflejos fotográficos (catchlights)** que, al reducir la imagen a 20×20 px, generan patrones de contraste similares a una opacidad de catarata.
2. **Distribución de entrenamiento limitada**: la red fue entrenada con imágenes de un único origen y condiciones homogéneas, lo que reduce su capacidad de generalización.

Esta limitación es consistente con lo reportado en la literatura (Torres y Santos, 2026), que observa el mismo fenómeno incluso en arquitecturas mucho más sofisticadas entrenadas con transferencia de aprendizaje.

> **Este sistema es una herramienta académica de apoyo al cribado, no un instrumento de diagnóstico clínico. No reemplaza la evaluación de un oftalmólogo.**

## Integrantes

- Aguirre Chávez, Rodrigo Osvaldo
- Barreno Rafael, Diego Fabricio
- Gutierrez Ambulay, Kevin David


## Referencias

- Bitto, A. K. (2024). *Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation.* Mendeley Data, 2. https://doi.org/10.17632/n9zp473wfw.2
- Torres Villan, M., & Santos Fernández, J. P. (2026). *Aplicación de redes neuronales artificiales para la detección binaria del síndrome del ojo rojo.* Ingeniería Investiga, 8, e1400. https://doi.org/10.47796/ing.v8i00.1400
- Cruz, P. P. (2011). *Inteligencia artificial con aplicaciones a la ingeniería.* Alfaomega.
- García Serrano, A. (2017). *Inteligencia artificial: fundamentos, práctica y aplicaciones* (2.ª ed.). Alfaomega.
- Palma Méndez, J. T., & Marín Morales, R. (2008). *Inteligencia artificial: métodos y técnicas.* McGraw-Hill.

---

<p align="center">Proyecto académico — Universidad Nacional Pedro Ruiz Gallo © 2026</p>
