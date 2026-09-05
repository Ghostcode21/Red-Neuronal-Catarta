"""
interfaz.py

Responsabilidad de este archivo:
---------------------------------
Interfaz grafica del proyecto, pensada para que la use una persona SIN
conocimientos de informatica. Esta ventana es solo una "capa visual":
NO reimplementa nada de la logica de la red neuronal. Unicamente llama
a las funciones que ya existen y estan probadas:

    - procesamiento_imagenes.cargar_y_procesar_imagen()
    - red_neuronal.RedNeuronal.cargar_pesos() / .predecir()
    - entrenamiento.entrenar_red()
    - evaluacion.evaluar_modelo()

Sobre CustomTkinter:
----------------------
Para el diseño se usa CustomTkinter (`pip install customtkinter`), una
libreria construida ENCIMA de Tkinter (no lo reemplaza) que ofrece
esquinas redondeadas, mejor tipografia y una apariencia mas moderna sin
tener que dibujar cada boton a mano. Esta es la UNICA libreria nueva
del proyecto y es puramente visual: no participa en absoluto en el
entrenamiento, la prediccion ni ningun calculo de la red neuronal.

La ventana tiene una barra lateral con 3 secciones: "Predecir",
"Entrenar" y "Evaluar". Solo se muestra una seccion a la vez.

Nota tecnica sobre los hilos (threads):
-----------------------------------------
Entrenar el modelo puede tardar bastante. Si se hiciera directamente al
hacer clic en el boton, toda la ventana se "congelaria" hasta terminar.
Para evitarlo, el entrenamiento corre en un hilo (thread) aparte, y se
usa una cola (queue) para que ese hilo le avise a la ventana principal
como va el progreso, sin congelarla. La ventana revisa esa cola cada
150 ms usando el metodo .after(), que es la forma segura de actualizar
una interfaz de Tkinter/CustomTkinter desde un hilo secundario.
"""

import os
import threading
import queue

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

import entrenamiento
import evaluacion
from procesamiento_imagenes import cargar_y_procesar_imagen, generar_vista_previa_procesada
from red_neuronal import RedNeuronal


# ----------------------------------------------------------------------
# Configuracion visual general
# ----------------------------------------------------------------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

COLOR_FONDO_APP = "#EEF1F6"
COLOR_SIDEBAR = "#1B2430"
COLOR_SIDEBAR_HOVER = "#28344A"
COLOR_SIDEBAR_ACTIVO = "#2F6FED"
COLOR_TARJETA = "#FFFFFF"
COLOR_TEXTO = "#1F2933"
COLOR_TEXTO_SECUNDARIO = "#6B7684"
COLOR_PRIMARIO = "#2F6FED"
COLOR_PRIMARIO_HOVER = "#2557C7"
COLOR_EXITO = "#1F9254"
COLOR_EXITO_FONDO = "#E6F6ED"
COLOR_ALERTA = "#C2410C"
COLOR_ALERTA_FONDO = "#FDECE3"
COLOR_BORDE = "#E1E6ED"
COLOR_BADGE_FONDO = "#EEF2FF"

# Las fuentes de CustomTkinter (CTkFont) NECESITAN que ya exista una
# ventana principal (root) creada antes de poder inicializarse. Por eso
# no se crean aqui arriba directamente (a nivel de modulo), sino que se
# dejan en None y se rellenan con _inicializar_fuentes(), que se llama
# al principio de InterfazProyecto.__init__(), una vez que la ventana
# ya existe.
FUENTE_TITULO_APP = None
FUENTE_SUBTITULO = None
FUENTE_NORMAL = None
FUENTE_BOTON = None
FUENTE_NAV = None
FUENTE_RESULTADO = None
FUENTE_STAT_NUMERO = None
FUENTE_STAT_ETIQUETA = None
FUENTE_BADGE = None


def _inicializar_fuentes():
    """
    Crea todas las fuentes (CTkFont) usadas en la interfaz. Debe
    llamarse DESPUES de crear la ventana principal (ctk.CTk()), porque
    CTkFont necesita una ventana ya existente para poder inicializarse.
    """
    global FUENTE_TITULO_APP, FUENTE_SUBTITULO
    global FUENTE_NORMAL, FUENTE_BOTON, FUENTE_NAV, FUENTE_RESULTADO
    global FUENTE_STAT_NUMERO, FUENTE_STAT_ETIQUETA, FUENTE_BADGE

    FUENTE_TITULO_APP = ctk.CTkFont(family="Segoe UI", size=20, weight="bold")
    FUENTE_SUBTITULO = ctk.CTkFont(family="Segoe UI", size=12)
    FUENTE_NORMAL = ctk.CTkFont(family="Segoe UI", size=12)
    FUENTE_BOTON = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
    FUENTE_NAV = ctk.CTkFont(family="Segoe UI", size=13)
    FUENTE_RESULTADO = ctk.CTkFont(family="Segoe UI", size=24, weight="bold")
    FUENTE_STAT_NUMERO = ctk.CTkFont(family="Segoe UI", size=26, weight="bold")
    FUENTE_STAT_ETIQUETA = ctk.CTkFont(family="Segoe UI", size=11)
    FUENTE_BADGE = ctk.CTkFont(family="Segoe UI", size=11, weight="bold")


class InterfazProyecto(ctk.CTk):
    """Ventana principal: barra lateral + area de contenido con 3 secciones."""

    def __init__(self):
        super().__init__()
        _inicializar_fuentes()

        self.title("Detector de Cataratas - Red Neuronal")
        self.geometry("980x680")
        self.minsize(860, 600)
        self.configure(fg_color=COLOR_FONDO_APP)

        self.cola_entrenamiento = queue.Queue()

        self._crear_layout()
        self._actualizar_estado_modelo()

    # ------------------------------------------------------------------
    def _crear_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._crear_barra_lateral()

        # Contenedor donde viven las 3 paginas, apiladas una encima de
        # otra; solo se "levanta" (tkraise) la que corresponde ver.
        self.contenedor_paginas = ctk.CTkFrame(self, fg_color=COLOR_FONDO_APP)
        self.contenedor_paginas.grid(row=0, column=1, sticky="nsew",
                                      padx=(0, 24), pady=24)
        self.contenedor_paginas.grid_rowconfigure(0, weight=1)
        self.contenedor_paginas.grid_columnconfigure(0, weight=1)

        self.pagina_predecir = PaginaPredecir(self.contenedor_paginas, self)
        self.pagina_entrenar = PaginaEntrenar(self.contenedor_paginas, self)
        self.pagina_evaluar = PaginaEvaluar(self.contenedor_paginas, self)

        for pagina in (self.pagina_predecir, self.pagina_entrenar, self.pagina_evaluar):
            pagina.grid(row=0, column=0, sticky="nsew")

        self._mostrar_pagina("predecir")

    def _crear_barra_lateral(self):
        barra = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color=COLOR_SIDEBAR)
        barra.grid(row=0, column=0, sticky="nsw")
        barra.grid_propagate(False)

        # --- Marca / titulo ---
        marco_marca = ctk.CTkFrame(barra, fg_color="transparent")
        marco_marca.pack(fill="x", padx=20, pady=(28, 20))

        ctk.CTkLabel(marco_marca, text="👁", font=ctk.CTkFont(size=28),
                     text_color="white").pack(anchor="w")
        ctk.CTkLabel(marco_marca, text="Detector de\nCataratas",
                     font=FUENTE_TITULO_APP, text_color="white",
                     justify="left").pack(anchor="w", pady=(6, 0))
        ctk.CTkLabel(marco_marca, text="Red neuronal desde cero",
                     font=FUENTE_SUBTITULO, text_color="#9AA5B5").pack(anchor="w")

        ctk.CTkFrame(barra, height=1, fg_color="#2A3446").pack(fill="x", padx=20, pady=(4, 16))

        # --- Botones de navegacion ---
        self.botones_nav = {}
        opciones = [
            ("predecir", "🔍", "Predecir"),
            ("entrenar", "⚙", "Entrenar"),
            ("evaluar", "📊", "Evaluar"),
        ]
        for clave, icono, texto in opciones:
            boton = ctk.CTkButton(
                barra, text=f"  {icono}   {texto}", anchor="w",
                font=FUENTE_NAV, height=44, corner_radius=10,
                fg_color="transparent", hover_color=COLOR_SIDEBAR_HOVER,
                text_color="#C7CFDB",
                command=lambda c=clave: self._mostrar_pagina(c))
            boton.pack(fill="x", padx=14, pady=4)
            self.botones_nav[clave] = boton

        # --- Aviso de estado del modelo, al pie de la barra ---
        self.marco_aviso = ctk.CTkFrame(barra, fg_color="#2A2116", corner_radius=10)
        self.etiqueta_aviso = ctk.CTkLabel(
            self.marco_aviso, text="", font=ctk.CTkFont(size=11),
            text_color="#F5B971", wraplength=170, justify="left")
        self.etiqueta_aviso.pack(padx=12, pady=10)

    def _mostrar_pagina(self, clave):
        paginas = {
            "predecir": self.pagina_predecir,
            "entrenar": self.pagina_entrenar,
            "evaluar": self.pagina_evaluar,
        }
        paginas[clave].tkraise()

        for c, boton in self.botones_nav.items():
            if c == clave:
                boton.configure(fg_color=COLOR_SIDEBAR_ACTIVO, text_color="white")
            else:
                boton.configure(fg_color="transparent", text_color="#C7CFDB")

    # ------------------------------------------------------------------
    # Utilidades compartidas entre paginas
    # ------------------------------------------------------------------
    def existe_modelo_entrenado(self):
        return os.path.exists(entrenamiento.RUTA_PESOS)

    def existe_conjunto_prueba(self):
        return os.path.exists(entrenamiento.RUTA_CONJUNTO_PRUEBA)

    def _actualizar_estado_modelo(self):
        hay_modelo = self.existe_modelo_entrenado()

        self.pagina_predecir.actualizar_disponibilidad(hay_modelo)
        self.pagina_evaluar.actualizar_disponibilidad(
            hay_modelo and self.existe_conjunto_prueba())

        if not hay_modelo:
            self.etiqueta_aviso.configure(
                text="⚠ Aún no hay un modelo entrenado.\nVe a 'Entrenar' para crear uno.")
            self.marco_aviso.pack(fill="x", padx=14, pady=(10, 16), side="bottom")
        else:
            self.marco_aviso.pack_forget()


# ==========================================================================
# Componentes reutilizables
# ==========================================================================
class Tarjeta(ctk.CTkFrame):
    """Contenedor blanco con esquinas redondeadas, usado como 'card' base."""

    def __init__(self, master, **kwargs):
        kwargs.setdefault("fg_color", COLOR_TARJETA)
        kwargs.setdefault("corner_radius", 16)
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", COLOR_BORDE)
        super().__init__(master, **kwargs)


class Badge(ctk.CTkLabel):
    """Pequeña etiqueta informativa tipo 'insignia', solo de lectura."""

    def __init__(self, master, texto, **kwargs):
        super().__init__(
            master, text=texto, font=FUENTE_BADGE, text_color=COLOR_PRIMARIO,
            fg_color=COLOR_BADGE_FONDO, corner_radius=8, padx=10, pady=5, **kwargs)


class TarjetaEstadistica(Tarjeta):
    """Tarjeta usada en la pestaña Evaluar: un numero grande + su nombre."""

    def __init__(self, master, nombre, **kwargs):
        super().__init__(master, **kwargs)
        self.etiqueta_valor = ctk.CTkLabel(self, text="—", font=FUENTE_STAT_NUMERO,
                                            text_color=COLOR_PRIMARIO)
        self.etiqueta_valor.pack(pady=(18, 2))
        ctk.CTkLabel(self, text=nombre, font=FUENTE_STAT_ETIQUETA,
                     text_color=COLOR_TEXTO_SECUNDARIO).pack(pady=(0, 16))

    def actualizar(self, valor_porcentaje):
        self.etiqueta_valor.configure(text=f"{valor_porcentaje * 100:.1f}%")


# ==========================================================================
# PAGINA: PREDECIR
# ==========================================================================
class PaginaPredecir(ctk.CTkFrame):
    """
    Flujo pensado para una persona sin conocimientos tecnicos:
        1) Elegir/soltar una imagen en la zona de carga.
        2) Ver la vista previa de la imagen elegida.
        3) Presionar "Analizar imagen".
        4) Ver el resultado en una tarjeta grande, con color e icono.
    """

    def __init__(self, contenedor, ventana_principal):
        super().__init__(contenedor, fg_color="transparent")
        self.ventana = ventana_principal
        self.ruta_imagen_seleccionada = None
        self.imagen_ctk = None
        self.imagen_diagnostico_ctk = None

        self._construir()

    def _construir(self):

        columnas = ctk.CTkFrame(self, fg_color="transparent")
        columnas.pack(fill="both", expand=True)
        columnas.grid_columnconfigure(0, weight=1, uniform="col")
        columnas.grid_columnconfigure(1, weight=1, uniform="col")
        columnas.grid_rowconfigure(0, weight=1)

        # --- Columna izquierda: zona de carga de imagen ---
        tarjeta_carga = Tarjeta(columnas)
        tarjeta_carga.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        contenido_carga = ctk.CTkFrame(tarjeta_carga, fg_color="transparent")
        contenido_carga.pack(fill="both", expand=True, padx=22, pady=22)

        ctk.CTkLabel(contenido_carga, text="1. Imagen del ojo",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLOR_TEXTO).pack(anchor="w", pady=(0, 10))

        self.zona_carga = ctk.CTkFrame(
            contenido_carga, fg_color="#F7F9FC", corner_radius=14,
            border_width=2, border_color=COLOR_BORDE, height=260, cursor="hand2")
        self.zona_carga.pack(fill="both", expand=True)
        self.zona_carga.pack_propagate(False)
        self.zona_carga.bind("<Button-1>", lambda e: self._seleccionar_imagen())

        self.icono_carga = ctk.CTkLabel(self.zona_carga, text="📁",
                                         font=ctk.CTkFont(size=40),
                                         text_color=COLOR_TEXTO_SECUNDARIO)
        self.icono_carga.pack(pady=(50, 6))
        self.icono_carga.bind("<Button-1>", lambda e: self._seleccionar_imagen())

        self.texto_carga = ctk.CTkLabel(
            self.zona_carga, text="Haz clic para seleccionar una imagen",
            font=FUENTE_NORMAL, text_color=COLOR_TEXTO_SECUNDARIO)
        self.texto_carga.pack()
        self.texto_carga.bind("<Button-1>", lambda e: self._seleccionar_imagen())

        self.etiqueta_previa = ctk.CTkLabel(self.zona_carga, text="", image=None)

        self.boton_seleccionar = ctk.CTkButton(
            contenido_carga, text="📁  Seleccionar otra imagen", font=FUENTE_BOTON,
            fg_color=COLOR_PRIMARIO, hover_color=COLOR_PRIMARIO_HOVER,
            corner_radius=10, height=40, command=self._seleccionar_imagen)
        self.boton_seleccionar.pack(fill="x", pady=(14, 0))

        # --- Vista de diagnostico: "lo que ve la red" ---
        # Esto no es necesario para que el programa funcione; es una
        # ayuda visual para que el usuario entienda como queda su
        # imagen despues del preprocesamiento (20x20 px en escala de
        # grises), ya que una foto grande y a color se transforma en
        # algo muy distinto antes de llegar a la red. Ayuda a explicar
        # por que el modelo a veces se equivoca con fotos que no se
        # parecen a las del dataset de entrenamiento.
        self.marco_diagnostico = ctk.CTkFrame(contenido_carga, fg_color="transparent")
        self.marco_diagnostico.pack(fill="x", pady=(16, 0))

        ctk.CTkLabel(
            self.marco_diagnostico,
            text="Así ve la red tu imagen (20×20 px):",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLOR_TEXTO_SECUNDARIO
        ).pack(anchor="w")

        fila_diagnostico = ctk.CTkFrame(self.marco_diagnostico, fg_color="transparent")
        fila_diagnostico.pack(anchor="w", pady=(6, 0))

        self.etiqueta_vista_diagnostico = ctk.CTkLabel(
            fila_diagnostico, text="—", width=90, height=90,
            fg_color="#F0F2F5", corner_radius=10,
            font=ctk.CTkFont(size=10), text_color=COLOR_TEXTO_SECUNDARIO)
        self.etiqueta_vista_diagnostico.pack(side="left")

        # --- Columna derecha: analizar + resultado ---
        tarjeta_resultado = Tarjeta(columnas)
        tarjeta_resultado.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        contenido_resultado = ctk.CTkFrame(tarjeta_resultado, fg_color="transparent")
        contenido_resultado.pack(fill="both", expand=True, padx=22, pady=22)

        ctk.CTkLabel(contenido_resultado, text="2. Resultado",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLOR_TEXTO).pack(anchor="w", pady=(0, 10))

        self.boton_analizar = ctk.CTkButton(
            contenido_resultado, text="🔍  Analizar imagen", font=FUENTE_BOTON,
            fg_color=COLOR_PRIMARIO, hover_color=COLOR_PRIMARIO_HOVER,
            corner_radius=10, height=42, state="disabled",
            command=self._analizar_imagen)
        self.boton_analizar.pack(fill="x", pady=(0, 16))

        # Tarjeta interna de resultado (cambia de color segun el diagnostico)
        self.marco_resultado = ctk.CTkFrame(
            contenido_resultado, fg_color="#F7F9FC", corner_radius=14, height=260)
        self.marco_resultado.pack(fill="both", expand=True)
        self.marco_resultado.pack_propagate(False)

        self.icono_resultado = ctk.CTkLabel(
            self.marco_resultado, text="🩺", font=ctk.CTkFont(size=34))
        self.icono_resultado.pack(pady=(38, 6))

        self.etiqueta_resultado = ctk.CTkLabel(
            self.marco_resultado, text="Esperando una imagen...",
            font=ctk.CTkFont(size=15), text_color=COLOR_TEXTO_SECUNDARIO)
        self.etiqueta_resultado.pack(pady=(0, 14))

        self.barra_confianza = ctk.CTkProgressBar(
            self.marco_resultado, width=220, height=10, corner_radius=6,
            progress_color=COLOR_PRIMARIO)
        self.barra_confianza.set(0)

        self.etiqueta_confianza = ctk.CTkLabel(
            self.marco_resultado, text="", font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXTO_SECUNDARIO)

        ctk.CTkLabel(
            contenido_resultado,
            text="Resultado orientativo generado por un modelo académico. "
                 "No reemplaza una evaluación médica profesional.",
            font=ctk.CTkFont(size=10), text_color=COLOR_TEXTO_SECUNDARIO,
            wraplength=340, justify="left").pack(anchor="w", pady=(12, 0))

    def actualizar_disponibilidad(self, hay_modelo):
        self._hay_modelo = hay_modelo
        estado = "normal" if (hay_modelo and self.ruta_imagen_seleccionada) else "disabled"
        self.boton_analizar.configure(state=estado)

    def _seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Selecciona una imagen de ojo",
            filetypes=[("Imagenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos los archivos", "*.*")]
        )
        if not ruta:
            return

        self.ruta_imagen_seleccionada = ruta
        self._mostrar_vista_previa(ruta)
        self._mostrar_vista_diagnostico(ruta)
        self._reiniciar_resultado()
        self.actualizar_disponibilidad(self.ventana.existe_modelo_entrenado())

    def _mostrar_vista_previa(self, ruta):
        try:
            imagen = Image.open(ruta)
            imagen.thumbnail((220, 190))
            self.imagen_ctk = ctk.CTkImage(light_image=imagen, size=imagen.size)

            self.icono_carga.pack_forget()
            self.texto_carga.pack_forget()

            self.etiqueta_previa.configure(image=self.imagen_ctk, text="")
            self.etiqueta_previa.pack(expand=True)
        except Exception as error:
            messagebox.showerror("Error al abrir la imagen", str(error))

    def _mostrar_vista_diagnostico(self, ruta):
        """
        Genera y muestra la version 20x20 (en escala de grises) de la
        imagen seleccionada, usando exactamente el mismo preprocesamiento
        que despues usara la red para predecir (generar_vista_previa_procesada
        llama internamente a la misma funcion que cargar_y_procesar_imagen).
        """
        try:
            imagen_procesada = generar_vista_previa_procesada(
                ruta, tamaño=entrenamiento.TAMAÑO_IMAGEN, factor_ampliacion=8)
            self.imagen_diagnostico_ctk = ctk.CTkImage(
                light_image=imagen_procesada, size=imagen_procesada.size)
            self.etiqueta_vista_diagnostico.configure(
                image=self.imagen_diagnostico_ctk, text="")
        except Exception:
            # Si por algun motivo no se puede generar la vista de
            # diagnostico, no interrumpimos el flujo principal: el
            # usuario igual puede analizar la imagen con normalidad.
            self.etiqueta_vista_diagnostico.configure(image=None, text="—")

    def _reiniciar_resultado(self):
        self.icono_resultado.configure(text="🩺")
        self.etiqueta_resultado.configure(
            text="Listo para analizar", text_color=COLOR_TEXTO_SECUNDARIO)
        self.barra_confianza.pack_forget()
        self.etiqueta_confianza.pack_forget()
        self.marco_resultado.configure(fg_color="#F7F9FC")

    def _analizar_imagen(self):
        if not self.ruta_imagen_seleccionada:
            return

        if not self.ventana.existe_modelo_entrenado():
            messagebox.showwarning(
                "Modelo no entrenado",
                "Todavía no existe un modelo entrenado.\nVe a la sección 'Entrenar' primero.")
            return

        try:
            red = RedNeuronal.cargar_pesos(entrenamiento.RUTA_PESOS)
            vector_imagen = cargar_y_procesar_imagen(
                self.ruta_imagen_seleccionada, tamaño=entrenamiento.TAMAÑO_IMAGEN)
            probabilidad, clase = red.predecir(vector_imagen)
        except Exception as error:
            messagebox.showerror("Error al analizar la imagen", str(error))
            return

        if clase == 1:
            self.marco_resultado.configure(fg_color=COLOR_ALERTA_FONDO)
            self.icono_resultado.configure(text="⚠")
            self.etiqueta_resultado.configure(
                text="CATARATA DETECTADA", text_color=COLOR_ALERTA,
                font=FUENTE_RESULTADO)
            confianza = probabilidad
        else:
            self.marco_resultado.configure(fg_color=COLOR_EXITO_FONDO)
            self.icono_resultado.configure(text="✓")
            self.etiqueta_resultado.configure(
                text="OJO NORMAL", text_color=COLOR_EXITO,
                font=FUENTE_RESULTADO)
            confianza = 1 - probabilidad

        self.barra_confianza.set(confianza)
        self.barra_confianza.pack(pady=(4, 6))
        self.etiqueta_confianza.configure(text=f"Confianza del modelo: {confianza * 100:.1f}%")
        self.etiqueta_confianza.pack()


# ==========================================================================
# PAGINA: ENTRENAR
# ==========================================================================
class PaginaEntrenar(ctk.CTkFrame):
    """
    Permite elegir las carpetas de imagenes y entrenar el modelo sin usar
    la terminal. El entrenamiento corre en un hilo aparte para no
    congelar la ventana; una barra de progreso muestra el avance.
    """

    def __init__(self, contenedor, ventana_principal):
        super().__init__(contenedor, fg_color="transparent")
        self.ventana = ventana_principal
        self.entrenando = False
        self.after(200, self._revisar_cola)

        tarjeta = Tarjeta(self)
        tarjeta.pack(fill="both", expand=True)
        contenido = ctk.CTkFrame(tarjeta, fg_color="transparent")
        contenido.pack(fill="both", expand=True, padx=26, pady=24)

        # --- Carpetas de datos ---
        ctk.CTkLabel(contenido, text="Carpetas de imágenes",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLOR_TEXTO).pack(anchor="w")

        self.var_ruta_normal = ctk.StringVar(value=entrenamiento.RUTA_NORMAL)
        self.var_ruta_catarata = ctk.StringVar(value=entrenamiento.RUTA_CATARATA)

        self._fila_ruta(contenido, "Ojos normales", self.var_ruta_normal)
        self._fila_ruta(contenido, "Ojos con catarata", self.var_ruta_catarata)

        # --- Insignias de configuracion (solo informativas) ---
        ctk.CTkLabel(contenido, text="Configuración del entrenamiento",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLOR_TEXTO).pack(anchor="w", pady=(18, 8))

        marco_badges = ctk.CTkFrame(contenido, fg_color="transparent")
        marco_badges.pack(anchor="w", pady=(0, 20))

        badges_info = [
            f"Épocas: {entrenamiento.EPOCAS}",
            f"Neuronas ocultas: {entrenamiento.N_OCULTAS}",
            f"Tasa de aprendizaje: {entrenamiento.TASA_APRENDIZAJE}",
            f"Tamaño imagen: {entrenamiento.TAMAÑO_IMAGEN}x{entrenamiento.TAMAÑO_IMAGEN}",
        ]
        for texto in badges_info:
            Badge(marco_badges, texto).pack(side="left", padx=(0, 8))

        self.boton_entrenar = ctk.CTkButton(
            contenido, text="⚙  Entrenar modelo", font=FUENTE_BOTON,
            fg_color=COLOR_PRIMARIO, hover_color=COLOR_PRIMARIO_HOVER,
            corner_radius=10, height=42, command=self._iniciar_entrenamiento)
        self.boton_entrenar.pack(anchor="w", pady=(0, 22))

        # --- Progreso ---
        ctk.CTkLabel(contenido, text="Progreso",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLOR_TEXTO).pack(anchor="w")

        self.barra_progreso = ctk.CTkProgressBar(
            contenido, height=16, corner_radius=8, progress_color=COLOR_PRIMARIO)
        self.barra_progreso.set(0)
        self.barra_progreso.pack(fill="x", pady=(10, 8))

        self.etiqueta_progreso = ctk.CTkLabel(
            contenido, text="Sin iniciar.", font=FUENTE_NORMAL,
            text_color=COLOR_TEXTO_SECUNDARIO)
        self.etiqueta_progreso.pack(anchor="w")

    def _fila_ruta(self, padre, etiqueta_texto, variable):
        fila = ctk.CTkFrame(padre, fg_color="transparent")
        fila.pack(fill="x", pady=6)

        ctk.CTkLabel(fila, text=etiqueta_texto, font=FUENTE_NORMAL, width=140,
                     anchor="w", text_color=COLOR_TEXTO_SECUNDARIO).pack(side="left")

        entrada = ctk.CTkEntry(fila, textvariable=variable, font=FUENTE_NORMAL,
                                corner_radius=8, height=34)
        entrada.pack(side="left", fill="x", expand=True, padx=8)

        ctk.CTkButton(
            fila, text="Examinar...", font=FUENTE_NORMAL, width=100, height=34,
            corner_radius=8, fg_color="#F0F2F5", text_color=COLOR_TEXTO,
            hover_color="#E1E6ED",
            command=lambda: self._elegir_carpeta(variable)).pack(side="left")

    def _elegir_carpeta(self, variable):
        carpeta = filedialog.askdirectory(title="Selecciona la carpeta de imágenes")
        if carpeta:
            variable.set(carpeta)

    def _iniciar_entrenamiento(self):
        if self.entrenando:
            return

        ruta_normal = self.var_ruta_normal.get().strip()
        ruta_catarata = self.var_ruta_catarata.get().strip()

        if not os.path.isdir(ruta_normal) or not os.path.isdir(ruta_catarata):
            messagebox.showerror(
                "Carpetas inválidas",
                "Revisa que ambas carpetas de imágenes existan y estén bien escritas.")
            return

        self.entrenando = True
        self.boton_entrenar.configure(state="disabled", text="Entrenando...")
        self.barra_progreso.set(0)
        self.etiqueta_progreso.configure(text="Preparando entrenamiento...")

        hilo = threading.Thread(
            target=self._ejecutar_entrenamiento_en_hilo,
            args=(ruta_normal, ruta_catarata),
            daemon=True
        )
        hilo.start()

    def _ejecutar_entrenamiento_en_hilo(self, ruta_normal, ruta_catarata):
        cola = self.ventana.cola_entrenamiento

        def callback_estado(texto):
            cola.put(("estado", texto))

        def callback_progreso(epoca, total, error):
            cola.put(("progreso", epoca, total, error))

        try:
            entrenamiento.entrenar_red(
                ruta_normal=ruta_normal,
                ruta_catarata=ruta_catarata,
                guardar_resultados=True,
                callback_estado=callback_estado,
                callback_progreso=callback_progreso,
            )
            cola.put(("finalizado_ok", None))
        except Exception as error:
            cola.put(("finalizado_error", str(error)))

    def _revisar_cola(self):
        try:
            while True:
                mensaje = self.ventana.cola_entrenamiento.get_nowait()
                tipo = mensaje[0]

                if tipo == "estado":
                    self.etiqueta_progreso.configure(text=mensaje[1])

                elif tipo == "progreso":
                    _, epoca, total, error = mensaje
                    self.barra_progreso.set(epoca / total)
                    self.etiqueta_progreso.configure(
                        text=f"Época {epoca}/{total}  ·  Error: {error:.5f}")

                elif tipo == "finalizado_ok":
                    self.entrenando = False
                    self.boton_entrenar.configure(state="normal", text="⚙  Entrenar modelo")
                    self.barra_progreso.set(1.0)
                    self.etiqueta_progreso.configure(
                        text="✔ Modelo entrenado y guardado correctamente.")
                    self.ventana._actualizar_estado_modelo()
                    messagebox.showinfo("Entrenamiento finalizado",
                                         "El modelo se entrenó y guardó correctamente.")

                elif tipo == "finalizado_error":
                    self.entrenando = False
                    self.boton_entrenar.configure(state="normal", text="⚙  Entrenar modelo")
                    self.etiqueta_progreso.configure(
                        text="✖ Ocurrió un error durante el entrenamiento.")
                    messagebox.showerror("Error durante el entrenamiento", mensaje[1])

        except queue.Empty:
            pass

        self.after(150, self._revisar_cola)


# ==========================================================================
# PAGINA: EVALUAR
# ==========================================================================
class PaginaEvaluar(ctk.CTkFrame):
    """
    Muestra la matriz de confusion y las metricas del modelo sobre el
    conjunto de prueba, usando evaluacion.evaluar_modelo() tal cual.
    """

    def __init__(self, contenedor, ventana_principal):
        super().__init__(contenedor, fg_color="transparent")
        self.ventana = ventana_principal


        tarjeta = Tarjeta(self)
        tarjeta.pack(fill="both", expand=True)
        contenido = ctk.CTkFrame(tarjeta, fg_color="transparent")
        contenido.pack(fill="both", expand=True, padx=26, pady=24)

        self.boton_evaluar = ctk.CTkButton(
            contenido, text="📊  Evaluar modelo", font=FUENTE_BOTON,
            fg_color=COLOR_PRIMARIO, hover_color=COLOR_PRIMARIO_HOVER,
            corner_radius=10, height=42, command=self._evaluar)
        self.boton_evaluar.pack(anchor="w", pady=(0, 22))

        # --- Cuadricula de 4 tarjetas de metricas ---
        ctk.CTkLabel(contenido, text="Métricas de desempeño",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLOR_TEXTO).pack(anchor="w", pady=(0, 10))

        marco_stats = ctk.CTkFrame(contenido, fg_color="transparent")
        marco_stats.pack(fill="x", pady=(0, 24))
        for i in range(4):
            marco_stats.grid_columnconfigure(i, weight=1, uniform="stats")

        self.tarjeta_exactitud = TarjetaEstadistica(marco_stats, "Exactitud")
        self.tarjeta_precision = TarjetaEstadistica(marco_stats, "Precisión")
        self.tarjeta_recall = TarjetaEstadistica(marco_stats, "Sensibilidad")
        self.tarjeta_f1 = TarjetaEstadistica(marco_stats, "F1-score")

        self.tarjeta_exactitud.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.tarjeta_precision.grid(row=0, column=1, sticky="nsew", padx=6)
        self.tarjeta_recall.grid(row=0, column=2, sticky="nsew", padx=6)
        self.tarjeta_f1.grid(row=0, column=3, sticky="nsew", padx=(6, 0))

        # --- Matriz de confusion ---
        ctk.CTkLabel(contenido, text="Matriz de confusión",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLOR_TEXTO).pack(anchor="w", pady=(0, 10))

        self.marco_matriz = ctk.CTkFrame(contenido, fg_color="transparent")
        self.marco_matriz.pack(anchor="w")

    def actualizar_disponibilidad(self, disponible):
        self.boton_evaluar.configure(state="normal" if disponible else "disabled")

    def _evaluar(self):
        if not (self.ventana.existe_modelo_entrenado() and self.ventana.existe_conjunto_prueba()):
            messagebox.showwarning(
                "Falta información",
                "Primero entrena el modelo en la sección 'Entrenar'.")
            return

        try:
            resultados = evaluacion.evaluar_modelo()
        except Exception as error:
            messagebox.showerror("Error al evaluar el modelo", str(error))
            return

        self.tarjeta_exactitud.actualizar(resultados["exactitud"])
        self.tarjeta_precision.actualizar(resultados["precision"])
        self.tarjeta_recall.actualizar(resultados["recall"])
        self.tarjeta_f1.actualizar(resultados["f1_score"])

        self._pintar_matriz_confusion(resultados["matriz_confusion"])

    def _pintar_matriz_confusion(self, matriz):
        for widget in self.marco_matriz.winfo_children():
            widget.destroy()

        vp, vn = matriz["VP"], matriz["VN"]
        fp, fn = matriz["FP"], matriz["FN"]

        # (texto, valor, es_acierto, es_encabezado)
        celdas = [
            [("", None, False, True), ("Predicho\nNormal", None, False, True),
             ("Predicho\nCatarata", None, False, True)],
            [("Real: Normal", None, False, True), ("VN", vn, True, False),
             ("FP", fp, False, False)],
            [("Real: Catarata", None, False, True), ("FN", fn, False, False),
             ("VP", vp, True, False)],
        ]

        for fila_idx, fila in enumerate(celdas):
            for col_idx, (texto, valor, acierto, encabezado) in enumerate(fila):
                if encabezado:
                    bg = COLOR_BADGE_FONDO
                    contenido_texto = texto
                    color_texto = COLOR_TEXTO
                    fuente = ctk.CTkFont(size=11, weight="bold")
                else:
                    bg = COLOR_EXITO_FONDO if acierto else COLOR_ALERTA_FONDO
                    color_texto = COLOR_EXITO if acierto else COLOR_ALERTA
                    contenido_texto = str(valor)
                    fuente = ctk.CTkFont(size=18, weight="bold")

                celda = ctk.CTkFrame(self.marco_matriz, fg_color=bg, corner_radius=10,
                                      width=140, height=64, border_width=1,
                                      border_color=COLOR_BORDE)
                celda.grid(row=fila_idx, column=col_idx, padx=4, pady=4)
                celda.grid_propagate(False)

                ctk.CTkLabel(celda, text=contenido_texto, font=fuente,
                             text_color=color_texto, justify="center").place(
                    relx=0.5, rely=0.5, anchor="center")


if __name__ == "__main__":
    app = InterfazProyecto()
    app.mainloop()
