from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from genericworker import *

# Importacion de librerías
import pyrealsense2 as pr2              # Gestion de camara
import cv2 as cv                        # Interfaz de usuario
import numpy as np                      # Gestion de imagenes
import json                             # Guardado de resultados
import sys                              # Control de errores controlado
import os                               # Gestion de rutas, archivos y directorios
import time                             # Para medir tiempos de ejecucion

class SpecificWorker(GenericWorker):

    periodo = 33            # A 30 FPS entonces 1000/30 = 33,... -> 33

    # Conexion con camara
    conexion = None

    rutaDestinoGrabacion = "/home/robocomp/prueba.bag"
    rutaDestinoResultadosJSON = "/home/robocomp/data.json"

    # Flags
    REEMPLAZAR_GRABACION_EXISTENTE = True
    NUMERO_SERIE_CAMARA = "146222252950"
    RESOLUCION_GRABACION = [640, 480]               # Anchura y altura de los fotogramas
    TASA_FOTOGRAMAS_SEGUNDO = 30

    # Flags modificables por codigo
    CONEXION_ESTABLECIDA = False

    # Variables para generacion de resultados
    contadorFotogramas = None
    tiempo_conexion = None
    motivo_finalizacion_video = None


    # CONSTRUCTOR, DESTRUCTOR Y METODOS PRINCIPALES
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        
        # Se comprueba que se cumplan los requisitos minimos
        self.comprobacion_requisitos_minimos ()

        # Se inicia la conexion con la cámara y el flujo de datos
        self.iniciar_conexion_camara_y_grabacion ()

        # Se activa el temporizador
        self.timer.timeout.connect(self.compute)
        self.timer.start(self.periodo)
        return

	# ----------------

    def __del__(self):
        if self.CONEXION_ESTABLECIDA:
            self.conexion.stop ()

        return

	# ----------------
	
    @QtCore.Slot()
    def compute(self):
        print ("Computing")

        # Realiza una solicitud (No bloqueante) a la cámara por nuevos datos (Fotogramas)
        fotogramasDisponibles, conjuntoFotogramas = self.conexion.try_wait_for_frames ()

        if fotogramasDisponibles:
            # Se muestra al usuario el contenido (SIEMPRE)
            self.interfaz_usuario (conjuntoFotogramas)

            # Se incrementa el contador de fotogramas
            self.contadorFotogramas += 1

        else:
            sys.exit ("ERROR (3): No se estan recibiendo fotogramas correctamente")
        return


    # METODOS SECUNDARIOS DE APOYO A LOS PRINCIPALES
    def comprobacion_requisitos_minimos (self):

        # Comprueba si existe un fichero en la ruta y no esta habilitado el reemplazo
        if os.path.exists (self.rutaDestinoGrabacion) and not self.REEMPLAZAR_GRABACION_EXISTENTE:
            sys.exit ("ERROR (1): Ya existe una grabación en la ruta " + self.rutaDestinoGrabacion)

        # Se comprueba que el directorio de la ruta existe
        if not (os.path.isdir (os.path.dirname (self.rutaDestinoGrabacion))):
            sys.exit ("ERROR (2): El directorio " + os.path.dirname (self.rutaDestinoGrabacion) + " no existe")

        # Separa el nombre del archivo en el nombre y la extension
        tupla_nombre_extension = os.path.splitext(os.path.basename (self.rutaDestinoGrabacion)) 

        if not tupla_nombre_extension[1][1:] == "bag":
            sys.exit ("ERROR (2): El nombre del archivo " + os.path.basename (self.rutaDestinoGrabacion) + " no tiene extension .bag")
        
        print ("INFORMACION (1) -> El primer paso se ha completado y los requisitos minimos se cumplen")

        return
    
    # ---------------------------------------------

    def iniciar_conexion_camara_y_grabacion (self):
        # Iniciamos los objetos por defecto de la librería
        self.conexion = pr2.pipeline ()
        configuracion = pr2.config ()

        # Se genera una conexion con la cámara (Activa el dispositivo dentro de la configuracion)
        configuracion.enable_stream (self.NUMERO_SERIE_CAMARA)

        # Se establecen unas configuraciones para los flujos de datos
        configuracion.enable_stream (pr2.stream.color,                  # Flujo de fotogramas de color
                                     self.RESOLUCION_GRABACION[0], 
                                     self.RESOLUCION_GRABACION[1], 
                                     pr2.format.bgr8,
                                     self.TASA_FOTOGRAMAS_SEGUNDO
                                     )

        configuracion.enable_stream (pr2.stream.depth,                  # Flujo de fotogramas de profundidad
                                     self.RESOLUCION_GRABACION[0], 
                                     self.RESOLUCION_GRABACION[1], 
                                     pr2.format.z16,
                                     self.TASA_FOTOGRAMAS_SEGUNDO
                                     )
        
        # Se indica que se quiere guardar en el fichero indica el contenido.
        configuracion.enable_record_to_file (self.rutaDestinoGrabacion)      

        # Finalmente se activa el flujo
        self.conexion.start (configuracion)
        
        # Modificacion de flags y valores
        self.CONEXION_ESTABLECIDA = True
        self.tiempo_conexion = time.time ()

        # Mensaje para el usuario
        print ("INFORMACION (2) -> El segundo paso se ha completado y se activado el flujo con la cámara correctamente")

        return

    # ----------------------------------------------

    def interfaz_usuario (self, conjuntoFotogramas):
        # Se obtiene el par de frames separado del conjunto recibido previamente
        fotogramaColor = conjuntoFotogramas.get_color_frame().get_data ()
        fotogramaProfundidad = conjuntoFotogramas.get_depth_frame().get_data ()

        # Se convierten en array para poder mostrarlos por la interfaz de opencv
        fotogramaColorArray = np.asarray (fotogramaColor)
        fotogramaProfundidadArray = np.asarray (fotogramaProfundidad)

        # Se muestran por la interfaz de opencv cuyas ventanas tienen asignadas el nombre indicado
        cv.imshow ("Fotogramas Color", fotogramaColorArray)
        cv.imshow ("Fotogramas Profundidad", fotogramaProfundidadArray)

        # Se le asigna la espera minima de 1ms ya que interesa la fluidez
        self.controlador_teclas (cv.waitKey (1))

        return

    # ------------------------------------------

    def controlador_teclas (self, letraPulsada):
        # Si el valor es -1 siginifca que no se pulso ninguna tecla (No merece la pena hacer ninguna comprobacion)
        if letraPulsada != -1:

            # Se ha pulsado la letra ESC
            if letraPulsada == 27:
                self.generacion_de_resultados ()
                sys.exit ("FIN EJECUCION: Revise el video " + self.rutaDestinoGrabacion + " para confirmar que todo está correcto")

            # Es escalable (Usando la siguiente estructura)
            #elif letraPulsada == <codigo letra>:
                # Codigo si se pulsa la tecla
        
        return

    # ----------------------------------

    def generacion_de_resultados (self):
        # Calculo final
        self.tiempo_conexion = time.time() - self.tiempo_conexion

        # Creacion de diccionario para guardar la informacion en formato JSON
        resultadoFormatoJSON = {"Tiempo de conexion" : self.tiempo_conexion,
                                "Numero de frames guardados" : (self.contadorFotogramas - 1),
                                "Ruta destino de la grabacion" : self.rutaDestinoGrabacion}
        
        # Resultados por consola (Temporales)
        print ("\n\n----------- INICIO RESULTADOS -----------")
        print ("Tiempo de conexion:", self.tiempo_conexion, " segundos")
        print ("Numero de fotogramas guardados:", (self.contadorFotogramas - 1), "fotogramas")
        print ("Ruta destino de la grabacion:", self.rutaDestinoGrabacion)
        print ("------------ FIN RESULTADOS ------------\n\n")

        # Resultados por archivo JSON (Permanentes)
        if os.exists (os.path.dirname (self.rutaDestinoResultadosJSON)):
            with open (self.rutaDestinoResultadosJSON, 'w') as flujoSalida:
                json.dump (resultadoFormatoJSON, flujoSalida)

        return

	# --------------------------

    def setParams(self, params):
        self.contadorFotogramas = 0                 # Se encargara de contar cuantos frames se estan guardando en el video
        self.motivo_finalizacion_video = -1     # Valor ilogico (Si se obtiene en los resultados un -1 es que algo fue mal)

        return True
