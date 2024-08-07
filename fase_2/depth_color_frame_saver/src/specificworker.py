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
import shutil

class SpecificWorker(GenericWorker):

    periodo = 15            # A 30 FPS entonces 1000/30 = 33,... -> 33

    # Conexion con camara
    conexion = None

    rutaGrabacion = "/media/robocomp/data_tfg/oficialVideos/video1.bag"                          # Obligatoriamente un archivo con extension .bag
    rutaDestinoFotogramas = "/media/robocomp/data_tfg"           # Debe ser un directorio

    rutaDestinoResultadosJSON = "/home/robocomp/data.json"              # Un archivo con extension .json

    # Flags
    PREVISUALIZACION_VIDEO = False
    REEMPLAZAR_DATASET_EXISTENTE = True

    # Flags modificables por codigo
    CONEXION_ESTABLECIDA = False

    # Variables para generacion de resultados
    contadorFotogramas = None
    tiempo_ejecucion = None

    # CONSTRUCTOR, DESTRUCTOR Y METODOS PRINCIPALES
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        
        # Se comprueba que se cumplan los requisitos minimos
        self.comprobacion_requisitos_minimos ()

        # Se inicia la conexion con la cámara y el flujo de datos
        self.iniciar_conexion_con_grabacion ()

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
        
        # Realiza una solicitud (No bloqueante) a la cámara por nuevos datos (Fotogramas)
        fotogramasDisponibles, conjuntoFotogramas = self.conexion.try_wait_for_frames ()

        if fotogramasDisponibles:
            fotogramaColor, fotogramaProfundidad = self.prepracion_fotogramas (conjuntoFotogramas)

            # Se muestra al usuario el contenido (SIEMPRE)
            self.interfaz_usuario (fotogramaColor, fotogramaProfundidad)

            if self.PREVISUALIZACION_VIDEO:
                self.guardar_frames_en_disco (fotogramaColor, fotogramaProfundidad)

            # Se incrementa el contador de fotogramas
            self.contadorFotogramas += 1

        else:
            self.generacion_de_resultados ()
            sys.exit ("FIN EJECUCION: El video se ha procesado correctamente. Dataset guardado en la ruta " + self.rutaDestinoFotogramas + "\n\n")   
        
        return


    # METODOS SECUNDARIOS DE APOYO A LOS PRINCIPALES
    def comprobacion_requisitos_minimos (self):

        # Separa el nombre del archivo en el nombre y la extension
        tupla_nombre_extension = os.path.splitext(os.path.basename (self.rutaGrabacion)) 

        # Comprueba si existe el archivo de grabacion y tiene extension .bag
        if not (os.path.exists (self.rutaGrabacion) and tupla_nombre_extension[1][1:] == "bag"):
            sys.exit ("ERROR (1): Comprueba que la ruta de grabacion existe y tiene extension .bag - Ruta: " + self.rutaGrabacion + "\n\n")

        # Si se lleva a cabo una previsualización solo es necesario que sea correcto el path de grabacion 
        #(Ahorra dos condicionales y 4 ejeucciones complejas)
        if not self.PREVISUALIZACION_VIDEO:

            # Comprueba si existe el directorio donde se va a guardar el dataset (Conjunto de frames)
            if not os.path.exists (self.rutaDestinoFotogramas):
                sys.exit ("ERROR (2): Comprueba que el directorio padre de la ruta destino de frames existe - Ruta: " + self.rutaDestinoFotogramas + "\n\n")

            self.rutaDestinoFotogramas = self.rutaDestinoFotogramas + "/dataset_sin_clasificar"

            # Si no esta habilitado el reemplazo entonces acaba (Es un control de seguridad para evitar posibles errores)
            if not self.REEMPLAZAR_DATASET_EXISTENTE : 
                sys.exit ("FIN EJECUCION: No esta habilitado el reemplazo\n\n")
            
            if os.path.exists (self.rutaDestinoFotogramas):
                shutil.rmtree(self.rutaDestinoFotogramas)

            os.makedirs (self.rutaDestinoFotogramas)
            os.makedirs (self.rutaDestinoFotogramas + "/fotogramas_color")
            os.makedirs (self.rutaDestinoFotogramas + "/fotogramas_profundidad")
        
        print ("INFORMACION (1) -> El primer paso se ha completado y los requisitos minimos se cumplen\n\n")

        return
    
    # ---------------------------------------------

    def iniciar_conexion_con_grabacion (self):
        # Iniciamos los objetos por defecto de la librería
        self.conexion = pr2.pipeline ()
        configuracion = pr2.config ()

        # Se indica que se va a recibir la informacion de un fichero y sin playback (Cuando llegue al final acaba)
        configuracion.enable_device_from_file (self.rutaGrabacion, repeat_playback=False)

        # Se abren todos los streams que tenga el archivo disponibles (El de color y el de profundidad).
        configuracion.enable_all_streams ()

        # Se inicia el flujo de datos con la configuracion predefinida
        self.conexion.start (configuracion)

        # Inicio del tiempo de ejecucion
        self.tiempo_ejecucion = time.time ()

        # Mensaje para el usuario
        print ("INFORMACION (2) -> El segundo paso se ha completado y se activado el flujo con la grabacion correctamente\n\n")

        return

    # ---------------------------------------------------

    def prepracion_fotogramas (self, conjuntoFotogramas):
        # Se extrae la información que se encuentra aglutinada.
        fotogramaColor = conjuntoFotogramas.get_color_frame().get_data ()
        fotogramaProfundidad = conjuntoFotogramas.get_depth_frame().get_data ()

        # Se convierten en array para poder mostrarlos por la interfaz de opencv
        fotogramaColorArray = np.asarray (fotogramaColor)
        fotogramaProfundidadArray = np.asarray (fotogramaProfundidad)

        return fotogramaColorArray, fotogramaProfundidadArray

    # ----------------------------------------------

    def interfaz_usuario (self, fotogramaColor, fotogramaProfundidad):
        # Si se le desea aplicar una escala de colores
        #fotogramaProfundidad = cv.applyColorMap(cv.convertScaleAbs(fotogramaProfundidad, alpha=0.03), cv.COLORMAP_JET)

        # Se muestran por la interfaz de opencv cuyas ventanas tienen asignadas el nombre indicado
        cv.imshow ("Fotogramas Color", fotogramaColor)
        cv.imshow ("Fotogramas Profundidad", fotogramaProfundidad)

        # Se le asigna la espera minima de 1ms ya que interesa la fluidez
        self.controlador_teclas (letraPulsada=cv.waitKey (1))

        return

    # ------------------------------------------

    def controlador_teclas (self, letraPulsada):
        # Si el valor es -1 siginifca que no se pulso ninguna tecla (No merece la pena hacer ninguna comprobacion)
        if letraPulsada != -1:

            # Se ha pulsado la letra ESC
            if letraPulsada == 27:
                self.generacion_de_resultados ()
                sys.exit ("FIN EJECUCION: Revise el dataset" + self.rutaDestinoFotogramas + "por parada manual del proceso. \n\n")

            # Es escalable (Usando la siguiente estructura)
            #elif letraPulsada == <codigo letra>:
                # Codigo si se pulsa la tecla
        
        return

    # ---------------------------------

    def guardar_frames_en_disco (self, fotogramaColor, fotogramaProfundidad):
        # Se crean las rutas a parte para poder tener un control sobre ellas (Para testing mas que nada)
        rutaFotogramaColor = self.rutaDestinoFotogramas + "/fotogramas_color/frame_" + str(self.contadorFotogramas + 1) + ".jpeg"
        rutaFotogramaProfundidad = self.rutaDestinoFotogramas + "/fotogramas_profundidad/frame_" + str(self.contadorFotogramas + 1) + ".jpeg"

        # Con las siguientes dos lineas se prepara la imagen para escala de grises.
        #fotogramaProfundidad = cv.normalize(fotogramaProfundidad, None, 0, 255, cv.NORM_MINMAX)
        #fotogramaProfundidad = np.uint8(fotogramaProfundidad)

        # Se guardan en la ruta designada
        cv.imwrite (rutaFotogramaColor, fotogramaColor)
        cv.imwrite (rutaFotogramaProfundidad, fotogramaProfundidad)

        return

    # ----------------------------------

    def generacion_de_resultados (self):
        # Calculo final
        self.tiempo_ejecucion = time.time() - self.tiempo_ejecucion

        # Creacion de diccionario para guardar la informacion en formato JSON
        resultadoFormatoJSON = {"Tiempo de conexion" : self.tiempo_ejecucion,
                                "Numero de frames guardados" : (self.contadorFotogramas - 1),
                                "Ruta destino de la grabacion" : self.rutaDestinoFotogramas}
        
        # Resultados por consola (Temporales)
        print ("\n\n----------- INICIO RESULTADOS -----------")
        print ("Tiempo de conexion:", self.tiempo_ejecucion, " segundos")
        print ("Numero de fotogramas guardados:", (self.contadorFotogramas - 1), "fotogramas")
        print ("Ruta destino de la grabacion:", self.rutaDestinoFotogramas)
        print ("------------ FIN RESULTADOS ------------\n\n")

        # Resultados por archivo JSON (Permanentes)
        if os.path.exists (os.path.dirname (self.rutaDestinoResultadosJSON)):
            with open (self.rutaDestinoResultadosJSON, 'w') as flujoSalida:
                json.dump (resultadoFormatoJSON, flujoSalida)

        else:
            print ("ADVERTENCIA (1): El JSON no se pudo imprimir debido a que el directorio donde esta situado no existe.\n\n")

        return

	# --------------------------

    def setParams(self, params):
        self.contadorFotogramas = 0                 # Se encargara de contar cuantos frames se estan guardando en el video

        return True
