# Librerías del framework
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from genericworker import *

# Librerías necesarias
import pyrealsense2 as pr2
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import os

# Librerías red neuronal
import torch 
import torch.nn as nn



# --------------------------

class RedNeuronal(nn.Module):
    def __init__(self, input_shape):
        super(RedNeuronal, self).__init__()

        self.capasCompartidas = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (input_shape[0] // 4) * (input_shape[1] // 4), 64),
            nn.ReLU()
        )

        self.resultado = nn.Linear (64 * 2, 1)
        self.resultadoPorcentaje = nn.Sigmoid ()

    def forward(self, fotograma1, fotograma2):
        resultadoFotograma1 = self.capasCompartidas (fotograma1)
        resultadoFotograma2 = self.capasCompartidas (fotograma2)
        resultadosConcatenados = torch.cat ((resultadoFotograma1, resultadoFotograma2), dim=1)
        resultadoUnico = self.resultado (resultadosConcatenados)
        porcentajeSimilitud = self.resultadoPorcentaje (resultadoUnico)
          
        return porcentajeSimilitud
    
# ----------------------------------

class SpecificWorker(GenericWorker):
    periodo = 33
    
    conexionGrabacion = None
    redNeuronalSimilitud = None
    redNeuronalYOLO = None
    
    
    # Rutas de archivos
    rutaGrabacion = "/media/robocomp/data_tfg/oficialVideos/video1.bag"
    rutaPesosRedNeuronalSimilitud = "/home/robocomp/funciona/Similarity/weightModel.pth"
    
    # Red neuronal YOLO
    PRECISION_MINIMA_ACEPTABLE = 0.75
    
    # Red neuronal obtencion grado similitud
    TAMANO_ENTRADA = (350, 150, 3)
    PORCENTAJE_MINIMO_ACEPTABLE = 0.8
    
    # Flags generales
    NUMERO_DECIMALES = 7
    
    
    # -------------------------------------------------
    
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = self.periodo

        self.comprobacion_requisitos_minimos ()

        # Conexion con el fichero de la grabación
        self.iniciar_conexion_grabacion ()
        
        # Carga y prepara las redes neuronales para ser tratados
        self.iniciar_redes_neuronales ()
        
        self.preparacion_entorno ()

        sys.exit ("Testing")
        # Arranca el timer
        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

    # ----------------
    
    def __del__(self):
        return

    # --------------------------

    def setParams(self, params):

        return

    # ----------------

    @QtCore.Slot()
    def compute(self):
        # Recepcion de fotograma junto con un flag que indica si se ha recibido o no.
        hayFotograma, fotograma = self.conexionGrabacion.try_wait_for_frames ()
        
        if hayFotograma:
            fotogramaColor, fotogramaProfundidad = self.preparacion_fotogramas (fotograma)    
                
            resultados = self.redNeuronalYOLO (fotogramaColor)
            
            listaCajasColisiones = self.filtrar_cajas_de_colisiones (resultados)
            
            indiceObjetivo = self.eleccion_usuario_objetivo (listaCajasColisiones)
                    
            self.interfaz_usuario (fotogramaColor)
            
        print ("Iteraccion")

        return
    
    # -------------------------------------
    
    def comprobacion_requisitos_minimos (self):
        # Comprobacion de que existe la ruta de la grabacion
        if not os.path.exists (self.rutaGrabacion):
            sys.exit ("ERROR (1): La ruta de la grabacion no existe. Ruta: " + self.rutaGrabacion)
        
        # Comprobacion de que existe la ruta de los pesos de la red neuronal    
        if not os.path.exists (self.rutaPesosRedNeuronalSimilitud):
            sys.exit ("ERROR (2): La ruta de los pesos de red neuronal no existe. Ruta: " + self.rutaPesosRedNeuronalSimilitud)
        
        return
    
    # ------------------------------------
    
    def iniciar_conexion_grabacion (self):
        # Se generan los objetos de conexion y configuracion
        self.conexionGrabacion = pr2.pipeline ()
        configuracion = pr2.config ()
        
        # Se configura para leer desde un archivo
        configuracion.enable_device_from_file (self.rutaGrabacion, repeat_playback=False)
        
        # Se inicia la conexion con la configuracion predefinida
        self.conexionGrabacion.start (configuracion)
        
        return
    
    # ----------------------------------
    
    def iniciar_redes_neuronales (self):
        # Crea la red neuronal y carga los pesos desde el archivo (Pesos entrenados)
        redNeuronalSimilitud = RedNeuronal (self.TAMANO_ENTRADA)
        redNeuronalSimilitud.load_state_dict(torch.load(self.rutaPesosRedNeuronalSimilitud))
        
        # Carga de red neuronal YOLO (Intrinsecamente se cargan los pesos pertinentes)
        self.redNeuronalYOLO = YOLO("yolov8s.pt")
        
        return
    
    # ---------------------------
    
    def preparacion_entorno (self):
        while True:
            # Recepcion de fotograma junto con un flag que indica si se ha recibido o no.
            hayFotograma, fotograma = self.conexionGrabacion.try_wait_for_frames ()
            
            if hayFotograma:
                fotogramaColor = self.preparacion_fotogramas (fotograma)  
                
                resultados = self.redNeuronalYOLO (fotogramaColor, verbose=False) 
                
                listaCajasColisiones = self.filtrar_cajas_de_colisiones (resultados)
                
                indiceObjetivo = self.obtencion_indice_objetivo_manual (fotogramaColor, listaCajasColisiones)
                
                self.interfaz_usuario (fotogramaColor, listaCajasColisiones, indiceObjetivo)
                
                if indiceObjetivo != -1:
                    break
                
        return
    
    # -------------------------------------------
    
    def preparacion_fotogramas (self, fotograma):
        # Extraccion de los fotogramas y sus datos y conversion a tipo array de numpy para su tratamiento y gestion
        fotogramaColor = np.asarray (fotograma.get_color_frame ().get_data ())
        #fotogramaProfundidad = np.asarray (fotograma.get_depth_frame ().get_data ())
        
        #return fotogramaColor, fotogramaProfundidad
        return fotogramaColor
    
    # -------------------------------------------------
    
    def filtrar_cajas_de_colisiones (self, resultados):
        # Se crean listas vacias para guardar la información
        listaCajaColisionesDetecciones = []

        # Se separan los resultados
        for deteccion in resultados[0].boxes:

            # Si es una persona los resultados se tienen que guardar. Si no, no interesan (Mejora la eficiencia)
            if deteccion.cls == 0 and deteccion.conf > self.PRECISION_MINIMA_ACEPTABLE:

                # Para las cajas de colision son 4 valores en lugar de uno
                listaCajaColisionesDetecciones.append ([int(coordenada.item()) for coordenada in deteccion.xyxy.to('cpu')[0]]) 

        return listaCajaColisionesDetecciones
    
    # -------------------------------
    
    def eleccion_usuario_objetivo (self, fotogramaOriginal, fotogramaObjetivoAnterior, listaCajasColisiones):
        indiceObjetivo = -1
        maximaSimilitud = 0
        contador = 0
        
        # Para cada caja de colision
        for cajaColision in listaCajasColisiones:
            # Se obtiene la region de interes (Parte de la imagen original)
            regionInteres = fotogramaOriginal[cajaColision[1]:cajaColision[3], cajaColision[0]:cajaColision[2]]

            regionInteresRedimensionado = cv.resize (regionInteres, (self.TAMANO_ENTRADA[1], self.TAMANO_ENTRADA[0]))
            
            resultado = self.redNeuronalSimilitud (fotogramaObjetivoAnterior, regionInteresRedimensionado)
            
            # Si supera el minimo aceptable y mejora al anterior entonces hay nuevo objetivo
            if resultado > maximaSimilitud and resultado > self.PORCENTAJE_MINIMO_ACEPTABLE:
                indiceObjetivo = contador
                maximaSimilitud = resultado
                
            contador += 1
                
        return indiceObjetivo, maximaSimilitud
    
    # ------------------------------------------
    
    def interfaz_usuario (self, fotogramaOriginal, listaCajaColisiones, indicePersonaObjetivo):
        # Primero se dibujan las bounding boxes sobre la imagen
        fotogramaConDetecciones = fotogramaOriginal.copy ()

        # Se hace manualmente ya que la opcion que ofrece la librería muestra todas las cajas de colisiones y solo interesan las personas
        for i in range (len (listaCajaColisiones)):
            # Se asigna un color distinto dependiendo si la persona es la objetivo o no
            if indicePersonaObjetivo == i:
                text = "Target"
                color = (0, 255, 0)
            else:
                text = "No target"
                color = (0, 0, 255)

            # Se dibuja un rectangulo simulando la bounding box (Verde si es la persona objetivo y roja si no)
            cv.rectangle (fotogramaConDetecciones, 
                          (listaCajaColisiones[i][0], listaCajaColisiones[i][1]),
                          (listaCajaColisiones[i][2], listaCajaColisiones[i][3]),
                          color,
                          2
                          )
            
            cv.putText (fotogramaConDetecciones, 
                        text, 
                        (listaCajaColisiones[i][0], listaCajaColisiones[i][1] - 10), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2)
            
        cv.imshow ("Fotograma original", fotogramaOriginal)
        cv.imshow ("Fotograma con detecciones", fotogramaConDetecciones)

        # Se le asigna la espera minima de 1ms ya que interesa la fluidez
        self.controlador_teclas (letraPulsada=cv.waitKey (1))

        return

    # ------------------------------------------

    def controlador_teclas (self, letraPulsada):
        # Si el valor es -1 siginifca que no se pulso ninguna tecla (No merece la pena hacer ninguna comprobacion)
        if letraPulsada != -1:

            # Se ha pulsado la letra ESC
            if letraPulsada == 27:
                sys.exit ("FIN EJECUCION")
                
            elif letraPulsada == 13:
                return True
                
            # Es escalable (Usando la siguiente estructura)
            #elif letraPulsada == <codigo letra>:
                # Codigo si se pulsa la tecla
        
        return False
    
    def obtencion_indice_objetivo_manual (self, fotograma, listaCajasColisiones):
        print ("AVISO (1): Para el primer fotograma cuando se muestre la region de interes pertinente presione la siguiente tecla")
        print ("\tPresione ENTER cuando vea a la persona objetivo. Si no, pulse cualquiera")

        indiceObjetivo = -1
        contadorDetecciones = 0

        for cajaColision in listaCajasColisiones:
            regionInteres = fotograma[cajaColision[1]:cajaColision[3], cajaColision[0]:cajaColision[2]]

            cv.imshow ("Region interes", regionInteres)

            if self.controlador_teclas (letraPulsada=cv.waitKey (0)):
                indiceObjetivo = contadorDetecciones
                self.fotogramaOobjetivoAnterior = regionInteres
                break

            contadorDetecciones += 1

        # Libera los recursos para no tener tantas ventanas de opencv y diferenciar
        cv.destroyWindow("Region interes")

        return indiceObjetivo