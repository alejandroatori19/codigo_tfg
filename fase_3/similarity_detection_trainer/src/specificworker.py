from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from genericworker import *

sys.path.append('/opt/robocomp/lib')

# Importaciones de librerias
import cv2 as cv                        # Interfaz de usuario
import numpy as np                      # Gestion de imagenes
import json                             # Guardado de resultados
import sys                              # Control de errores controlado
import time                             # Para medir tiempos de ejecucion
import os                               # Para gestion de rutas de ficheros
import random

# Librería pythorch
import torch
import torch.nn as nn
import torch.optim as optim


class SpecificWorker(GenericWorker):

    periodo = 20

    redNeuronalDeteccionSimilaridad = None

    rutaDatasetClasificado = "/media/robocomp/data_tfg/dataset_clasificado"
    rutaDestinoRedEntrenada = "/home/robocomp"

    listaFotogramasObjetivo = []
    listaFotogramasNoObjetivo = []


    TAMANO_ENTRADA = [350, 150, 3]

    DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------

    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = self.periodo

        # Comprobacion inicial de requirimientos minimos
        self.comprobacion_requisitos_minimos ()
        self.preparacion_entorno ()

        # Activacion del timer
        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

        return

    # ----------------

    def __del__(self):

        return

    # --------------------------

    def setParams(self, params):

        return True

    # ----------------

    @QtCore.Slot()
    def compute(self):
        print('SpecificWorker.compute...')


        return True

    # METODOS SECUNDARIOS DE APOYO A LOS PRINCIPALES

    # -----------------------------------------

    def comprobacion_requisitos_minimos (self):
        # Comprueba que existe la carpeta con el contenido del dataset sin clasificar
        if not os.path.exists (self.rutaDatasetClasificado):
            sys.exit ("ERROR (1): El directorio que contiene el dataset clasificado no existe - Ruta: " + self.rutaDatasetClasificado + "\n\n")
        
        
        # Comprueba que existe la ruta con el destino del dataset clasificado (El directorio padre)
        if not os.path.exists (self.rutaDestinoRedEntrenada):
            sys.exit ("ERROR (2): Comprueba que el directorio de la ruta destino del modelo existe - Ruta: " + self.rutaDestinoRedEntrenada + "\n\n")
        
        """
        # Si no esta habilitado el reemplazo entonces acaba (Es un control de seguridad para evitar posibles errores)
        if not self.REEMPLAZAR_DATASET_EXISTENTE : 
            sys.exit ("ERROR (3): No esta habilitado el reemplazo de dataset\n\n")

        # Actualizacion de la ruta para asignarle el nombre del hijo (Carpeta que contendrá el dataset clasificado)
        self.rutaDestinoDatasetClasificado = self.rutaDestinoDatasetClasificado + "/dataset_clasificado"
        """

        print ("INFORMACION (1) -> El primer paso se ha completado y los requisitos minimos se cumplen")
    
        return
    
    # -----------------------------

    def preparacion_entorno (self):
        # Generacion del modelo de red neuronal
        self.redNeuronalDeteccionSimilaridad = RedNeuronal(self.TAMANO_ENTRADA).to(self.DISPOSITIVO)

        # Carga de rutas de archivos para el dataset (Persona Objetivo)
        rutaCarpeta = self.rutaDatasetClasificado + "/persona_objetivo"
        for nombreArchivo in os.listdir (rutaCarpeta):
            self.listaFotogramasObjetivo.append (rutaCarpeta + "/" + nombreArchivo)

        # Carga de rutas de archivos para el dataset (Persona No Objetivo)
        rutaCarpeta = self.rutaDatasetClasificado + "/persona_no_objetivo"
        for nombreArchivo in os.listdir (rutaCarpeta):
            self.listaFotogramasNoObjetivo.append (rutaCarpeta + "/" + nombreArchivo)

        print ("INFORMACION (2) -> El segundo paso se ha completado y el entorno esta preparado")
        
        return

    # ----------------------------------------



    # -----------------------------------------------------------------------------------------

    def interfaz_usuario (self, fotograma1, fotograma2, porcentajeSimilitud):
        print ("Porcentaje Similitud:", porcentajeSimilitud)


        cv.imshow ("Fotograma 1", fotograma1)
        cv.imshow ("Fotograma 2", fotograma2)

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
                sys.exit ("FIN EJECUCION (2): Revise el dataset" + self.rutaDestinoDatasetClasificado + "por parada manual del proceso.")
            
            # Es escalable (Usando la siguiente estructura)
            #elif letraPulsada == <codigo_letra>:
                # Actuacion  
                  
        return

    # ----------------------------------

    def generacion_de_resultados (self):
        
        # Calculo final
        self.tiempoEjecucion = time.time() - self.tiempoEjecucion

        # Creacion de diccionario para guardar la informacion en formato JSON
        resultadoFormatoJSON = {"Tiempo de conexion" : self.tiempoEjecucion,
                                "Numero de frames guardados" : (self.contadorFotogramas - 1),
                                "Ruta destino de la grabacion" : self.rutaDestinoDatasetClasificado
                                }
        
        # Resultados por consola (Temporales)
        print ("\n\n----------- INICIO RESULTADOS -----------")
        print ("Tiempo de conexion:", self.tiempoEjecucion, " segundos")
        print ("Numero de fotogramas guardados:", (self.contadorFotogramas), "fotogramas")
        print ("Ruta destino de la grabacion:", self.rutaDestinoDatasetClasificado)
        print ("------------ FIN RESULTADOS ------------\n\n")

        # Resultados por archivo JSON (Permanentes)
        if os.path.exists (os.path.dirname (self.rutaDestinoResultadosJSON)):
            with open (self.rutaDestinoResultadosJSON, 'w') as flujoSalida:
                json.dump (resultadoFormatoJSON, flujoSalida)
        
        return






















    
# -----------------------------------

class RedNeuronal(nn.Module):
    def __init__(self, tamano_entrada):
        super(RedNeuronal, self).__init__()

        self.capasCompartidas = nn.Sequential(
            nn.Conv2d(tamano_entrada[2], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (tamano_entrada[0] // 4) * (tamano_entrada[1] // 4), 64),
            nn.ReLU()
        )

        # Transforma la salida en un solo valor
        self.salida = nn.Linear(64 * 2, 1)

        # Convierte el valor unico en un porcentaje entre 0 y 1
        self.salidaPorcentaje = nn.Sigmoid()

        return

    def predecir_resultado(self, imagen1, imagen2):
        resultadoImagen1 = self.capasCompartidas(imagen1)
        resultadoImagen2 = self.capasCompartidas(imagen2)
        resultadosConcatenados = torch.cat((resultadoImagen1, resultadoImagen2), dim=1)
        resultadoValorUnico = self.salida(resultadosConcatenados)
        resultadoPorcentaje = self.salidaPorcentaje(resultadoValorUnico)
          
        return resultadoPorcentaje