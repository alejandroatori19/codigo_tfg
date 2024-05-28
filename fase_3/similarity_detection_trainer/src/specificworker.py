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
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset


class SpecificWorker(GenericWorker):

    periodo = 20

    redNeuronalDeteccionSimilaridad = None
    funcionPerdida = None
    optimizador = None

    # Las siguientes tres constantes se pueden modificar para intentar mejorar el rendimiento dela red (Los valores asignados son los recomendados)
    NUMERO_EPOCAS = 10
    NUMERO_IMAGENES_POR_EPOCA = 50
    TASA_APRENDIZAJE = 0.001

    # Lo siguiente es para validar el modelo si se desea
    NUMERO_IMAGENES_VALIDACION = 10           # Se recomienda poner de 5 a 10 veces menos que las que se asignen para entrenar

    # Guarda las perdidas por cada epoca
    listaPerdidasPorEpocas = []

    rutaDatasetClasificado = "/media/robocomp/data_tfg/dataset_clasificado"
    rutaDestinoRedEntrenada = "/home/robocomp"

    listaFotogramasObjetivo = []
    listaFotogramasNoObjetivo = []


    TAMANO_ENTRADA = [350, 150, 3]
    DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VALIDACION_ACTIVA = False
    CONVERSOR = transforms.Compose([
        transforms.ToTensor(),  # Convertir la imagen a un tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Assuming ImageNet normalization
    ])


    contadorEpocas = None
    contadorImagenes = None
    contadorPerdidaEpoca = None

    # -------------------------------------------------

    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = self.periodo

        # Comprobacion inicial de requirimientos minimos
        self.comprobacion_requisitos_minimos ()

        # Se prepara el entorno de la fase (Creacion de red neuronal y carga de rutas de archivos)
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

        self.contadorEpocas = 0
        self.contadorImagenes = 0
        self.contadorPerdidaEpoca = 0

        return True

    # ----------------

    @QtCore.Slot()
    def compute(self):
        # Si la validacion no esta activa significa que se esta entrenando
        if not self.VALIDACION_ACTIVA:

            # En el entrenamiento se tienen que procesar n frames (Definido en variable global)
            if self.contadorImagenes < self.NUMERO_IMAGENES_POR_EPOCA:

                # Eleccion de dos imagenes aleatoriamente (2 de objetivo o 1 del objetivo y 1 del no objetivo)
                persona1, persona2, resultadoSimilitud = self.seleccion_imagenes_aleatoriamente ()
                
                self.entrenamiento_red_neuronal_una_imagen (persona1, persona2, resultadoSimilitud)

                self.interfaz_usuario (persona1, persona2, resultadoSimilitud)
                    
            # Significa que ya proceso entrenando los n frames predefinidos
            else:
                # Se inicia la validacion despues del entrenamiento
                self.VALIDACION_ACTIVA = True

                # Reset de variables
                self.contadorImagenes = 0

        # Si entra aqui es que esta validando la epoca
        else:
            a = 0

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
        # Carga de rutas de archivos para el dataset (Persona Objetivo)        
        rutaCarpeta = self.rutaDatasetClasificado + "/persona_objetivo"

        for nombreArchivo in os.listdir (rutaCarpeta):
            self.listaFotogramasObjetivo.append (rutaCarpeta + "/" + nombreArchivo)

        # Carga de rutas de archivos para el dataset (Persona No Objetivo)
        rutaCarpeta = self.rutaDatasetClasificado + "/persona_no_objetivo/"

        for nombreArchivo in os.listdir (rutaCarpeta):
            self.listaFotogramasNoObjetivo.append (rutaCarpeta + "/" + nombreArchivo)

        # Se baraja el contenido de las listas (Para que no salga en orden)
        random.shuffle (self.listaFotogramasObjetivo)
        random.shuffle (self.listaFotogramasNoObjetivo)

        # Generacion del modelo de red neuronal
        self.redNeuronalDeteccionSimilaridad = RedNeuronal(self.TAMANO_ENTRADA).to(self.DISPOSITIVO)

        # Carga una funcion predefinida de perdida (Se pueden crear por el usuario)
        self.funcionPerdida = nn.BCELoss ()

        # Se carga el optimizador que ayuda a optimizar el aprendizaje de la red neuronal
        self.optimizador = torch.optim.Adam (self.redNeuronalDeteccionSimilaridad.parameters (), lr = self.TASA_APRENDIZAJE)

        print ("INFORMACION (2) -> El segundo paso se ha completado y el entorno esta preparado")
        
        return

    # ---------------------------------------

    def seleccion_imagenes_aleatoriamente (self):
        # Lo primero es seleccionar dos imagenes aleatoriamente 
        # Tanto la persona 1 como la 2 pueden tener personas objetivo y no objetivo (Si no crea conflicto)


        # Valores aleatorios entre 0 y 1 (Para ver si se coje objetivo o no objetivo)
        eleccionAleatoriaPersona1 = random.randint (0, 1)
        eleccionAleatoriaPersona2 = random.randint (0, 1)

        # Si es igual a 1 se coje una persona objetivo
        if eleccionAleatoriaPersona1 == 1:
            persona1 = cv.imread (random.choice(self.listaFotogramasObjetivo))

        # Persona no objetivo si es igual a 0 (!= 1)
        else:
            persona1 = cv.imread (random.choice (self.listaFotogramasNoObjetivo))


        # Si es igual a 1 se coje una persona objetivo
        if eleccionAleatoriaPersona2 == 1:
            persona2 = cv.imread (random.choice(self.listaFotogramasObjetivo))

        # Persona no objetivo si es igual a 0 (!= 1)
        else:
            persona2 = cv.imread (random.choice (self.listaFotogramasNoObjetivo))

        persona1Redimensionada = cv.resize (persona1, (self.TAMANO_ENTRADA[1], self.TAMANO_ENTRADA[0]))
        persona2Redimensionada = cv.resize (persona2, (self.TAMANO_ENTRADA[1], self.TAMANO_ENTRADA[0]))


        resultadoSimilitud = float (eleccionAleatoriaPersona1 and eleccionAleatoriaPersona2)

        return persona1Redimensionada, persona2Redimensionada, resultadoSimilitud

    # ---------------------------------------------------------------------------------------

    def entrenamiento_red_neuronal_una_imagen (self, persona1, persona2, resultadoSimilitud):
        # Primero es necesario transformar las imagenes a tensores
        persona1Tensor = self.CONVERSOR (persona1)
        persona2Tensor = self.CONVERSOR (persona2)
        resultadoSimilitudTensor = torch.tensor([resultadoSimilitud], dtype=torch.float32)      # Dandole formato [1] para luego poder igualar a la salida de la red neuronal
        
        # Se pasan a la GPU o CPU
        persona1EnDispositivo = persona1Tensor.to (self.DISPOSITIVO).unsqueeze (0)      # Se añade una dimension extra (No sirve para nada)
        persona2EnDispositivo = persona2Tensor.to (self.DISPOSITIVO).unsqueeze (0)      # Se añade una dimension extra (No sirve para nada)
        resultadoSimilitudEnDispositivo = resultadoSimilitudTensor.unsqueeze (0).to (self.DISPOSITIVO)

        # Limpieza de gradientes acumulados
        self.optimizador.zero_grad ()

        # Prediccion de la similitud por la red neuronal
        prediccionSimilitud = self.redNeuronalDeteccionSimilaridad (persona1EnDispositivo, persona2EnDispositivo)
        #resultado = self.redNeuronalDeteccionSimilaridad.predecir_resultado_individual (persona1EnDispositivo)

        # Perform the matrix multiplication
        perdida = self.funcionPerdida(prediccionSimilitud, resultadoSimilitudEnDispositivo)

        # Calcula la funcion de perdida con respecto a todos los parametros de la red neuronal (Magnitud y direccion del cambio a aplicar)
        perdida.backward ()

        # Actualiza los parametros en funcion de la perdida (Correccion en base a la perdida, a mayor, mayor la correccion)
        self.optimizador.step ()

        # Incremento de variables
        self.contadorPerdidaEpoca += perdida.item ()
        self.contadorImagenes += 1
        
        return
    




    # -----------------------------------------------------------------------

    def interfaz_usuario (self, fotograma1, fotograma2, porcentajeSimilitud):
        print ("Porcentaje Similitud:", porcentajeSimilitud)

        cv.imshow ("Fotograma 1", fotograma1)
        cv.imshow ("Fotograma 2", fotograma2)

        # Se le asigna la espera minima de 1ms ya que interesa la fluidez
        self.controlador_teclas (letraPulsada=cv.waitKey (0))

        return

    # ------------------------------------------

    def controlador_teclas (self, letraPulsada):
        # Si el valor es -1 siginifca que no se pulso ninguna tecla (No merece la pena hacer ninguna comprobacion)
        if letraPulsada != -1:

            # Se ha pulsado la letra ESC
            if letraPulsada == 27:
                self.generacion_de_resultados ()
                sys.exit ("FIN EJECUCION (2): Asegurese de comprobar el modelo guardado en la ruta " + self.rutaDestinoRedEntrenada + "/model_trained")
            
            # Es escalable (Usando la siguiente estructura)
            #elif letraPulsada == <codigo_letra>:
                # Actuacion  
                  
        return

    # ----------------------------------

    def generacion_de_resultados (self):
        """
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
        """
        return











    
# -----------------------------------

class RedNeuronal(nn.Module):
     def __init__(self, input_shape):
        super(RedNeuronal, self).__init__()

        self.shared_layers = nn.Sequential(
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

        self.fc = nn.Linear(64 * 2, 1)
        self.sigmoid = nn.Sigmoid()

     def forward(self, x1, x2):
        x1 = self.shared_layers(x1)
        x2 = self.shared_layers(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)
          
        return x