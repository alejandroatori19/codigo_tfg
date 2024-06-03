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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')  # Use Agg backend, which does not require QApplication


# Librería pythorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class SpecificWorker(GenericWorker):

    periodo = 33

    redNeuronalDeteccionSimilaridad = None
    funcionPerdida = None
    optimizador = None

    # Las siguientes tres constantes se pueden modificar para intentar mejorar el rendimiento dela red (Los valores asignados son los recomendados)
    NUMERO_EPOCAS = 10
    NUMERO_IMAGENES_ENTRENAMIENTO = 1000
    NUMERO_IMAGENES_VALIDACION = 100
    TASA_APRENDIZAJE = 0.001
    BATCH_SIZE = 32                                              # Tiene que ser exponente de 2 y ser > 1
    VALIDANDO_RED_NEURONAL = False

    # Rutas del dataset y destino del modelo entrenado
    rutaDatasetClasificado = "/media/robocomp/data_tfg/dataset_clasificado"
    rutaDestinoResultados = "/home/robocomp"

    # Lista de rutas absolutas y dataset
    datasetEntrenamiento = None
    datasetValidacion = None

    # Constantes
    NUMERO_IMAGENES_DATASET = 5000
    NUMERO_DECIMALES = 8
    TAMANO_ENTRADA = [350, 150, 3]
    DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONVERSOR = transforms.Compose([
        transforms.ToTensor(),  # Convertir la imagen a un tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Assuming ImageNet normalization
    ])

    # Contadores
    contadorEpocas = None
    contadorImagenes = None
    contadorPerdida = None

    # Guarda las perdidas por cada epoca
    listaPerdidasEntrenamiento = []
    listaPerdidasValidacion = []

    tiempoEjecucion = None

    # -------------------------------------------------

    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = self.periodo

        # Comprobacion inicial de requirimientos minimos
        self.comprobacion_requisitos_minimos ()

        # Se prepara el entorno de la fase (Creacion de red neuronal y carga de rutas de archivos)
        self.preparacion_entorno ()

        self.tiempoEjecucion = time.time ()

        print ("----------- INICIO ENTRENAMIENTO/VALIDACION -----------")

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
        self.contadorPerdida = 0

        return True

    # ----------------

    @QtCore.Slot()
    def compute(self):
        # Obtencion de datos del DataLoader

        # Si la validacion no esta activa significa que se esta entrenando
        if not self.VALIDANDO_RED_NEURONAL:

            # En el entrenamiento se tienen que procesar n frames (Definido en variable global)
            if self.contadorImagenes < self.NUMERO_IMAGENES_ENTRENAMIENTO:

                # Se pone al modelo en el modo de entrenamiento
                self.redNeuronalDeteccionSimilaridad.train ()

                self.entrenamiento_red_neuronal ()
                #sys.exit ("pruebas")
                    
            # Significa que ya proceso entrenando los n frames predefinidos
            else:
                # Se inicia la validacion despues del entrenamiento
                self.VALIDANDO_RED_NEURONAL = True

                # Guardados de valores permanentes
                self.listaPerdidasEntrenamiento.append (round (self.contadorPerdida / self.contadorImagenes, self.NUMERO_DECIMALES))

                # Reset e incremento de variables
                self.contadorImagenes = 0
                self.contadorPerdida = 0                      

                

        # Si entra aqui es que esta validando la epoca
        else:
            if self.contadorImagenes < self.NUMERO_IMAGENES_VALIDACION:

                # Se pone al modelo en el modo de evaluacion
                self.redNeuronalDeteccionSimilaridad.eval ()

                # Eleccion de dos imagenes aleatoriamente (2 de objetivo o 1 del objetivo y 1 del no objetivo)
                self.validacion_red_neuronal ()

            else:
                # Se inicia el entremiento despues de la validacion
                self.VALIDANDO_RED_NEURONAL = False

                # Guardados de valores permanentes
                self.listaPerdidasValidacion.append (round (self.contadorPerdida / self.contadorImagenes, self.NUMERO_DECIMALES))

                # Reset e incremento de variables
                self.contadorImagenes = 0
                self.contadorPerdida = 0
                self.contadorEpocas += 1          

                # Se muestran los datos de la epoca (Perdidas en validacion y entrenamiento)
                self.mostrar_datos_epoca_individual ()

                # Se comprueba si se ha acabado 
                if self.contadorEpocas >= self.NUMERO_EPOCAS:
                    print ("----------- FIN ENTRENAMIENTO/VALIDACION -----------\n")

                    self.generacion_de_resultados ()
                    sys.exit ("Final numero maximo de epocas alcanzada")

        return True
    
    # METODOS SECUNDARIOS DE APOYO A LOS PRINCIPALES

    # -----------------------------------------

    def comprobacion_requisitos_minimos (self):
        # Comprueba que existe la carpeta con el contenido del dataset sin clasificar
        if not os.path.exists (self.rutaDatasetClasificado):
            sys.exit ("ERROR (1): El directorio que contiene el dataset clasificado no existe - Ruta: " + self.rutaDatasetClasificado + "\n\n")
        
        # Comprueba que existe la ruta con el destino del dataset clasificado (El directorio padre)
        if not os.path.exists (self.rutaDestinoResultados):
            sys.exit ("ERROR (2): Comprueba que el directorio de la ruta destino del modelo existe - Ruta: " + self.rutaDestinoResultados + "\n\n")

        print ("INFORMACION (1) -> El primer paso se ha completado y los requisitos minimos se cumplen")
    
        return
    
    # -----------------------------

    def preparacion_entorno (self):
        # Generacion del modelo de red neuronal
        self.redNeuronalDeteccionSimilaridad = RedNeuronal(self.TAMANO_ENTRADA).to(self.DISPOSITIVO)

        # Carga una funcion predefinida de perdida (Se pueden crear por el usuario)
        self.funcionPerdida = nn.BCELoss ()

        # Se carga el optimizador que ayuda a optimizar el aprendizaje de la red neuronal
        self.optimizador = torch.optim.Adam (self.redNeuronalDeteccionSimilaridad.parameters (), lr = self.TASA_APRENDIZAJE)

        self.generacion_datasets_entrenamiento_validacion ()

        print ("INFORMACION (2) -> El segundo paso se ha completado y el entorno esta preparado")
        
        return

    # ------------------------------------------

    def generacion_datasets_entrenamiento_validacion (self):
        # Carga de rutas de archivos para el dataset (Persona Objetivo)        
        listaFotogramasObjetivo = [self.rutaDatasetClasificado + "/persona_objetivo/" + nombreArchivo for nombreArchivo in os.listdir(self.rutaDatasetClasificado + "/persona_objetivo")]

        # Carga de rutas de archivos para el dataset (Persona No Objetivo)
        listaFotogramasNoObjetivo = [self.rutaDatasetClasificado + "/persona_no_objetivo/" + nombreArchivo for nombreArchivo in os.listdir(self.rutaDatasetClasificado + "/persona_no_objetivo/")]
        
        # Genera el dataset para entrenamiento (En batch sizes y mezclando los valores)
        fotogramas1, fotogramas2, resultados = self.generar_dataset_general (listaFotogramasObjetivo, listaFotogramasNoObjetivo, True)
    
        datasetF = DatasetPersonalizado(fotogramas1, fotogramas2, resultados, self.TAMANO_ENTRADA, self.CONVERSOR)
        self.datasetEntrenamiento = DataLoader (datasetF, batch_size=self.BATCH_SIZE, shuffle=True)

        # Genera el dataset para validacion
        fotogramas1, fotogramas2, resultados = self.generar_dataset_general (listaFotogramasObjetivo, listaFotogramasNoObjetivo, False)
        
        datasetF = DatasetPersonalizado(fotogramas1, fotogramas2, resultados, self.TAMANO_ENTRADA, self.CONVERSOR)
        self.datasetValidacion = DataLoader (datasetF, batch_size=self.BATCH_SIZE, shuffle=True)
        return
    
    # ------------------------------------------
    
    def generar_dataset_general (self, listaFotogramasObjetivo, listaFotogramasNoObjetivo, datasetEntrenamiento):
        listaFotogramas1 = listaFotogramasObjetivo
        listaFotogramas2 = []
        resultados = []

        numeroImagenesACargar = self.NUMERO_IMAGENES_ENTRENAMIENTO if datasetEntrenamiento else self.NUMERO_IMAGENES_VALIDACION

        # Lo primero de todo se va a rellenar la lista X_train_frame1 con valores aleatorios del mismo
        if len (listaFotogramas1) < numeroImagenesACargar:
            while len(listaFotogramas1) < numeroImagenesACargar:
                listaFotogramas1.append (random.choice (listaFotogramas1))    
        
        else:
            listaFotogramas1 = [random.choice (listaFotogramasNoObjetivo)for _ in range (numeroImagenesACargar)]

        # Se añade aleatoriamente imagenes del objetivo o no objetivos
        for i in range (numeroImagenesACargar):
            valorAleatorio = random.randint (0, 1)
            listaFotogramas2.append(random.choice(listaFotogramasNoObjetivo if valorAleatorio == 0 else listaFotogramasObjetivo))
            resultados.append(float (valorAleatorio))

        # Mezclar los datos (Basicamente que no siempre la primera imagen sea la objetivo)
        for i in range (len (resultados)):
            
            # Se cambia el orden
            if random.randint (0, 1) == 0 and resultados[i] == 0:
                image1 = listaFotogramas1 [i]
                listaFotogramas1 [i] = listaFotogramas2 [i]
                listaFotogramas2 [i] = image1
        
        return listaFotogramas1, listaFotogramas2, resultados

    # ------------------------------------

    def entrenamiento_red_neuronal (self):

        fotograma1, fotograma2, resultados = next (iter (self.datasetEntrenamiento))

        # Se pasa los tensores al dispositivo correspondiente
        fotograma1Dispositivo = fotograma1.to (self.DISPOSITIVO)
        fotograma2Dispositivo = fotograma2.to (self.DISPOSITIVO)
        resultadosDispositivo = resultados.float ().to (self.DISPOSITIVO)           # Conversion a float porque si no es double

        self.optimizador.zero_grad ()

        resultadoPredecido = self.redNeuronalDeteccionSimilaridad (fotograma1Dispositivo, fotograma2Dispositivo)

        perdida = self.funcionPerdida (resultadoPredecido, resultadosDispositivo.unsqueeze(1))

        perdida.backward ()

        self.optimizador.step ()

        # Incremento de variables
        self.contadorPerdida += perdida.item () * self.BATCH_SIZE
        self.contadorImagenes += self.BATCH_SIZE
        
        return
    
    # ---------------------------------

    def validacion_red_neuronal (self):

        fotograma1, fotograma2, resultados = next (iter (self.datasetValidacion))

        # Se pasa los tensores al dispositivo correspondiente
        fotograma1Dispositivo = fotograma1.to (self.DISPOSITIVO)
        fotograma2Dispositivo = fotograma2.to (self.DISPOSITIVO)
        resultadosDispositivo = resultados.float ().to (self.DISPOSITIVO)           # Conversion a float porque si no es double
        
        with torch.no_grad ():      # Deshabilita la computacion de gradiente (No aprende)
            
            # Prediccion del resultado
            resultadoPredecido = self.redNeuronalDeteccionSimilaridad (fotograma1Dispositivo, fotograma2Dispositivo)
            
            # Calculo de la perdida (Diferencia entre lo esperado y lo obtenido)
            perdida = self.funcionPerdida (resultadoPredecido, resultadosDispositivo.unsqueeze(1))
            
            # Aumenta el contador de perdida
            self.contadorPerdida += perdida.item () * self.BATCH_SIZE

            # Aumenta el contador
            self.contadorImagenes += self.BATCH_SIZE

        return

    # ----------------------------------------

    def mostrar_datos_epoca_individual (self):

        print ("Epoca[" + str(self.contadorEpocas) + "/" + str(self.NUMERO_EPOCAS) + "]", end="\t")        
        print ("Perdida entrenamiento (Medias): " + str(self.listaPerdidasEntrenamiento[-1]), end="\t")
        print ("Perdida validacion (Medias): " + str(self.listaPerdidasValidacion[-1]), end="\t")
        print ("")

        return

    # ----------------------------------

    def generacion_de_resultados (self):

        # Guardado del modelo de red neuronal (Junto con todo el aprendizaje). Arquitectura más parametros entrenados
        torch.save (self.redNeuronalDeteccionSimilaridad.state_dict (), self.rutaDestinoResultados + "/model.pth")

        # Calculo final
        self.tiempoEjecucion = time.time() - self.tiempoEjecucion

        resultadosPorEpoca = [f"EPOCA[{i+1}/{self.NUMERO_EPOCAS}]  -  Perdida Entrenamiento: {self.listaPerdidasEntrenamiento[i]}  -  Perdida validacion: {self.listaPerdidasValidacion[i]}" for i in range(len(self.listaPerdidasEntrenamiento))]
        
        # Creacion de diccionario para guardar la informacion en formato JSON
        resultadoFormatoJSON = {"Tiempo de conexion" : self.tiempoEjecucion,
                                "Ruta destino del modelo" : self.rutaDestinoResultados,
                                "Numero de epocas" : self.NUMERO_EPOCAS,
                                "Tasa de aprendizaje" : self.TASA_APRENDIZAJE,
                                "Tamano de batch" : self.BATCH_SIZE,
                                "Resultados entrenamiento y validacion" : resultadosPorEpoca
                                }
        
        # Resultados por consola (Temporales)
        print ("\n\n----------- INICIO RESULTADOS -----------")
        print ("Tiempo de conexion:", self.tiempoEjecucion, " segundos")
        print ("Numero de epocas:", str(self.NUMERO_EPOCAS), "epocas")
        print ("Tasa de aprendizaje", self.TASA_APRENDIZAJE)
        print ("Tamano de batch", str (self.BATCH_SIZE))
        print ("------------ FIN RESULTADOS ------------\n\n")

        # Resultados por archivo JSON (Permanentes)
        if os.path.exists (os.path.dirname (self.rutaDestinoResultados)):
            with open (self.rutaDestinoResultados + "/data.json", 'w') as flujoSalida:
                json.dump (resultadoFormatoJSON, flujoSalida)

        # Configura el grafico
        plt.figure(figsize=(10, 6))
        plt.plot(list(range(1, self.NUMERO_EPOCAS + 1)), self.listaPerdidasEntrenamiento, 'b', label='Training Loss')
        plt.plot(list(range(1, self.NUMERO_EPOCAS + 1)), self.listaPerdidasValidacion, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Guarda la gráfica en disco
        plt.savefig (self.rutaDestinoResultados + "/training_validation_loss_plot.png")

        return
    
# -----------------------------------

# CLASE PARA GESTION DE DATASET. Ayuda con la abstraccion
class DatasetPersonalizado (Dataset):

    # Constructor
    def __init__(self, fotogramasObjetivo, fotogramasNoObjetivo, resultadosSimilitud, tamano_entrada, transformador):
        self.listaRutasObjetivo = fotogramasObjetivo
        self.listaRutasNoObjetivo = fotogramasNoObjetivo
        self.listaResultadosSimilitud = resultadosSimilitud
        self.tamanoEntrada = tamano_entrada
        self.CONVERSOR = transformador

    # Devuelve la longitud de la lista resultados
    def __len__(self):
        return len(self.listaResultadosSimilitud)

    # Devuelve items cuando se le solicita
    def __getitem__(self, idx):
        fotograma1 = cv.imread(self.listaRutasObjetivo[idx])
        fotograma2 = cv.imread(self.listaRutasNoObjetivo[idx])
        resultado = self.listaResultadosSimilitud[idx]

        fotograma1Redimensionado = cv.resize(fotograma1, (self.tamanoEntrada[1], self.tamanoEntrada[0]))
        fotograma2Redimensionado = cv.resize(fotograma2, (self.tamanoEntrada[1], self.tamanoEntrada[0]))

        fotograma1Arreglado = self.CONVERSOR(fotograma1Redimensionado)
        fotograma2Arreglado = self.CONVERSOR(fotograma2Redimensionado)

        return fotograma1Arreglado, fotograma2Arreglado, resultado
    
# ---------------------------------------------------------
    
# CLASE PARA GESTION RED NEURONAL. Ayuda con la abstraccion
class RedNeuronal(nn.Module):
     
    # Constructor
    def __init__(self, tamano_entrada):
        super(RedNeuronal, self).__init__()

        self.capas_compartidas = nn.Sequential(
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

        self.capa_full_conectada = nn.Linear(64 * 2, 1)
        self.salida = nn.Sigmoid()

        return

    # Pasa 2 inputs a traves de la red neuronal y obtiene un resultado unificado
    def forward(self, fotograma1, fotograma2):
        resultadoFotograma1 = self.capas_compartidas(fotograma1)
        resultadoFotograma2 = self.capas_compartidas(fotograma2)
        resultadoUnido = torch.cat((resultadoFotograma1, resultadoFotograma2), dim=1)
        resultadoValorUnico = self.capa_full_conectada(resultadoUnido)
        resultadoPorcentaje = self.salida(resultadoValorUnico)
          
        return resultadoPorcentaje