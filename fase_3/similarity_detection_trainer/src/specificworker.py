from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from genericworker import *

sys.path.append('/opt/robocomp/lib')

# Importacion de librerias necesarias
import cv2 as cv
import numpy as np
import random
import sys
import time

# Librería pytorch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch

# Matplotlib para hacer graficos
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# -------------------------------

class SimilarityDataset(Dataset):
    def __init__(self, fotogramas1, fotogramas2, resultados, tamano_entrada, transform):
        self.listaFotogramas1 = fotogramas1
        self.listaFotogramas2 = fotogramas2
        self.resultadoSimilitud = resultados
        self.tamanoEntrada = tamano_entrada
        self.TRANSFORMADOR = transform

    def __len__ (self):
        return len (self.listaFotogramas1)

    def __getitem__(self, idx):
        fotograma1 = cv.imread(self.listaFotogramas1[idx])
        fotograma2 = cv.imread(self.listaFotogramas2[idx])
        resultado = self.resultadoSimilitud[idx]

        fotograma1Redimensionado = cv.resize(fotograma1, (self.tamanoEntrada[1], self.tamanoEntrada[0]))
        fotograma2Redimensionado = cv.resize(fotograma2, (self.tamanoEntrada[1], self.tamanoEntrada[0]))

        fotograma1Arreglado = self.TRANSFORMADOR(fotograma1Redimensionado)
        fotograma2Arreglado = self.TRANSFORMADOR(fotograma2Redimensionado)

        return fotograma1Arreglado, fotograma2Arreglado, resultado

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

        self.resultado = nn.Linear(64 * 2, 1)
        self.resultadoPorcentaje = nn.Sigmoid()

    def forward(self, fotograma1, fotograma2):
        resultadoFotograma1 = self.capasCompartidas(fotograma1)
        resultadoFotograma2 = self.capasCompartidas(fotograma2)
        resultadosConcatenados = torch.cat((resultadoFotograma1, resultadoFotograma2), dim=1)
        resultadoUnico = self.resultado(resultadosConcatenados)
        porcentajeSimilitud = self.resultadoPorcentaje(resultadoUnico)
          
        return porcentajeSimilitud

# ----------------------------------

class SpecificWorker(GenericWorker):
  
    # Referencias al timer
    periodo = 33

    rutaDataset = "/media/robocomp/data_tfg/oficialDatasetFiltered1"
    rutaDestinoModelo = "/home/robocomp/funciona"

    # Red neuronal
    redNeuronalDeteccionSimilitud = None
    funcionPerdida = None
    optimizador = None
    
    # Variables que afectan al proceso de entrenamiento/validacion
    NUMERO_IMAGENES_ENTRENAMIENTO = 1000
    NUMERO_EPOCAS = 10
    TAMANO_LOTE = 32
    TASA_DE_APRENDIZAJE = 0.001    
    
    # Dataset
    datasetEntrenamiento = None
    datasetValidacion = None

    # Constantes
    INPUT_SIZE = (350, 150, 3)
    MEZCLAR_DATASET = True
    NUMERO_DECIMALES = 7
    
    # Contadores
    contadorEpocas = None
    contadorFotogramasProcesados = None
    contadorPerdida = None
    listaPerdidasEntrenamiento = []
    
    # Extra
    DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRANSFORMADOR = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Assuming ImageNet normalization
        ])

    # -------------------------------------------------

    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = self.periodo

        self.comprobacion_condiciones_necesarias ()

        self.preparacion_entorno ()

        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

    # ----------------

    def __del__(self):
        return

    # --------------------------
    
    def setParams(self, params):
        self.contadorEpocas = 0
        self.contadorFotogramasProcesados = 0
        self.contadorPerdida = 0.0

        return True


    # ----------------

    @QtCore.Slot()
    def compute(self):
        finEpoca = False
        
        if self.contadorEpocas < self.NUMERO_EPOCAS:
            self.redNeuronalDeteccionSimilitud.train ()

            if self.contadorFotogramasProcesados < self.NUMERO_IMAGENES_ENTRENAMIENTO:
                # Se procesan las imagenes, se obtiene resultado y se calcula una perdida en funcion al valor.
                self.procesamiento_imagenes ()

                # Incrementamos el numero de imagenes procesadas
                self.contadorFotogramasProcesados += self.TAMANO_LOTE
                
            else:
                
                # Reset e incrementos de variables
                self.contadorEpocas += 1
                self.contadorFotogramasProcesados = 0
                self.listaPerdidasEntrenamiento.append (round (self.contadorPerdida / self.NUMERO_IMAGENES_ENTRENAMIENTO, self.NUMERO_DECIMALES))
                self.contadorPerdida = 0.0
                
                # Muestra resultado epocas
                self.mostrar_resultados_epoca ()
                


        else:
            self.guardar_modelo_mostrar_resultados ()
            sys.exit ("FIN: Se ha acabado el código debido a que se han alcanzado el numero de epocas especificado")

        return True

    # ---------------------------------------------

    def comprobacion_condiciones_necesarias (self):

        # Se comprueba si existe la ruta del dataset
        if not os.path.exists (self.rutaDataset) :            
            sys.exit (f"ERROR (1): No existe el directorio {self.rutaDataset}. Compruebelo bien.")

        if not os.path.exists (self.rutaDestinoModelo):
            sys.exit (f"ERROR (3): No existe el directorio {self.rutaDestinoModelo}. Compruebelo bien.")

        return
    
    # -----------------------------

    def preparacion_entorno (self):
        # Se crea la red neuronal basada en la arquitectura de la clase llamada
        self.redNeuronalDeteccionSimilitud = RedNeuronal(self.INPUT_SIZE).to(self.DISPOSITIVO)

        # Se define la funcion de perdida
        self.funcionPerdida = nn.BCELoss ()

        # Se declara el optimizador (Tipo Adam) y se le asocia una tasa de aprendizaje
        self.optimizador = torch.optim.Adam (self.redNeuronalDeteccionSimilitud.parameters (), lr = self.TASA_DE_APRENDIZAJE)
        
        # Se carga el dataset que se utiliza para entrenar/validar la red
        self.carga_dataset_desde_disco ()
        
        return

    # ----------------------------------

    def carga_dataset_desde_disco (self):
        # Se cargan las rutas de los archivos que se van a utilizar (Archivos del dataset. Mayor eficiencia)
        fotogramasObjetivo = [self.rutaDataset + "/targetPerson/" + nombreArchivo for nombreArchivo in os.listdir(self.rutaDataset + "/targetPerson")]
        fotogramasNoObjetivo = [self.rutaDataset + "/noTargetPerson/" + nombreArchivo for nombreArchivo in os.listdir(self.rutaDataset + "/noTargetPerson")]

        # Se crean las 3 listas
        fotogramas1, fotogramas2, resultados = [], [], []

        # Lo primero de todo se va a rellenar la lista fotogramas1 con valores aleatorios de la persona objetivo
        while len(fotogramas1) < self.NUMERO_IMAGENES_ENTRENAMIENTO:
            # Las primeras n interacciones guardarán a personas objetivo (Se quieren guardar todas las personas objetivo al menos 1 vez)
            if len (fotogramas1) < len (fotogramasObjetivo):
                fotogramas1.append (fotogramasObjetivo[len(fotogramas1)])    
        
            # Cuando se ha llenado se cogen aleatoriamente para completar
            else:
                fotogramas1.append (random.choice (fotogramasObjetivo))
                
        # Después hace falta poner la otra persona con la que compararla y puede ser de la persona objetivo o de las no objetivo
        for _ in range (self.NUMERO_IMAGENES_ENTRENAMIENTO):
            valorAleatorio = random.randint (0, 1)
            if valorAleatorio == 1:
                fotogramas2.append (random.choice (fotogramasObjetivo))
            
            else:
                fotogramas2.append (random.choice (fotogramasNoObjetivo))
                
            resultados.append (valorAleatorio)
            
        # Mezclar los datos (Basicamente que no siempre la primera imagen sea la objetivo)
        for i in range (self.NUMERO_IMAGENES_ENTRENAMIENTO):
            # Se cambia el orden
            if random.randint (0, 1) == 0 and resultados[i] == 0:
                fotograma1 = fotogramas1 [i]
                fotogramas1 [i] = fotogramas2 [i]
                fotogramas2 [i] = fotograma1

                                    

        """
        # Se añade aleatoriamente imagenes del objetivo o no objetivos
        for i in range (self.NUMERO_IMAGENES_ENTRENAMIENTO):
            valorAleatorio = random.randint (0, 1)
            if valorAleatorio == 0:
                X_train_frame2.append (random.choice(listaRutasAbsolutasFramesnT))
                y_train_similarity.append (0)

            if valorAleatorio == 1:
                X_train_frame2.append (random.choice(listaRutasAbsolutasFramesT))
                y_train_similarity.append (1)
     
        # Mezclar los datos (Basicamente que no siempre la primera imagen sea la objetivo)
        for i in range (self.NUMERO_IMAGENES_ENTRENAMIENTO):
            # Se cambia el orden
            if random.randint (0, 1) == 0 and y_train_similarity[i] == 0:
                image1 = fotogramas1 [i]
                fotogramas1 [i] = X_train_frame2 [i]
                X_train_frame2 [i] = image1
        """
        # Creacion del dataset y su asociacion en la variable global
        datasetOriginal = SimilarityDataset(fotogramas1, fotogramas2, resultados, self.INPUT_SIZE, self.TRANSFORMADOR)
        
        self.datasetEntrenamiento = DataLoader(datasetOriginal, batch_size=self.TAMANO_LOTE, shuffle=True)
        
        return

    # --------------------------------
        
    def procesamiento_imagenes (self):

        imagen1, imagen2, resultado = next (iter (self.datasetEntrenamiento))

        # Se pasan al dispositivo correspondiente
        imagen1 = imagen1.to (self.DISPOSITIVO)
        imagen2 = imagen2.to (self.DISPOSITIVO)
        resultado = resultado.float ().to (self.DISPOSITIVO)
        
        self.optimizador.zero_grad ()

        resultadoRedNeuronal = self.redNeuronalDeteccionSimilitud (imagen1, imagen2)

        valorPerdida = self.funcionPerdida (resultadoRedNeuronal, resultado.unsqueeze(1))

        valorPerdida.backward ()

        self.optimizador.step ()

        self.contadorPerdida += valorPerdida.item () * self.TAMANO_LOTE

        return
    
    # ----------------------------------

    def mostrar_resultados_epoca (self):
        
        print ("Epoca[" + str(self.contadorEpocas) + "/" + str(self.NUMERO_EPOCAS) + "]", end="\t")        
        print ("Perdida entrenamiento (Medias): " + str(self.listaPerdidasEntrenamiento[-1]), end="\t")
        #print ("Perdida validacion (Medias): " + str(self.listaPerdidasValidacion[-1]), end="\t")
        print ("")

        return
    
    # -------------------------------

    def guardar_modelo_mostrar_resultados (self):
        # Guardado del modelo de red neuronal
        torch.save(self.redNeuronalDeteccionSimilitud.state_dict(), self.rutaDestinoModelo + "/model.pth")
        """
        # Calculo final
        self.tiempoEjecucion = time.time() - self.tiempoEjecucion

        resultadosPorEpoca = [f"EPOCA[{i+1}/{self.NUMERO_EPOCAS}]  -  Perdida Entrenamiento: {self.listaPerdidasEntrenamiento[i]}  -  Perdida validacion: {self.listaPerdidasValidacion[i]}" for i in range(len(self.listaPerdidasEntrenamiento))]
        
        # Creacion de diccionario para guardar la informacion en formato JSON
        resultadoFormatoJSON = {"Tiempo de conexion" : self.tiempoEjecucion,
                                "Ruta destino del modelo" : self.rutaDestinoModelo,
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

        """
        return