from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


# Importacion de librerias necesarias
import cv2 as cv
import numpy as np
import random
import torchvision.transforms as transforms
import torch
import sys
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

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

class SpecificWorker(GenericWorker):
  
    # Referencias al timer
    periodo = 33
    tiempoInicio = None
    tiempoMedioEpocas = []

    directorioObjetivo = "/media/robocomp/data_tfg/oficialDatasetFiltered1/targetPerson"
    directorioNoObjetivo = "/media/robocomp/data_tfg/oficialDatasetFiltered1/noTargetPerson"
    directorioGuardarModelo = "/home/robocomp/funciona"

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
    datasetOriginal = None

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


    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = self.periodo

        self.comprobacion_condiciones_necesarias ()

        self.preparacion_entorno ()

        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

    def __del__(self):
        

        return

    def setParams(self, params):
        self.contadorEpocas = 0
        self.contadorFotogramasProcesados = 0
        self.contadorPerdida = 0.0

        return True


    @QtCore.Slot()
    def compute(self):
        """
        Realiza el entrenamiento del modelo.

        Returns:
            bool: Verdadero si se completó el cálculo.
        """

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
            self.save_model_and_figure ()
            sys.exit ("FIN: Se ha acabado el código debido a que se han alcanzado el numero de epocas especificado")

        return True



    # ------------------------------------------------
    # ------------------ INITIALIZE ------------------
    # ------------------------------------------------

    def comprobacion_condiciones_necesarias (self):

        # Genera seeds aleatoria
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Se comprueba si existe la ruta del dataset
        if not os.path.exists (self.directorioObjetivo) :            
            sys.exit (f"ERROR (1): No existe el directorio {self.directorioObjetivo}. Compruebelo bien.")
        
        if not os.path.exists (self.directorioNoObjetivo):
            sys.exit (f"ERROR (2): No existe el directorio {self.directorioNoObjetivo}. Compruebelo bien.")

        if not os.path.exists (self.directorioGuardarModelo):
            sys.exit (f"ERROR (3): No existe el directorio {self.directorioGuardarModelo}. Compruebelo bien.")


        # Si se llega a este punto existen las dos carpetas
        self.cargar_dataset_del_disco ()

        return
    
    # -----------------------------

    def preparacion_entorno (self):
        """
        Prepara el entorno para el entrenamiento.

        Returns:
            None
        """
        # Se crea el modelo de red neuronal
        self.redNeuronalDeteccionSimilitud = RedNeuronal(self.INPUT_SIZE).to(self.DISPOSITIVO)

        # Se define la funcion de perdida
        self.funcionPerdida = nn.BCELoss ()

        # Se declara el optimizador (Tipo Adam) y se le asocia un learning rate
        self.optimizador = torch.optim.Adam (self.redNeuronalDeteccionSimilitud.parameters (), lr = self.TASA_DE_APRENDIZAJE)
        
        return

    # ----------------------------------

    def cargar_dataset_del_disco (self):
        """
        Carga el conjunto de datos desde el disco.

        Returns:
            None
        """

        filesT = os.listdir(self.directorioObjetivo)
        filesnT = os.listdir(self.directorioNoObjetivo)

        listaRutasAbsolutasFramesT = [os.path.join(self.directorioObjetivo, file) for file in filesT]
        listaRutasAbsolutasFramesnT = [os.path.join(self.directorioNoObjetivo, file) for file in filesnT]

        X_train_frame1 = listaRutasAbsolutasFramesT
        X_train_frame2 = []
        y_train_similarity = []

        # Lo primero de todo se va a rellenar la lista X_train_frame1 con valores aleatorios del mismo
        if len (X_train_frame1) < self.NUMERO_IMAGENES_ENTRENAMIENTO:
            while len(X_train_frame1) < self.NUMERO_IMAGENES_ENTRENAMIENTO:
                X_train_frame1.append (random.choice (X_train_frame1))    
        
        else:
            #X_train_frame1 = X_train_frame1[:self.NUMERO_DATOS_DATASET]
            X_train_frame1 = [random.choice (listaRutasAbsolutasFramesT)for _ in range (self.NUMERO_IMAGENES_ENTRENAMIENTO)]

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
                image1 = X_train_frame1 [i]
                X_train_frame1 [i] = X_train_frame2 [i]
                X_train_frame2 [i] = image1

        # Creacion del dataset y su asociacion en la variable global
        self.datasetOriginal = SimilarityDataset(X_train_frame1, X_train_frame2, y_train_similarity, self.INPUT_SIZE, self.TRANSFORMADOR)
        
        self.datasetEntrenamiento = DataLoader(self.datasetOriginal, batch_size=self.TAMANO_LOTE, shuffle=True)
        
        return

    # -------------------------------------------------
    # -------------------- COMPUTE --------------------
    # -------------------------------------------------
    
    def procesamiento_imagenes (self):
        """
        Procesa las imágenes y calcula la pérdida durante el entrenamiento.

        Returns:
            None
        """
        imagen1, imagen2, resultado = next (iter (self.datasetEntrenamiento))

        #print ("imagen1 shape:", imagen1.shape)
        #print ("imagen2 shape:", imagen2.shape)
        #print ("resultado shape:", resultado.shape)


        # Se pasan al dispositivo correspondiente
        imagen1 = imagen1.to (self.DISPOSITIVO)
        imagen2 = imagen2.to (self.DISPOSITIVO)
        resultado = resultado.float ().to (self.DISPOSITIVO)
        
        #print ("resultado", resultado.shape)
        #print ("resultado", resultado)
        
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

    def save_model_and_figure (self):
        # Guardado del modelo de red neuronal
        torch.save(self.redNeuronalDeteccionSimilitud.state_dict(), self.directorioGuardarModelo + "/model.pth")


        return