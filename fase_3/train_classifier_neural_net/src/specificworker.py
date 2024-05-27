from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


# Importacion de librerias necesarias
from neuralNet_Dataset import SimilarityModel, SimilarityDataset

import cv2 as cv
import numpy as np
import random
import torchvision.transforms as transforms
import torch
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time



class SpecificWorker(GenericWorker):
    """
    Clase SpecificWorker para entrenar un modelo de red neuronal para detección de similitud.

    Atributos:
        periodo (int): Período del temporizador.
        tiempoInicio (float): Tiempo de inicio del entrenamiento.
        tiempoMedioEpocas (list): Lista para almacenar los tiempos de las épocas.
        directorioObjetivo (str): Directorio que contiene imágenes con personas objetivo.
        directorioNoObjetivo (str): Directorio que contiene imágenes sin personas objetivo.
        directorioGuardarModelo (str): Directorio para guardar modelos entrenados.
        modeloRedNeuronal (SimilarityModel): Modelo de red neuronal.
        funcionPerdida (torch.nn.BCELoss): Función de pérdida.
        optimizador (torch.optim.Adam): Optimizador para el entrenamiento del modelo.
        contadorEpocas (int): Contador para las épocas.
        contadorImagenes (int): Contador para las imágenes procesadas.
        contadorPerdida (float): Pérdida acumulada durante el entrenamiento.
        listaContadorPerdida (list): Lista para almacenar los valores de pérdida.
        datasetEntrenamiento (DataLoader): Cargador del conjunto de datos de entrenamiento.
        datasetOriginal (SimilarityDataset): Conjunto de datos de entrenamiento original.
        NUMERO_DATOS_DATASET (int): Número de puntos de datos en el conjunto de datos.
        INPUT_SIZE (tuple): Tamaño de entrada de las imágenes (alto, ancho, canales).
        MEZCLAR_DATASET (bool): Si se debe mezclar el conjunto de datos.
        NUMERO_DECIMALES (int): Número de decimales para redondear.
        NUMERO_EPOCAS (int): Número de épocas para el entrenamiento.
        BATCH_SIZE (int): Tamaño del lote para el entrenamiento.
        LEARNING_RATE (float): Tasa de aprendizaje para el optimizador.
        device (torch.device): Dispositivo para el entrenamiento (CPU o GPU).
        transform (torchvision.transforms.Compose): Transformaciones de imagen.
    """

    # Referencias al timer
    periodo = 33
    tiempoInicio = None
    tiempoMedioEpocas = []

    directorioObjetivo = "/media/robocomp/data_tfg/oficialDatasetFiltered1/targetPerson"
    directorioNoObjetivo = "/media/robocomp/data_tfg/oficialDatasetFiltered1/noTargetPerson"
    directorioGuardarModelo = "/home/robocomp"

    # Red neuronal
    modeloRedNeuronal = None
    funcionPerdida = None
    optimizador = None
    
    # Contadores
    contadorEpocas = None
    contadorImagenes = None
    contadorPerdida = None
    listaContadorPerdida = []
    
    # Dataset
    datasetEntrenamiento = None
    datasetOriginal = None

    # Constantes
    NUMERO_DATOS_DATASET = 5000
    INPUT_SIZE = (350, 150, 3)
    MEZCLAR_DATASET = True
    NUMERO_DECIMALES = 7

    NUMERO_EPOCAS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Extra
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
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
        self.contadorImagenes = 0
        self.contadorPerdida = 0.0
        self.tiempoMedioEpocas.append (time.time ())
        return True


    @QtCore.Slot()
    def compute(self):
        """
        Realiza el entrenamiento del modelo.

        Returns:
            bool: Verdadero si se completó el cálculo.
        """

        if self.contadorEpocas < self.NUMERO_EPOCAS:
            self.modeloRedNeuronal.train ()

            if self.contadorImagenes < self.NUMERO_DATOS_DATASET:
                # Se procesan las imagenes, se obtiene resultado y se calcula una perdida en funcion al valor.
                self.procesamiento_imagenes ()

                # Incrementamos el numero de imagenes procesadas
                self.contadorImagenes += self.BATCH_SIZE
                
            else:
                # Muestra resultado epocas
                self.mostrar_resultados_epoca ()
                self.tiempoMedioEpocas[self.contadorEpocas] = time.time () - self.tiempoMedioEpocas[self.contadorEpocas]

                # Reset e incrementos de variables
                self.contadorEpocas += 1
                self.contadorImagenes = 0
                self.listaContadorPerdida.append (round (self.contadorPerdida / self.NUMERO_DATOS_DATASET, self.NUMERO_DECIMALES))
                self.contadorPerdida = 0.0

                if self.NUMERO_EPOCAS > self.contadorEpocas:
                    self.tiempoMedioEpocas.append (time.time ())

        else:
            self.save_model_and_figure ()
            sys.exit ("FIN: Se ha acabado el código debido a que se han alcanzado el numero de epocas especificado")

        return True



    # ------------------------------------------------
    # ------------------ INITIALIZE ------------------
    # ------------------------------------------------

    def comprobacion_condiciones_necesarias (self):
        """
        Realiza la comprobación de condiciones necesarias.

        Returns:
            None
        """

        # Genera seeds aleatoria
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        self.tiempoInicio = time.time ()

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
        self.modeloRedNeuronal = SimilarityModel(self.INPUT_SIZE).to(self.device)

        # Se define la funcion de perdida
        self.funcionPerdida = nn.BCELoss ()

        # Se declara el optimizador (Tipo Adam) y se le asocia un learning rate
        self.optimizador = torch.optim.Adam (self.modeloRedNeuronal.parameters (), lr = self.LEARNING_RATE)
        
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
        if len (X_train_frame1) < self.NUMERO_DATOS_DATASET:
            while len(X_train_frame1) < self.NUMERO_DATOS_DATASET:
                X_train_frame1.append (random.choice (X_train_frame1))    
        
        else:
            #X_train_frame1 = X_train_frame1[:self.NUMERO_DATOS_DATASET]
            X_train_frame1 = [random.choice (listaRutasAbsolutasFramesT)for _ in range (self.NUMERO_DATOS_DATASET)]

        # Se añade aleatoriamente imagenes del objetivo o no objetivos
        for i in range (self.NUMERO_DATOS_DATASET):
            valorAleatorio = random.randint (0, 1)
            if valorAleatorio == 0:
                X_train_frame2.append (random.choice(listaRutasAbsolutasFramesnT))
                y_train_similarity.append (0)

            if valorAleatorio == 1:
                X_train_frame2.append (random.choice(listaRutasAbsolutasFramesT))
                y_train_similarity.append (1)
     
        # Mezclar los datos (Basicamente que no siempre la primera imagen sea la objetivo)
        for i in range (self.NUMERO_DATOS_DATASET):
            # Se cambia el orden
            if random.randint (0, 1) == 0 and y_train_similarity[i] == 0:
                image1 = X_train_frame1 [i]
                X_train_frame1 [i] = X_train_frame2 [i]
                X_train_frame2 [i] = image1

        # Creacion del dataset y su asociacion en la variable global
        self.datasetOriginal = SimilarityDataset(X_train_frame1, X_train_frame2, y_train_similarity, self.INPUT_SIZE, transform=self.transform)
        
        self.datasetEntrenamiento = DataLoader(self.datasetOriginal, batch_size=self.BATCH_SIZE, shuffle=self.MEZCLAR_DATASET)
        
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

        # Se pasan al dispositivo correspondiente
        imagen1 = imagen1.to (self.device)
        imagen2 = imagen2.to (self.device)
        resultado = resultado.float ().to (self.device)
        
        self.optimizador.zero_grad ()

        resultadoRedNeuronal = self.modeloRedNeuronal (imagen1, imagen2)

        valorPerdida = self.funcionPerdida (resultadoRedNeuronal, resultado.unsqueeze(1))

        valorPerdida.backward ()

        self.optimizador.step ()

        self.contadorPerdida += valorPerdida.item () * imagen1.size (0)

        return
    
    # ----------------------------------

    def mostrar_resultados_epoca (self):
        """
        Muestra los resultados del entrenamiento para cada época.

        Returns:
            None
        """

        # Si es el inicio
        if self.contadorEpocas == 0:
            print ("-------------------- INICIO INFORMACION ENTRENAMIENTO --------------------\n")
        
        print (f"EPOCA [{self.contadorEpocas + 1}/{self.NUMERO_EPOCAS}]", end="")
        #print (f"\t Precision de la fase de validacion -> {1}")
        print (f"\t Perdida entrenamiento: {round (self.contadorPerdida / self.NUMERO_DATOS_DATASET, self.NUMERO_DECIMALES)}")

        if self.contadorEpocas == self.NUMERO_EPOCAS - 1:
            print ("\n-------------------- FIN INFORMACION ENTRENAMIENTO --------------------")
    
        return
    
    # ------------------------------------------------
    # -------------------- FINISH --------------------
    # ------------------------------------------------

    def save_model_and_figure (self):
        """
        Guarda el modelo entrenado y un gráfico de pérdida.

        Returns:
            None
        """

        # Creacion de la grafica de perdida por epochs
        plt.plot(self.listaContadorPerdida)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)

        # Guardado de la gráfica
        plt.savefig(self.directorioGuardarModelo + "/grafica.png")

        # Guardado del modelo de red neuronal
        torch.save(self.modeloRedNeuronal.state_dict(), self.directorioGuardarModelo + "/model.pth")

        # Calculando valores de tiempo finales
        tiempoTranscurrido = time.time() - self.tiempoInicio
        tiempoMedio = (sum (self.tiempoMedioEpocas)) / self.NUMERO_EPOCAS

        # Imprimiendo datos por consola
        print ("\n\n--------------- INFORMACION ---------------")
        print (f"Tiempo transcurrido total: {round (tiempoTranscurrido, self.NUMERO_DECIMALES)} segundos")        
        print (f"Tiempo medio por epoca: {round (tiempoMedio, self.NUMERO_DECIMALES)} segundos")
        print (f"Numero de epocas: {self.NUMERO_EPOCAS}")
        print (f"Imagenes procesadas por epoca: {self.NUMERO_DATOS_DATASET}")
        print ("------------- FIN INFORMACION -------------\n\n")

        return

    # ------------------------------------------------
    # ------------- INNECESARY / USEFULL -------------
    # ------------------------------------------------

    def print_dataset (self):
        """
        Muestra información sobre el conjunto de datos.

        Returns:
            None
        """

        print (f"longitud datset: {len (self.datasetOriginal)}")

        for i in range (self.datasetOriginal.__len__ ()):
            imagen1, imagen2, resultado = self.datasetOriginal.get_data_without_transform (i)
            
            cv.imshow ("imagen1", imagen1)
            cv.imshow ("imagen2", imagen2)
            print (f"Resultado -> {resultado}")

            if cv.waitKey (0) == 27:
                sys.exit ("Fin")

        return