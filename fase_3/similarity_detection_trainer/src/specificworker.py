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
import json

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
            #nn.Dropout(0.5)         # Ayuda para reducir el overfitting
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
    tiempoEjecucion = None

    rutaDataset = "/media/robocomp/data_tfg/oficialDatasetFiltered1"
    rutaDestinoResultados = "/home/robocomp/funciona"

    # Red neuronal
    redNeuronalDeteccionSimilitud = None
    funcionPerdida = None
    optimizador = None
    
    # Variables que afectan al proceso de entrenamiento/validacion
    NUMERO_IMAGENES_ENTRENAMIENTO = 5000
    NUMERO_IMAGENES_VALIDACION = 1000
    NUMERO_EPOCAS = 10
    TAMANO_LOTE = 32
    TASA_DE_APRENDIZAJE = 0.001    
    MODO_FUNCIONAMIENTO = None                  # 0 Entrenamiento, 1 Validacion
    
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
    listaPerdidasValidacion = []
    
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

        # Comprueba que las condiciones minimas necesarias se cumplen
        self.comprobacion_condiciones_necesarias ()

        # Prepara el entorno necesario para la fase (Inicio de la red neuronal, optimizador, funcion de perdida y dataset)
        self.preparacion_entorno ()
        
        self.tiempoEjecucion = time.time ()

        print ("----------------- INICIO ENTRENAMIENTO -----------------")
        
        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

    # ----------------

    def __del__(self):
        return

    # --------------------------
    
    def setParams(self, params):
        # Inicio de variables
        self.contadorEpocas = 0
        self.contadorFotogramasProcesados = 0
        self.contadorPerdida = 0.0
        self.MODO_FUNCIONAMIENTO = 0

        return True


    # ----------------

    @QtCore.Slot()
    def compute(self):
        # Proceso del entrenamiento
        if self.MODO_FUNCIONAMIENTO == 0:        
            self.redNeuronalDeteccionSimilitud.train ()
            
            # Metodo que se encarga del entrenamiento de la red neuronal
            self.entrenamiento_red_neuronal ()
            
            # Se comprueba si se han procesado todos los frames indicados (el valor indicado por NUMERO_IMAGENES_ENTRENAMIENTO)
            if self.contadorFotogramasProcesados > self.NUMERO_IMAGENES_ENTRENAMIENTO:                
                # Incrementos de variables
                self.listaPerdidasEntrenamiento.append (round (self.contadorPerdida / self.contadorFotogramasProcesados, self.NUMERO_DECIMALES))
                
                # Reset de variables
                self.contadorFotogramasProcesados = 0
                self.contadorPerdida = 0.0
              
                # Cambia el modo de funcionamiento
                self.MODO_FUNCIONAMIENTO = 1
            
        # Proceso de la validacion
        elif self.MODO_FUNCIONAMIENTO == 1:
            self.redNeuronalDeteccionSimilitud.eval ()

            # Metodo que se encarga de la validacion de la red neuronal
            self.validacion_red_neuronal ()
                        
            if self.contadorFotogramasProcesados > self.NUMERO_IMAGENES_VALIDACION:
               # Incrementos y modificaciones
                self.listaPerdidasValidacion.append (round (self.contadorPerdida / self.contadorFotogramasProcesados, self.NUMERO_DECIMALES))
                self.contadorEpocas += 1 
                
                # Reset de variables
                self.contadorFotogramasProcesados = 0
                self.contadorPerdida = 0.0                
                          
                # Muestra resultado epocas
                self.mostrar_resultados_epoca ()
                
                # Cambia el modo de funcionamiento
                self.MODO_FUNCIONAMIENTO = 0
                
                if self.contadorEpocas == self.NUMERO_EPOCAS:
                    self.guardar_modelo_mostrar_resultados ()
                    sys.exit ("FIN: Se ha acabado el código debido a que se han alcanzado el numero de epocas especificado")
                
        # Si el valor es erroneo se acaba el programa
        else:
            sys.exit ("ERROR (3): El modo de funcionamiento no tiene un valor correcto, su valor es ->", self.MODO_FUNCIONAMIENTO)


        return True

    # ---------------------------------------------

    def comprobacion_condiciones_necesarias (self):

        # Se comprueba si existe la ruta del dataset
        if not os.path.exists (self.rutaDataset) :            
            sys.exit (f"ERROR (1): No existe el directorio {self.rutaDataset}. Compruebelo bien.")

        if not os.path.exists (self.rutaDestinoResultados):
            sys.exit (f"ERROR (2): No existe el directorio {self.rutaDestinoResultados}. Compruebelo bien.")

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

    # -----------------------------------

    def carga_dataset_desde_disco (self):
        # Se cargan las rutas de los archivos que se van a utilizar (Archivos del dataset. Mayor eficiencia)
        fotogramasObjetivo = [self.rutaDataset + "/targetPerson/" + nombreArchivo for nombreArchivo in os.listdir(self.rutaDataset + "/targetPerson")]
        fotogramasNoObjetivo = [self.rutaDataset + "/noTargetPerson/" + nombreArchivo for nombreArchivo in os.listdir(self.rutaDataset + "/noTargetPerson")]

        # Se crean las 3 listas
        fotogramas1E, fotogramas2E, resultadosE = [], [], []
        fotogramas1V, fotogramas2V, resultadosV = [], [], []

        # Lo primero de todo se va a rellenar la lista fotogramas1 (Los primeros se añaden todos y después aleatoriamente)
        while len(fotogramas1E) < self.NUMERO_IMAGENES_ENTRENAMIENTO:
            if len (fotogramas1E) < len (fotogramasObjetivo):
                fotogramas1E.append (fotogramasObjetivo[len(fotogramas1E)])    
        
            else:
                fotogramas1E.append (random.choice (fotogramasObjetivo))
                
        # Después se rellena la segunda lista con frames aleatorios del objetivo o de los no objetivos y se asigna el resultado (1 si son la misma persona y encima objetivo)
        for _ in range (self.NUMERO_IMAGENES_ENTRENAMIENTO):
            valorAleatorio = random.randint (0, 1)
            if valorAleatorio == 1:
                fotogramas2E.append (random.choice (fotogramasObjetivo))
            
            else:
                fotogramas2E.append (random.choice (fotogramasNoObjetivo))
                
            resultadosE.append (float (valorAleatorio))
            
        # Mezclar los datos (Basicamente que no siempre la primera imagen sea la objetivo)
        for i in range (self.NUMERO_IMAGENES_ENTRENAMIENTO):
            # Se cambia el orden
            if random.randint (0, 1) == 0 and resultadosE[i] == 0:
                fotograma1 = fotogramas1E [i]
                fotogramas1E [i] = fotogramas2E [i]
                fotogramas2E [i] = fotograma1

        # Creacion del dataset y su asociacion en la variable global
        datasetOriginal = SimilarityDataset(fotogramas1E, fotogramas2E, resultadosE, self.INPUT_SIZE, self.TRANSFORMADOR)
        self.datasetEntrenamiento = DataLoader(datasetOriginal, batch_size=self.TAMANO_LOTE, shuffle=True)
        
        # Creacion del dataset de validación (A partir del de entrenamiento)
        for _ in range (self.NUMERO_IMAGENES_VALIDACION):
            valorAleatorio = random.randint (0, 1)
            
            fotogramas1V.append (random.choice (fotogramas1E))
            
            if valorAleatorio == 0:
                fotogramas2V.append (random.choice (fotogramasNoObjetivo))
                
            else:
                fotogramas2V.append (random.choice (fotogramasObjetivo))
                
            resultadosV.append (float (valorAleatorio))
    
        # Creacion del dataset y su asociacion en la variable global
        datasetOriginal = SimilarityDataset(fotogramas1V, fotogramas2V, resultadosV, self.INPUT_SIZE, self.TRANSFORMADOR)
        self.datasetValidacion = DataLoader(datasetOriginal, batch_size=self.TAMANO_LOTE, shuffle=True)
        
        return

    # ------------------------------------
        
    def entrenamiento_red_neuronal (self):

        # Obtiene el lote a procesar
        imagen1, imagen2, resultado = next (iter (self.datasetEntrenamiento))

        # Se pasan al dispositivo correspondiente
        imagen1Dispositivo = imagen1.to (self.DISPOSITIVO)
        imagen2Dispositivo = imagen2.to (self.DISPOSITIVO)
        resultadoDispositivo = resultado.float ().to (self.DISPOSITIVO)
        
        # Se le asigna el gradiente a cero (Aprende)
        self.optimizador.zero_grad ()

        # Prediccion de la red neuronal sobre las dos imagenes
        resultadoRedNeuronal = self.redNeuronalDeteccionSimilitud (imagen1Dispositivo, imagen2Dispositivo)

        # Calculo de la perdida entre la prediccion y la realidad
        valorPerdida = self.funcionPerdida (resultadoRedNeuronal, resultadoDispositivo.unsqueeze(1))

        # Lleva a cabo una retropropagacion de la perdida para el calculo de las modificaciones a llevar a cabo en las distintas capas de la red
        valorPerdida.backward ()

        # Actualiza los parametros de la red
        self.optimizador.step ()

        self.contadorPerdida += valorPerdida.item () * self.TAMANO_LOTE
        
        # Incrementamos el numero de imagenes procesadas
        self.contadorFotogramasProcesados += self.TAMANO_LOTE

        return
    
    # ---------------------------------
    
    def validacion_red_neuronal (self):
        
        imagen1, imagen2, resultado = next (iter (self.datasetEntrenamiento))
            
        # Se pasan al dispositivo correspondiente
        imagen1Dispositivo = imagen1.to (self.DISPOSITIVO)
        imagen2Dispositivo = imagen2.to (self.DISPOSITIVO)
        resultadoDispositivo = resultado.float ().to (self.DISPOSITIVO)
            
        with torch.no_grad():  # Deshabilita el gradiente para la validacion

            # Calculo de prediccion por la red neuronal
            resultadoRedNeuronal = self.redNeuronalDeteccionSimilitud(imagen1Dispositivo, imagen2Dispositivo)

            # Calculo de la perdida
            valorPerdida = self.funcionPerdida(resultadoRedNeuronal, resultadoDispositivo.unsqueeze(1))

            # Acumula la perdida por el tamaño del lote
            self.contadorPerdida += valorPerdida.item() * self.TAMANO_LOTE
            
            # Incremento del número de imagenes que se han procesado correctamente
            self.contadorFotogramasProcesados += self.TAMANO_LOTE
        
        return
    
    # ----------------------------------

    def mostrar_resultados_epoca (self):
        
        print ("Epoca[" + str(self.contadorEpocas) + "/" + str(self.NUMERO_EPOCAS) + "]", end="\t")        
        print ("Perdida entrenamiento (Medias): " + str(self.listaPerdidasEntrenamiento[-1]), end="\t")
        print ("Perdida validacion (Medias): " + str(self.listaPerdidasValidacion[-1]), end="\t")
        print ("")

        return
    
    # -------------------------------

    def guardar_modelo_mostrar_resultados (self):

        print ("----------------- FIN ENTRENAMIENTO -----------------\n\n")
        
        # Guardado del modelo de red neuronal
        torch.save(self.redNeuronalDeteccionSimilitud.state_dict(), self.rutaDestinoResultados + "/weightModel.pth")
        
        # Calculo del tiempo total de ejecucion
        self.tiempoEjecucion = time.time() - self.tiempoEjecucion
        
        # Guardado en un string de los resultados por epoca de perdidas en entrenamiento y validacion
        resultadosPorEpoca = [f"EPOCA[{i+1}/{self.NUMERO_EPOCAS}]  -  Perdida Entrenamiento: {self.listaPerdidasEntrenamiento[i]}  -  Perdida validacion: {self.listaPerdidasValidacion[i]}" for i in range(self.NUMERO_EPOCAS)]

        # Creacion de diccionario para guardar la informacion en formato JSON
        resultadoFormatoJSON = {"Tiempo de conexion" : self.tiempoEjecucion,
                                "Ruta destino del modelo" : self.rutaDestinoResultados + "/weightModel.pth",
                                "Numero de epocas" : self.NUMERO_EPOCAS,
                                "Tasa de aprendizaje" : self.TASA_DE_APRENDIZAJE,
                                "Tamano de batch" : self.TAMANO_LOTE,
                                "Resultados entrenamiento y validacion" : resultadosPorEpoca
                                }
        
        # Resultados por archivo JSON (Permanentes)
        if os.path.exists (os.path.dirname (self.rutaDestinoResultados)):
            with open (self.rutaDestinoResultados + "/data.json", 'w') as flujoSalida:
                json.dump (resultadoFormatoJSON, flujoSalida)
                
        # Configura el grafico
        plt.figure(figsize=(20, 12))
        plt.plot(list(range(1, self.NUMERO_EPOCAS + 1)), self.listaPerdidasEntrenamiento, 'b', label='Training Loss')
        plt.plot(list(range(1, self.NUMERO_EPOCAS + 1)), self.listaPerdidasValidacion, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Guarda la gráfica en disco
        plt.savefig (self.rutaDestinoResultados + "/training_validation_loss_plot.png")
        
        print ("----------------- INICIO RESULTADOS -----------------")
        # Resultados por consola (Temporales)
        print ("Tiempo de conexion:", self.tiempoEjecucion, "segundos")
        print ("Numero de epocas:", str(self.NUMERO_EPOCAS), "epocas")
        print ("Tasa de aprendizaje:", self.TASA_DE_APRENDIZAJE)
        print ("Tamano de lote:", str (self.TAMANO_LOTE), "imagenes por iteraccion")
        print ("Numero imagenes entrenamiento:", str (self.NUMERO_IMAGENES_ENTRENAMIENTO))
        print ("Numero imagenes validacion:", str (self.NUMERO_IMAGENES_VALIDACION))        
        print ("Ruta destino de los resultados:", self.rutaDestinoResultados)
        print ("----------------- FIN RESULTADOS -----------------\n\n")
        
        return