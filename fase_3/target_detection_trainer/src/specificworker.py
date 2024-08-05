from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from genericworker import *

import cv2 as cv
import numpy as np
import sys
import random
import os
import json

# Dataset y Red Neuronal
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Matplotlib para hacer graficos
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# ----------------------------------

class DatasetPersonalizado(Dataset):
    def __init__(self, rutas_imagenes, etiquetas, tamano_entrada=[350, 150, 3]):
        self.rutas_imagenes = rutas_imagenes
        self.etiquetas = etiquetas
        self.tamanoEntrada = tamano_entrada
        
        # Transformación de datos
        self.transformacion = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.rutas_imagenes)

    def __getitem__(self, indice):
        imagen = cv.imread(self.rutas_imagenes[indice])
        imagenRedimensionada = cv.resize (imagen, (self.tamanoEntrada[1], self.tamanoEntrada[0]))
        
        imagenTransformada = self.transformacion(imagenRedimensionada)
        
        etiqueta = self.etiquetas[indice]
        
        return imagenTransformada, etiqueta
    
# -------------------------------

class ResNetRegresion(nn.Module):
    def __init__(self):
        super(ResNetRegresion, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x
    
# ----------------------------------

class SpecificWorker(GenericWorker):
    # Referencias timers
    periodo = 33
    
    # Referencia dataset
    rutaDataset = "/media/robocomp/data_tfg/dataset_clasificado"
    porcentajeValidacion = 0.1
    cargadorDatasetEntrenamiento = None
    cargadorDatasetValidacion = None
    
    # Modificadores red neuronal
    destinoModelo = "/home/robocomp/pruebas"
    redNeuronal = None
    funcionPerdida = None
    optimizador = None
    perdidasEntrenamiento = []
    perdidasValidacion = []
    
    # Variables modificables para entrenamiento
    epoca = None
    estaEntrenando = None
    contadorImagenesProcesadas = None
    perdidasProceso = None
    
    # Flags y constantes para entrenamiento
    batchSize = 32
    numeroEpocas = 10
    tasaAprendizaje = 0.001
    tamanoEntrada = [350, 150, 3]
    numeroImagenesEntrenamiento = None
    numeroImagenesValidacion = None

    # Flags
    REEMPLAZAR_MODELO = False
    NUMERO_DECIMALES = 8
    DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -------------------------------------------------
    
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = self.periodo
        
        # Comprobacion iniciales
        self.comprobacion_requisitos_minimos ()
        
        # Prepara el entorno (Carga del dataset, inicializa la red neuronal y sus componentes)
        self.preparacion_entorno ()
        
        # Inicio del timer
        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)
        
        return

    # ----------------

    def __del__(self):
        
        
        return
    
    # --------------------------

    def setParams(self, params):
        self.epoca = 0
        self.contadorImagenesProcesadas = 0
        self.perdidasProceso = 0
        self.estaEntrenando = True
        
        
        return

    # ----------------
    
    @QtCore.Slot()
    def compute(self):
        
        # Entrenamiento
        if self.estaEntrenando:
            
            # Se entrena y se obtiene una perdida del lote correspondiente, aumeenta el numero de imagenes procesadas, etc.
            self.entrenamiento_red_neuronal ()
                        
            if self.contadorImagenesProcesadas >= self.numeroImagenesEntrenamiento:
                # Se cambia el modo de funcionamiento de entrenamiento a validacion
                self.estaEntrenando = False
                
                # Reset y adicción de variables
                self.perdidasEntrenamiento.append (round (self.perdidasProceso / self.contadorImagenesProcesadas, self.NUMERO_DECIMALES))
                self.contadorImagenesProcesadas = 0
                self.perdidasProceso = 0
                
        # Validacion 
        else:
            
            self.validacion_red_neuronal ()
            
            if self.contadorImagenesProcesadas >= self.numeroImagenesValidacion:        
                # Se cambia el modo de funcionamiento de entrenamiento a validacion
                self.estaEntrenando = True
                
                # Reset y adicción de variables
                self.perdidasValidacion.append (round (self.perdidasProceso / self.contadorImagenesProcesadas, self.NUMERO_DECIMALES))
                self.contadorImagenesProcesadas = 0
                self.perdidasProceso = 0
                self.epoca += 1
                
                self.mostrar_resultados_epoca ()
                
                if self.numeroEpocas <= self.epoca:
                    self.generacion_de_resultados ()
                    sys.exit ("FIN: Se ha acabado el código debido a que se han alcanzado el numero de epocas especificado")
        
        return


    # METODOS SECUNDARIOS DE APOYO A LOS PRINCIPALES
    def comprobacion_requisitos_minimos (self):
        # Comprueba que existe la carpeta con el contenido del dataset sin clasificar
        if not os.path.exists (self.rutaDataset):
            sys.exit ("ERROR (1): El directorio que contiene el dataset no existe - Ruta: " + self.rutaDataset + "\n\n")
            
        # Comprueba que esta dentro del rango
        if self.porcentajeValidacion < 0 and self.porcentajeValidacion > 1:
            sys.exit ("ERROR (2): El valor del porcentaje de datos que se usa para validacion esta fuera del rango.\n\n")
    
        # Comprueba que existe la ruta con el destino del dataset clasificado (El directorio padre)
        if not os.path.exists (self.destinoModelo):
            sys.exit ("ERROR (3): Comprueba que el directorio destino del modelo existe - Ruta: " + self.destinoModelo + "\n\n")
            
        # Si no esta habilitado el reemplazo entonces acaba (Es un control de seguridad para evitar posibles errores)
        if not self.REEMPLAZAR_MODELO and os.path.exists (self.destinoModelo + "/model_state.pth"): 
            sys.exit ("ERROR (4): No esta habilitado el reemplazo de dataset\n\n")
        return
    
    # -----------------------------
    
    def preparacion_entorno (self):
        
        # Cargar dataset
        x_datos_entrenamiento, y_datos_entrenamiento, x_datos_validacion, y_datos_validacion = self.cargar_dataset ()
        
        self.numeroImagenesEntrenamiento = len (x_datos_entrenamiento)
        self.numeroImagenesValidacion = len (x_datos_validacion)
        
        # Creación del dataset
        datasetEntrenamiento = DatasetPersonalizado (x_datos_entrenamiento, y_datos_entrenamiento)
        datasetValidacion = DatasetPersonalizado (x_datos_validacion, y_datos_validacion)
        
        # Creacion de dataLoader
        self.cargadorDatasetEntrenamiento = DataLoader(datasetEntrenamiento, batch_size = self.batchSize, shuffle=True)
        self.cargadorDatasetValidacion = DataLoader(datasetValidacion, batch_size = self.batchSize, shuffle=True)
        
        # Cargar red neuronal
        self.redNeuronal = ResNetRegresion ()
        self.redNeuronal = self.redNeuronal.to (self.DISPOSITIVO)
        
        self.funcionPerdida = nn.MSELoss()
        self.optimizador = torch.optim.SGD(self.redNeuronal.parameters(), lr=self.tasaAprendizaje, momentum=0.9) # Momentum = convergencia mas rapida
    
        return

    # ------------------------

    def cargar_dataset (self):
        # Lectura de las rutas (Objetivo y no objetivo)
        fotogramasObjetivo = [self.rutaDataset + "/persona_objetivo/" + nombreArchivo for nombreArchivo in os.listdir(self.rutaDataset + "/persona_objetivo")]
        fotogramasNoObjetivo = [self.rutaDataset + "/persona_no_objetivo/" + nombreArchivo for nombreArchivo in os.listdir(self.rutaDataset + "/persona_no_objetivo")]

        # Unificacion de los datos
        fotogramasGeneral = fotogramasObjetivo + fotogramasNoObjetivo
        resultadosGeneral = [1] * len(fotogramasObjetivo) + [0] * len(fotogramasNoObjetivo)
        
        x_datos_entrenamiento = []
        y_datos_entrenamiento = []
        x_datos_validacion = []
        y_datos_validacion = []
        
        # Se cargan todas las imagenes
        while len (fotogramasGeneral) > 0:
            indiceAleatorio = random.randint (0, len (fotogramasGeneral) - 1)
            
            # Generación de dataset dentro del código
            x_datos_entrenamiento.append (fotogramasGeneral.pop (indiceAleatorio))
            y_datos_entrenamiento.append (resultadosGeneral.pop (indiceAleatorio))
            
        # Con todas las imagenes cargadas se genera el dataset de validación a partir de un porcentaje del dataset de entrenamiento.
        if self.porcentajeValidacion > 0:
            numeroDatosValidacion = int (len (x_datos_entrenamiento) * self.porcentajeValidacion)
            
            for i in range (numeroDatosValidacion):
                indiceAleatorio = random.randint (0, len (x_datos_entrenamiento) - 1)
                
                x_datos_validacion.append (x_datos_entrenamiento.pop (indiceAleatorio))
                y_datos_validacion.append (y_datos_entrenamiento.pop (indiceAleatorio))
               
        return x_datos_entrenamiento, y_datos_entrenamiento, x_datos_validacion, y_datos_validacion

    # ------------------------------------

    def entrenamiento_red_neuronal (self):
        # Obtencion de datos
        imagenes, etiquetas = next(iter(self.cargadorDatasetEntrenamiento))
        
        # Se pasa a los dispositivos correspondientes
        imagenesDispositivo = imagenes.to (self.DISPOSITIVO)
        etiquetasDispositivo = etiquetas.float ().to (self.DISPOSITIVO)
        
        # Se le asigna al modelo el modo de entrenamiento
        self.redNeuronal.train ()
        
        # Se le asigna el gradiente a cero (Aprende)
        self.optimizador.zero_grad ()

        # Prediccion de la red neuronal sobre las imagenes (Lote de x imagenes)
        resultadoRedNeuronal = self.redNeuronal (imagenesDispositivo)

        # Calculo de la perdida entre la prediccion y la realidad
        perdidaBatch = self.funcionPerdida (resultadoRedNeuronal, etiquetasDispositivo.unsqueeze(1))

        # Lleva a cabo una retropropagacion de la perdida para el calculo de las modificaciones a llevar a cabo en las distintas capas de la red
        perdidaBatch.backward ()

        # Actualiza los parametros de la red
        self.optimizador.step ()

        self.perdidasProceso += perdidaBatch.item () * self.batchSize
        
        # Incrementamos el numero de imagenes procesadas
        self.contadorImagenesProcesadas += self.batchSize

        return
    
    # -----------------------------
    
    def validacion_red_neuronal (self):
        # Obtencion de datos
        imagenes, etiquetas = next(iter(self.cargadorDatasetValidacion))
        
        # Se pasa a los dispositivos correspondientes
        imagenesDispositivo = imagenes.to (self.DISPOSITIVO)
        etiquetasDispositivo = etiquetas.float ().to (self.DISPOSITIVO)
        
        # Se asigna el modelo en modo evaluacion
        self.redNeuronal.eval ()
        
        # Deshabilita el gradiente para la validacion
        with torch.no_grad():  

            # Calculo de prediccion por la red neuronal
            resultadoRedNeuronal = self.redNeuronal(imagenesDispositivo)

            # Calculo de la perdida
            valorPerdida = self.funcionPerdida(resultadoRedNeuronal, etiquetasDispositivo.unsqueeze(1))

            # Acumula la perdida por el tamaño del lote
            self.perdidasProceso += valorPerdida.item() * self.batchSize
            
            # Incremento del número de imagenes que se han procesado correctamente
            self.contadorImagenesProcesadas += self.batchSize
        
        return

    # ----------------------------------

    def mostrar_resultados_epoca (self):
        
        print ("Epoca[" + str(self.epoca) + "/" + str(self.numeroEpocas) + "]", end="\t")        
        print ("Perdida entrenamiento (Medias): " + str(self.perdidasEntrenamiento[-1]), end="\t")
        print ("Perdida validacion (Medias): " + str(self.perdidasValidacion[-1]), end="\t")
        print ("")

        return

    # ----------------------------------

    def generacion_de_resultados (self):
        
        print ("----------------- FIN ENTRENAMIENTO -----------------\n\n")
        
        # Guardado del modelo de red neuronal
        torch.save(self.redNeuronal.state_dict(), self.destinoModelo + "/model_state.pth")
        
        
        # Guardado en un string de los resultados por epoca de perdidas en entrenamiento y validacion
        resultadosPorEpoca = [f"EPOCA[{i+1}/{self.numeroEpocas}]  -  Perdida Entrenamiento: {self.perdidasEntrenamiento[i]}  -  Perdida validacion: {self.perdidasValidacion[i]}" for i in range(self.numeroEpocas)]
        
        
        # Creacion de diccionario para guardar la informacion en formato JSON
        resultadoFormatoJSON = {"Ruta destino del modelo" : self.destinoModelo + "/model_state.pth",
                                "Numero de epocas" : self.numeroEpocas,
                                "Tasa de aprendizaje" : self.tasaAprendizaje,
                                "Tamano de batch" : self.batchSize,
                                "Resultados entrenamiento y validacion" : resultadosPorEpoca
                                }
        
        # Resultados por archivo JSON (Permanentes)
        with open (self.destinoModelo + "/data.json", 'w') as flujoSalida:
            json.dump (resultadoFormatoJSON, flujoSalida)
            
            
        # Configura el grafico
        plt.figure(figsize=(20, 12))
        plt.plot(list(range(1, self.numeroEpocas + 1)), self.perdidasEntrenamiento, 'b', label='Training Loss')
        plt.plot(list(range(1, self.numeroEpocas + 1)), self.perdidasValidacion, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Guarda la gráfica en disco
        plt.savefig (self.destinoModelo + "/training_validation_loss_plot.png")
        
        # Resultados por consola (Temporales)
        print ("----------------- INICIO RESULTADOS -----------------")
        print ("Numero de epocas:", str(self.numeroEpocas), "epocas")
        print ("Tasa de aprendizaje:", self.tasaAprendizaje)
        print ("Tamano de lote:", str (self.batchSize), "imagenes por iteraccion")
        print ("Numero imagenes entrenamiento:", str (self.numeroImagenesEntrenamiento))
        print ("Numero imagenes validacion:", str (self.numeroImagenesValidacion))        
        print ("Ruta destino de los resultados:", self.destinoModelo)
        print ("----------------- FIN RESULTADOS -----------------\n\n")
        
        return