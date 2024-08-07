# Librerías del framework
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from genericworker import *

# Librerías necesarias especificas
import pyrealsense2 as pr2
from ultralytics import YOLO

# Librerías generales
import cv2 as cv
import numpy as np
import os
import warnings

# Dataset y Red Neuronal
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Ignora un error que aparece al principio y que no afecta al código 
warnings.filterwarnings("ignore", message=".*cudnn.*", category=UserWarning)

# ----------------------------------

class DatasetPersonalizado(Dataset):
    def __init__(self, imagenesEntrada, resultadosImagenes):
        self.imagenesEntrada = imagenesEntrada
        self.resultadosImagenes = resultadosImagenes
        transformacion = transforms.Compose([
            transforms.ToPILImage(),  # Conversión de numpy a PIL Image
            transforms.Resize((350, 150)),  # Redimensiona la imagen
            transforms.ToTensor(),  # Conversion de imagen a tensor
        ])              
        
    def __len__(self):
        return len(self.imagenesEntrada)

    def __getitem__(self, indice):
        
        imagenTransformada = self.transformacion(self.imagenesEntrada[indice])
        
        resultados = self.resultadosImagenes[indice]
        
        return imagenTransformada, resultados

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
    periodo = 33
    
    # Grabación
    conexionGrabacion = None
    rutaGrabacion = "/media/robocomp/data_tfg/oficialVideos/video1.bag"
    
    # Red neuronal YOLO
    redNeuronalYOLO = None
    PRECISION_MINIMA_YOLO = 0.8
    
    # Red neuronal encargada del proceso de trackeo
    redNeuronalEleccionObjetivo = None
    rutaParametrosRedNeuronal = "/home/robocomp/pruebas/model_state.pth"
    optimizador = None
    funcionPerdida = None
    historialPerdidasEpoca = []
        
    # Parametros de la red neuronal y sus componentes
    PRECISION_MINIMA_SEGUIMIENTO = 0.8
    TASA_APRENDIZAJE = 0.001
    MOMENTUM = 0.9
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Conversión de numpy a PIL Image
        transforms.Resize((350, 150)),  # Redimensiona la imagen
        transforms.ToTensor(),  # Conversion de imagen a tensor
    ])
    
    # Entrenamiento de red neuronal
    TAMANO_ENTRADA = [350, 150, 3 ]
    IMAGENES_POR_EPOCA = 1000
    batchSize = 32
    
    # Variables para entrenamiento (Modificables)    
    entradaEntrenamiento = []
    resultadoEntrenamiento = []
    perdidasEpoca = None
    contadorImagenesProcesadas = None
    contadorEpocas = None
    
    
    contadorImagenes = None
    
    # Asigna el dispositivo
    DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUMERO_DECIMALES = 8
    
    # -------------------------------------------------
    
    # -------------------------------------------------
    
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = self.periodo

        # Comprobación de requisitos mínimos
        self.comprobacion_requisitos_minimos ()

        # Conexion con el fichero de la grabación
        self.iniciar_conexion_grabacion ()
        
        # Prepara el entorno
        self.preparacion_entorno ()
        
        # Arranca el timer
        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

    # ----------------
    
    def __del__(self):
        self.conexionGrabacion.stop ()
        
        return

    # --------------------------

    def setParams(self, params):
        self.contadorImagenesProcesadas = 0
        self.perdidasEpoca = 0
        self.contadorEpocas = 0
        
        self.contadorImagenes = 0
        return

    # ----------------

    @QtCore.Slot()
    def compute(self):
        
        print ("hola")

        # Recepcion de fotograma junto con un flag que indica si se ha recibido o no.
        hayFotograma, fotogramas = self.conexionGrabacion.try_wait_for_frames ()
                
        # Si hay fotogramas actua, si no, no hace nada
        if hayFotograma:
            fotogramaColor, fotogramaProfundidad = self.preparacion_fotogramas (fotogramas)
            
            # Procesa la imagen para obtener los resultados (Bounding Boxes de personas)
            resultados = self.redNeuronalYolo (fotogramaColor, verbose=False)
            
            # Separación de datos, obtención del indice persona objetivo, aplicación de resultados
            cajaColisiones = self.separacion_filtracion_resultados (resultados)
            indiceObjetivo = self.obtencion_indice_persona_objetivo (fotogramaColor, cajaColisiones)
            fotogramaConResultados = self.aplicar_resultados_fotograma (fotogramaColor, cajaColisiones, indiceObjetivo)
            
            # Interfaz de usuario (Muestra imagen al usuario)
            self.interfaz_usuario (fotogramaColor, fotogramaConResultados)     
            
            self.contadorImagenes += 1
            print ("Numero Imagenes:", self.contadorImagenes)
            
        else:
            print ("Error")
            sys.exit ("Testing")

        return
    
    # -----------------------------------------
    
    def comprobacion_requisitos_minimos (self):
        # Comprobacion de que existe la ruta de la grabacion
        if not os.path.exists (self.rutaGrabacion):
            sys.exit ("ERROR (1): La ruta de la grabacion no existe. Ruta: " + self.rutaGrabacion)
        
        if not os.path.exists (self.rutaParametrosRedNeuronal):
            sys.exit ("ERROR (2): La ruta de los parámetros de la red no existe. Ruta: " + self.rutaParametrosRedNeuronal)
        
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
    
    # -----------------------------
    
    def preparacion_entorno (self):
        # Lo primero es cargar el modelo de red neuronal de yolo (Hay distitos modelos, mirar en la web de ultralytics)
        self.redNeuronalYolo = YOLO ("yolov8s.pt")

        # Red neuronal de seguimiento
        self.redNeuronalEleccionObjetivo = ResNetRegresion ()
        self.redNeuronalEleccionObjetivo.load_state_dict(torch.load(self.rutaParametrosRedNeuronal))
        self.redNeuronalEleccionObjetivo = self.redNeuronalEleccionObjetivo.to (self.DISPOSITIVO)
        
        # Parametros red neuronal de seguimiento (Entrenamiento y resultados)
        self.funcionPerdida = nn.MSELoss()
        self.optimizador = torch.optim.SGD(self.redNeuronalEleccionObjetivo.parameters(), lr=self.TASA_APRENDIZAJE, momentum=self.MOMENTUM) # Momentum = convergencia
        return

    # --------------------------------------------
    
    def preparacion_fotogramas (self, fotogramas):
        # Obtención de datos de los fotogramas
        fotogramaColor = fotogramas.get_color_frame().get_data ()
        fotogramaProfundidad = fotogramas.get_depth_frame().get_data ()
       
        # Se convierte en array para poder procesarlo con la red neuronal
        fotogramaColorArray = np.asanyarray(fotogramaColor)
        fotogramaProfundidadArray = np.asanyarray(fotogramaColor)
       
        return fotogramaColorArray, fotogramaProfundidadArray

    # ------------------------------------------------------
    
    def separacion_filtracion_resultados (self, resultados):
        # Se crean listas vacias para guardar la información
        listaCajaColisionesDetecciones = []

        # Se separan los resultados
        for deteccion in resultados[0].boxes:

            # Si es una persona los resultados se tienen que guardar. Si no, no interesan (Intento de mejora de eficiencia)
            if deteccion.cls == 0 and deteccion.conf > self.PRECISION_MINIMA_YOLO:

                # Para las cajas de colision son 4 valores en lugar de uno
                listaCajaColisionesDetecciones.append ([int(coordenada.item()) for coordenada in deteccion.xyxy.to('cpu')[0]]) 

        return listaCajaColisionesDetecciones

    # ---------------------------------------------------------------------------
    
    def obtencion_indice_persona_objetivo (self, imagenOriginal, cajaColisiones):
        indiceObjetivo = -1
        valorMaximo = 0
        
        for i in range (len (cajaColisiones)):
            roi = imagenOriginal[cajaColisiones[i][1]:cajaColisiones[i][3], cajaColisiones[i][0]:cajaColisiones[i][2]]

            # Transforma la imagen para poder ser procesada y le añade la dimension de batch (Es una sola imagen por batch)
            imagenPreparada = self.transform(roi)
            imagenPreparada = imagenPreparada.unsqueeze(0)
            imagenPreparadaDispositivo = imagenPreparada.to (self.DISPOSITIVO)
            
            # En modo evaluación la red neuronal no se modifica (No aprende)
            self.redNeuronalEleccionObjetivo.eval ()
            
            # Obtiene una predicción del roi insertado
            with torch.no_grad():
                prediccion = self.redNeuronalEleccionObjetivo(imagenPreparadaDispositivo)
                prediccion = prediccion.item ()
            
            # Actualiza el mejor hasta ahora
            if valorMaximo < prediccion and prediccion > self.PRECISION_MINIMA_SEGUIMIENTO:
                indiceObjetivo = i
                valorMaximo = prediccion
        
        return indiceObjetivo

    # -----------------------------------------------------------------------------------------
    
    def aplicar_resultados_fotograma (self, fotogramaOriginal, cajaColisiones, indiceObjetivo):
        # Primero se dibujan las bounding boxes sobre la imagen
        fotogramaConResultados = fotogramaOriginal.copy ()

        # Se hace manualmente ya que la opcion que ofrece la librería muestra todas las cajas de colisiones y solo interesan las personas
        for i in range (len (cajaColisiones)):

            # Se asigna un color distinto dependiendo si la persona es la objetivo o no
            if indiceObjetivo == i:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            # Se dibuja un rectangulo simulando la bounding box (Verde si es la persona objetivo y roja si no)
            cv.rectangle (fotogramaConResultados, 
                          (cajaColisiones[i][0], cajaColisiones[i][1]),
                          (cajaColisiones[i][2], cajaColisiones[i][3]),
                          color,
                          2
                          )
          
            
        return fotogramaConResultados
    
    # ------------------------------------------------------------------
    
    def interfaz_usuario (self, fotogramaColor, fotogramaConResultados):
        # Se muestran por la interfaz de opencv cuyas ventanas tienen asignadas el nombre indicado
        cv.imshow ("Fotogramas Color", fotogramaColor)
        cv.imshow ("Fotogramas Profundidad", fotogramaConResultados) 
        
        # Gestiona la tecla pulsada
        self.controlador_teclas (cv.waitKey (1))
        
        return
    
    # ------------------------------------------
    
    def controlador_teclas (self, letraPulsada):
        
        # Si el valor es -1 siginifca que no se pulso ninguna tecla (No merece la pena hacer ninguna comprobacion)
        if letraPulsada != -1:

            # Se ha pulsado la letra ESC
            if letraPulsada == 27:
                sys.exit ("FIN EJECUCION: Presionada tecla ESC")

        
        return