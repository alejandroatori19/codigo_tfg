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

# Ignora un error que aparece al principio y que no afecta al código 
warnings.filterwarnings("ignore", message=".*cudnn.*", category=UserWarning)

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
    
    # Camara
    conexionCamara = None
    NUMERO_SERIE_CAMARA = "146222252950"
    RESOLUCION_GRABACION = [640, 480]               # Anchura y altura de los fotogramas
    TASA_FOTOGRAMAS_SEGUNDO = 30
    
    # Red neuronal YOLO
    redNeuronalYOLO = None
    PRECISION_MINIMA_YOLO = 0.8
    
    # Red neuronal encargada del proceso de trackeo
    redNeuronalEleccionObjetivo = None
    rutaParametrosRedNeuronal = "/home/robocomp/pruebas/model_state.pth"
        
    # Parametros de la red neuronal y sus componentes
    PRECISION_MINIMA_SEGUIMIENTO = 0.8
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Conversión de numpy a PIL Image
        transforms.Resize((350, 150)),  # Redimensiona la imagen
        transforms.ToTensor(),  # Conversion de imagen a tensor
    ])
    
    # Asigna el dispositivo
    DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -------------------------------------------------
    
    # -------------------------------------------------
    
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = self.periodo

        # Comprobación de requisitos mínimos
        self.comprobacion_requisitos_minimos ()

        # Conexion con el fichero de la grabación
        self.iniciar_conexion_cámara ()
        
        # Prepara el entorno
        self.preparacion_entorno ()
        
        # Arranca el timer
        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

    # ----------------
    
    def __del__(self):
        self.conexionCamara.stop ()
        
        return

    # --------------------------

    def setParams(self, params):
        
        return

    # ----------------

    @QtCore.Slot()
    def compute(self):
        # Recepcion de fotograma junto con un flag que indica si se ha recibido o no.
        hayFotograma, fotogramas = self.conexionCamara.try_wait_for_frames ()
                
        # Si hay fotogramas actua, si no, no hace nada
        if hayFotograma:
            fotogramaColor = self.preparacion_fotogramas (fotogramas)
            
            # Procesa la imagen para obtener los resultados (Bounding Boxes de personas)
            resultados = self.redNeuronalYolo (fotogramaColor, verbose=False)
            
            # Separación de datos, obtención del indice persona objetivo, aplicación de resultados
            cajaColisiones = self.separacion_filtracion_resultados (resultados)
            indiceObjetivo = self.obtencion_indice_persona_objetivo (fotogramaColor, cajaColisiones)
            fotogramaConResultados = self.aplicar_resultados_fotograma (fotogramaColor, cajaColisiones, indiceObjetivo)
            
            # Interfaz de usuario (Muestra imagen al usuario)
            self.interfaz_usuario (fotogramaColor, fotogramaConResultados)     
            
        else:
            sys.exit ("Testing")

        return
    
    # -----------------------------------------
    
    def comprobacion_requisitos_minimos (self):
        if not os.path.exists (self.rutaParametrosRedNeuronal):
            sys.exit ("ERROR (2): La ruta de los parámetros de la red no existe. Ruta: " + self.rutaParametrosRedNeuronal)
        
        return
    
    # ------------------------------------
    
    def iniciar_conexion_cámara (self):
        # Se generan los objetos de conexion y configuracion
        self.conexionCamara = pr2.pipeline ()
        configuracion = pr2.config ()
    
        # Se genera una conexion con la cámara (Activa el dispositivo dentro de la configuracion)
        configuracion.enable_stream (self.NUMERO_SERIE_CAMARA)

        # Se establecen unas configuraciones para los flujos de datos
        configuracion.enable_stream (pr2.stream.color,                  # Flujo de fotogramas de color
                                     self.RESOLUCION_GRABACION[0], 
                                     self.RESOLUCION_GRABACION[1], 
                                     pr2.format.bgr8,
                                     self.TASA_FOTOGRAMAS_SEGUNDO
                                     )

        configuracion.enable_stream (pr2.stream.depth,                  # Flujo de fotogramas de profundidad
                                     self.RESOLUCION_GRABACION[0], 
                                     self.RESOLUCION_GRABACION[1], 
                                     pr2.format.z16,
                                     self.TASA_FOTOGRAMAS_SEGUNDO
                                     )
        
        # Se inicia la conexion con la configuracion predefinida
        self.conexionCamara.start (configuracion)
        
        return
    
    # -----------------------------
    
    def preparacion_entorno (self):
        # Lo primero es cargar el modelo de red neuronal de yolo (Hay distitos modelos, mirar en la web de ultralytics)
        self.redNeuronalYolo = YOLO ("yolov8s.pt")

        # Red neuronal de seguimiento
        self.redNeuronalEleccionObjetivo = ResNetRegresion ()
        self.redNeuronalEleccionObjetivo.load_state_dict(torch.load(self.rutaParametrosRedNeuronal))
        self.redNeuronalEleccionObjetivo = self.redNeuronalEleccionObjetivo.to (self.DISPOSITIVO)
        
        return

    # --------------------------------------------
    
    def preparacion_fotogramas (self, fotogramas):
        # Obtención de datos de los fotogramas
        fotogramaColor = fotogramas.get_color_frame().get_data ()
        #fotogramaProfundidad = fotogramas.get_depth_frame().get_data ()
       
        # Se convierte en array para poder procesarlo con la red neuronal
        fotogramaColorArray = np.asanyarray(fotogramaColor)
        #fotogramaProfundidadArray = np.asanyarray(fotogramaColor)
       
        #return fotogramaColorArray, fotogramaProfundidadArray
        return fotogramaColorArray

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