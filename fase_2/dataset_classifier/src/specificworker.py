from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from genericworker import *

# Importacion de librerías
from ultralytics import YOLO            # Librería para la red neuronal
import cv2 as cv                        # Interfaz de usuario
import numpy as np                      # Gestion de imagenes
import json                             # Guardado de resultados
import sys                              # Control de errores controlado
import os                               # Gestion de rutas, archivos y directorios
import time                             # Para medir tiempos de ejecucion
import shutil                           # Para borrar directorios con contenido (Borrado recursivo)

class SpecificWorker(GenericWorker):

    periodo = 15            # A 30 FPS entonces 1000/30 = 33,... -> 33

    rutaDatasetSinClasificar = "/media/robocomp/data_tfg/dataset/colorFrames"   # Debe ser un directorio
    rutaDestinoDatasetClasificado = "/media/robocomp/data_tfg"                                  # Debe ser un directorio
    rutaDestinoResultadosJSON = "/home/robocomp/data.json"        # Un archivo con extension .json

    lista_ruta_absoluta_fotogramas = []

    # Red neuronal
    redNeuronalYolo = None

    # Flags
    PREVISUALIZACION_VIDEO = True
    REEMPLAZAR_DATASET_EXISTENTE = True
    PRECISION_MINIMA_ACEPTABLE = 0.80           # A mayor precision se le asigne más necesidad del usuario tendrá (Si esta relativamente lejos pierde precision de deteccion)
    MSE_MAXIMO_ACEPTABLE = 80

    # Flags modificables por codigo
    PERSONA_OBJETIVO_ENCONTRADA = False

    # Variables para generacion de resultados
    contadorFotogramas = None
    tiempoEjecucion = None
    regionInteresAnterior = None

    # CONSTRUCTOR, DESTRUCTOR Y METODOS PRINCIPALES
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        
        # Se comprueba que se cumplan los requisitos minimos
        self.comprobacion_requisitos_minimos ()

        self.preparacion_entorno ()

        # Se activa el temporizador
        self.timer.timeout.connect(self.compute)
        self.timer.start(self.periodo)

        return
    
    # --------------------------

    def setParams(self, params):
        self.contadorFotogramas = 0             # Se encargara de contar cuantos frames se estan guardando en el video
        self.tiempoEjecucion = time.time ()

        return True

	# ----------------

    def __del__(self):

        return

	# ----------------
	
    @QtCore.Slot()
    def compute(self):
        # Comprueba si se han procesado todos los frames, si no continua procesando
        if self.contadorFotogramas < len (self.lista_ruta_absoluta_fotogramas):

            # Carga el fotograma a procesar
            fotogramaActual = cv.imread (self.lista_ruta_absoluta_fotogramas[self.contadorFotogramas])

            # Procesa la imagen y obtiene unos resultados (Verbose anula los outputs de la propia libreria (Solo ocupan espacio pero no interesan))
            resultados = self.redNeuronalYolo (fotogramaActual, verbose=False)
            #resultados = self.redNeuronalYolo (fotogramaActual)

            # Separacion de resultados
            listaCajasColisiones, listaPrecisionDetecciones = self.resultados_filtrados (resultados)

            # Si no se tiene fotograma inciial del objetivo se tiene que encontrar una
            if not self.PERSONA_OBJETIVO_ENCONTRADA:
                indiceObjetivo = self.obtencion_indice_objetivo_manual (fotogramaActual, listaCajasColisiones)

            else:
                indiceObjetivo = -1
                
                # Obtencion de la persona objetivo (Si no la encuentra de manera automatica recurre al modo manual)
                indiceObjetivo = self.obtencion_indice_objetivo_automatico (fotogramaActual, listaCajasColisiones)
            
            self.guardado_rois_clasificados (fotogramaActual, listaCajasColisiones, indiceObjetivo)

            self.interfaz_usuario (fotogramaActual, listaPrecisionDetecciones, listaCajasColisiones, indiceObjetivo)


            self.contadorFotogramas += 1

            print ("\n\n -------------- NUEVA ITERACION -------------- \n\n")

        else:
            self.generacion_de_resultados ()
            sys.exit ("FIN EJECUCION (1): Se han procesado ya todos los frames")
        
        return


    # METODOS SECUNDARIOS DE APOYO A LOS PRINCIPALES
    def comprobacion_requisitos_minimos (self):
        # Comprueba que existe la carpeta con el contenido del dataset sin clasificar
        if not os.path.exists (self.rutaDatasetSinClasificar):
            sys.exit ("ERROR (1): El directorio que contiene el dataset sin clasificar no existe - Ruta: " + self.rutaDatasetSinClasificar + "\n\n")
        
        # Comprueba que existe la ruta con el destino del dataset clasificado (El directorio padre)
        if not os.path.exists (self.rutaDestinoDatasetClasificado):
            sys.exit ("ERROR (2): Comprueba que el directorio padre de la ruta destino de frames existe - Ruta: " + self.rutaDestinoDatasetClasificado + "\n\n")
        
        # Si no esta habilitado el reemplazo entonces acaba (Es un control de seguridad para evitar posibles errores)
        if not self.REEMPLAZAR_DATASET_EXISTENTE : 
            sys.exit ("ERROR (3): No esta habilitado el reemplazo de dataset\n\n")

        # Actualizacion de la ruta para asignarle el nombre del hijo (Carpeta que contendrá el dataset clasificado)
        self.rutaDestinoDatasetClasificado = self.rutaDestinoDatasetClasificado + "/dataset_clasificado"

        # Se comprueba que no exista (En caso de que si se borra)
        if os.path.exists (self.rutaDestinoDatasetClasificado):
            shutil.rmtree (self.rutaDestinoDatasetClasificado)

        # Se generan las carpetas dentro del directorio dataset clasificado
        os.makedirs (self.rutaDestinoDatasetClasificado)
        os.makedirs (self.rutaDestinoDatasetClasificado + "/persona_objetivo")
        os.makedirs (self.rutaDestinoDatasetClasificado + "/persona_no_objetivo")

        print ("INFORMACION (1) -> El primer paso se ha completado y los requisitos minimos se cumplen")
    
        return

    # -----------------------------

    def preparacion_entorno (self):
        # Lo primero es cargar el modelo de red neuronal de yolo (Hay distitos modelos, mirar en la web de ultralytics)
        self.redNeuronalYolo = YOLO ("yolov8s.pt")

        # Despues es cargar los paths absolutos de los archivos de imagen con la extension (.jpeg). Se asume que solo contiene imagenes
        for nombreArchivo in os.listdir (self.rutaDatasetSinClasificar):
            self.lista_ruta_absoluta_fotogramas.append (self.rutaDatasetSinClasificar + "/" + nombreArchivo)

        print ("INFORMACION (2) -> El segundo paso se ha completado y el entorno esta preparado")
        
        return

    # ------------------------------------------
    
    def resultados_filtrados (self, resultados):
        # Se crean listas vacias para guardar la información
        listaCajaColisionesDetecciones = []
        listaPrecisionDetecciones = []

        # Se separan los resultados
        for deteccion in resultados[0].boxes:

            # Si es una persona los resultados se tienen que guardar. Si no, no interesan (Intento de mejora de eficiencia)
            if deteccion.cls == 0 and deteccion.conf > self.PRECISION_MINIMA_ACEPTABLE:

                # Para las cajas de colision son 4 valores en lugar de uno
                listaCajaColisionesDetecciones.append ([int(coordenada.item()) for coordenada in deteccion.xyxy.to('cpu')[0]]) 
                listaPrecisionDetecciones.append (float (deteccion.conf.to("cpu")[0]))

        return listaCajaColisionesDetecciones, listaPrecisionDetecciones

    # ---------------------------------------------------------------------------

    def obtencion_indice_objetivo_manual (self, fotograma, listaCajasColisiones):
        print ("AVISO (1): Para el primer fotograma cuando se muestre la region de interes pertinente presione la siguiente tecla")
        print ("\tPresione ENTER cuando vea a la persona objetivo. Si no, pulse cualquiera")

        indiceObjetivo = -1
        contadorDetecciones = 0

        for cajaColision in listaCajasColisiones:
            regionInteres = fotograma[cajaColision[1]:cajaColision[3], cajaColision[0]:cajaColision[2]]

            cv.imshow ("Region interes", regionInteres)

            if self.controlador_teclas (letraPulsada=cv.waitKey (0)):
                indiceObjetivo = contadorDetecciones
                self.PERSONA_OBJETIVO_ENCONTRADA = True
                break

            contadorDetecciones += 1

        # Libera los recursos para no tener tantas ventanas de opencv y diferenciar
        cv.destroyWindow("Region interes")

        return indiceObjetivo

    # -------------------------------------------------------------------------------

    def obtencion_indice_objetivo_automatico (self, fotograma, listaCajasColisiones):
        mejorMSE = 500          # El valor maximo que he obtenido ha sido 120 o por ahi
        contador = 0
        indiceObjetivo = -1

        # Para cada caja de colision individual (Region de interes del fotograma actual)
        for cajaColision in listaCajasColisiones:

            # EXtraccion de la region de interes
            regionInteres = fotograma[cajaColision[1]:cajaColision[3], cajaColision[0]:cajaColision[2]]

            # Redimension de ambas regiones de interes a la misma (Preparacion)
            regionInteresAnteriorRedimensionada = cv.resize (self.regionInteresAnterior, (480, 640))
            regionInteresRedimensionada = cv.resize (regionInteres, (480, 640))

            # Conversion a escala de grises ambas regiones (Preparacion)
            regionInteresAnteriorEscalaGrises = cv.cvtColor (regionInteresAnteriorRedimensionada, cv.COLOR_BGR2GRAY)
            regionInteresEscalaGrises = cv.cvtColor(regionInteresRedimensionada, cv.COLOR_BGR2GRAY)

            # Obtencion de la similitud (Cuanto menor es la diferencia entre ambas mas parecidas son)
            indiceMSE = ((regionInteresAnteriorEscalaGrises - regionInteresEscalaGrises) ** 2).mean ()

            # Si supera cierto limite predefinido y es mejor que el mejor hasta ahora entonces se asigna como el mejor hasta ahora
            if (indiceMSE < self.MSE_MAXIMO_ACEPTABLE) and (mejorMSE > indiceMSE):
                indiceObjetivo = contador
                mejorMSE = indiceMSE

            # Incrementa el contador
            contador += 1

        # Es obligatorio que el usuario ofrezca un objetivo, en caso de no haber ningun objetivo se pasara (Se pone el contador para control de fallo por parte del usuario)
        #contador = 0
        
        #while indiceObjetivo == -1 and contador < 2:
        
        if indiceObjetivo == -1:
            indiceObjetivo = self.obtencion_indice_objetivo_manual (fotograma, listaCajasColisiones)
        #contador += 1

        return indiceObjetivo

    # ------------------------------------------------------------------------------------
    
    def guardado_rois_clasificados (self, fotograma, listaCajaColisiones, indiceObjetivo):
        # Lleva a cabo el conteo de detecciones
        contadorDetecciones = 0

        # Para todas las las bounding boxes
        for cajaColision in listaCajaColisiones:

            # Se obtiene la region de interes
            regionInteres = fotograma[cajaColision[1]:cajaColision[3], cajaColision[0]:cajaColision[2]]

            # Se le asigna la ruta (Si es persona objetivo o no lo es)
            if contadorDetecciones == indiceObjetivo:
                rutaFotograma = self.rutaDestinoDatasetClasificado + "/persona_objetivo/imagen_" + str(self.contadorFotogramas + 1) + "-" + str (contadorDetecciones) + ".jpeg"

                # Actualiza la region de interes anterior
                self.regionInteresAnterior = regionInteres.copy ()
                
            else:
                rutaFotograma = self.rutaDestinoDatasetClasificado + "/persona_no_objetivo/imagen_" + str(self.contadorFotogramas + 1) + "-" + str (contadorDetecciones) + ".jpeg"

            # Se guarda en disco en la ruta designada        
            cv.imwrite (rutaFotograma, regionInteres)

            # Incrementa el contador de detecciones
            contadorDetecciones += 1

        return

    # --------------------------------------------------------------------------------------------------------------------

    def interfaz_usuario (self, fotogramaOriginal, listaPrecisionDetecciones, listaCajaColisiones, indicePersonaObjetivo):
        # Primero se dibujan las bounding boxes sobre la imagen
        fotogramaConDetecciones = fotogramaOriginal.copy ()

        # Se hace manualmente ya que la opcion que ofrece la librería muestra todas las cajas de colisiones y solo interesan las personas
        for i in range (len (listaCajaColisiones)):
            # Se asigna un color distinto dependiendo si la persona es la objetivo o no
            if indicePersonaObjetivo == i:
                text = "Target - "
                color = (0, 255, 0)
            else:
                text = "No target - "
                color = (0, 0, 255)

            # Se dibuja un rectangulo simulando la bounding box (Verde si es la persona objetivo y roja si no)
            cv.rectangle (fotogramaConDetecciones, 
                          (listaCajaColisiones[i][0], listaCajaColisiones[i][1]),
                          (listaCajaColisiones[i][2], listaCajaColisiones[i][3]),
                          color,
                          2
                          )
            
            text = text + str (round (listaPrecisionDetecciones[i], 2))

            cv.putText (fotogramaConDetecciones, 
                        text, 
                        (listaCajaColisiones[i][0], listaCajaColisiones[i][1] - 10), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2)
            
        cv.imshow ("Fotograma original", fotogramaOriginal)
        cv.imshow ("Fotograma con detecciones", fotogramaConDetecciones)

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

            # Si se pulsa Enter entonces devuelve True (Si no se devuelve False)
            elif letraPulsada == 13:            
                return True
            
            # Es escalable (Usando la siguiente estructura)
            #elif letraPulsada == <codigo_letra>:
                # Actuacion  

            else:
                return False
                  
        return

    # ----------------------------------

    def generacion_de_resultados (self):
        
        # Calculo final
        self.tiempoEjecucion = time.time() - self.tiempoEjecucion

        # Creacion de diccionario para guardar la informacion en formato JSON
        resultadoFormatoJSON = {"Tiempo de ejecucion" : self.tiempoEjecucion,
                                "Numero de fotogramas procesados" : (self.contadorFotogramas - 1),
                                "Ruta destino del dataset clasificado" : self.rutaDestinoDatasetClasificado
                                }
        
        # Resultados por consola (Temporales)
        print ("\n\n----------- INICIO RESULTADOS -----------")
        print ("Tiempo de ejecucion:", self.tiempoEjecucion, " segundos")
        print ("Numero de fotogramas procesados:", (self.contadorFotogramas), "fotogramas")
        print ("Ruta destino del dataset clasificado:", self.rutaDestinoDatasetClasificado)
        print ("------------ FIN RESULTADOS ------------\n\n")

        # Resultados por archivo JSON (Permanentes)
        if os.path.exists (os.path.dirname (self.rutaDestinoResultadosJSON)):
            with open (self.rutaDestinoResultadosJSON, 'w') as flujoSalida:
                json.dump (resultadoFormatoJSON, flujoSalida)
        
        return
