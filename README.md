# Generación de Conjuntos de Datos para Redes Neuronales de Seguimiento de Personas

## Descripción

Este proyecto está diseñado para automatizar la creación de conjuntos de datos a partir de datos de vídeo, con el objetivo de entrenar redes neuronales en tareas de seguimiento de personas. El sistema procesa vídeos, extrae características relevantes y genera conjuntos de datos etiquetados que pueden ser utilizados directamente para entrenar modelos de aprendizaje automático para la detección y seguimiento de personas.

## Propósito

El objetivo principal de este proyecto es simplificar el proceso de generación de conjuntos de datos de alta calidad y etiquetados, que son esenciales para entrenar modelos de visión por computadora en el seguimiento de personas. Al automatizar la creación de estos datos, se busca mejorar la precisión y eficiencia de los modelos de seguimiento.

## Características

- **Extracción de Datos**: Procesa vídeos para extraer información relevante.
- **Generación de Etiquetas**: Crea etiquetas para identificar y seguir personas en los vídeos.
- **Formato de Datos**: Genera conjuntos de datos en formatos estándar compatibles con herramientas de entrenamiento de redes neuronales.

## Uso

1. **Preparar Vídeos**: Asegúrate de tener los vídeos que deseas procesar en el directorio de entrada.
2. **Ejecutar el Script**: Corre el script de procesamiento para generar los conjuntos de datos.
3. **Revisar Resultados**: Los conjuntos de datos generados estarán disponibles en el directorio de salida especificado.

## Requisitos

- Python 3.x
- Bibliotecas: `opencv-python`, `numpy`, `pandas`, etc. (ver `requirements.txt` para la lista completa)

## Instalación

1. Clona el repositorio: `git clone <URL del repositorio>`
2. Navega al directorio del proyecto: `cd nombre-del-proyecto`
3. Instala las dependencias: `pip install -r requirements.txt`

## Contribuciones

Las contribuciones al proyecto son bienvenidas. Si deseas colaborar, por favor, sigue estos pasos:
1. Realiza un fork del repositorio.
2. Crea una rama para tu característica o corrección: `git checkout -b mi-nueva-caracteristica`
3. Realiza tus cambios y haz commit: `git commit -am 'Añadir nueva característica'`
4. Envía un pull request.
