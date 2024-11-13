"""
    Cuatro imagene: alto y bajo contraste, alta y baja iluminación
    Implemente las tecnicas basadas en el procesamiento de histogra:
        - Ecualización de histograma
            - Local
            - Global
        - Calculo de la media y varianza
            - local
            - global

    Pruebe las tecnicas en las imagenes y muestre los debera recomendar una técnica para cada imagen
"""
import cv2
import numpy as np
from o.imagenes_utilidades import mostrar_imagen, mostrar_histograma, mostrar_imagenes, ecualizacion_histograma, ecualizacion_histograma_local
from o.media import media

def media_varianza(imagen):
    pass

def media_varianza_local(imagen):
    pass

def trans_gamma(imagen, gamma):
    return np.power(imagen, gamma)

#IMAGEN='imagenes/pr.jpg'
IMAGEN  ='poca_iluminacion.jpg'

img = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE)


# mostrar con cv

# hist_original = mostrar_histograma(img, 'Histograma uno')
np.random.seed(0)

#img_ecualizada_global, sk = ecualizacion_histograma(img)
# mostrar_imagen(img_ecualizada, 'Imagen ecualizada')
# mostrar tambien los histogramas
# mostrar_histograma(img_ecualizada, 'Histograma ecualizada')


# mostrar_imagenes([img, img_ecualizada], ['Imagen original', 'Imagen ecualizada', '', ''], False)
# mostrar_imagenes([img, img_ecualizada], ['Imagen original', 'Imagen ecualizada', '', ''], True)

img_ecualizada = ecualizacion_histograma_local(img)
# mostrar_imagenes([img, img_ecualizada], ['Imagen original', 'Imagen ecualizada', '', ''], False)

# Ecualizar histograma con cv
# img_ecualizada_global_cv = cv2.equalizeHist(img)

mostrar_imagenes([img, img_ecualizada], ['Imagen original', 'Imagen ecualizada cv', 'gamma'], True)

#matriz_prueba = np.array([[0, 0, 1, 1, 2], [1, 2, 3, 0, 1], [3, 3, 2, 2, 0], [2, 3, 1, 0, 0], [1, 1, 3, 2, 2]])

#media = media(matriz_prueba)
#print(media)