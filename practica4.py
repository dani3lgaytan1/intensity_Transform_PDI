#Codigo practica 4

import matplotlib.pyplot as plt
import numpy as np
import cv2




def ecualizar_histograma(imagen):
    # Crear un histograma de la imagen
    histograma = cv2.calcHist([imagen],[0],None,[256],[0,256])
    # Crear un histograma acumulado
    histograma_acumulado = np.cumsum(histograma)
    # Normalizar el histograma acumulado
    histograma_acumulado = (histograma_acumulado * 255) / histograma_acumulado[-1]
    # Redondear los valores del histograma acumulado
    histograma_acumulado = histograma_acumulado.round().astype(np.uint8)
    # Aplicar la transformación de intensidades
    imagen_ecualizada = histograma_acumulado[imagen]
    return imagen_ecualizada

def equalize_histogram(image):
    equalize = cv2.equalizeHist(image)
    return equalize


def media_varianza_local(imagen, ventana=(3, 3)):
    # Calcular la media y varianza global
    media_global = np.mean(imagen)
    varianza_global = np.var(imagen)

    # Mostrar la media y varianza global
    print(f"Media Global: {media_global}")
    print(f"Varianza Global: {varianza_global}")

    # Definir el tamaño de la ventana para calcular media y varianza local
    ventana_size = 15  # Tamaño de la ventana, ajustable

    # Calcular la media y varianza local
    media_local = cv2.blur(imagen.astype(np.float32), (ventana_size, ventana_size))
    varianza_local = cv2.blur((imagen.astype(np.float32) - media_local)**2, (ventana_size, ventana_size))

    # Mostrar el resultado
    cv2.imshow("Media Local", media_local / 255)  # Escalar para visualizar mejor
    cv2.imshow("Varianza Local", varianza_local / np.max(varianza_local))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def media_varianza_global(imagen):
    # Calcular el histograma de la imagen
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    N = imagen.size  # Número total de píxeles

    # Calcular la media global
    intensidades = np.arange(256)  # Intensidades de 0 a 255
    media_global = np.sum(intensidades * histograma) / N

    # Calcular la varianza global
    varianza_global = np.sum(((intensidades - media_global) ** 2) * histograma) / N

    # Mostrar los resultados
    print(f"Media Global (a partir del histograma): {media_global}")
    print(f"Varianza Global (a partir del histograma): {varianza_global}")
    plt.plot(intensidades, histograma, color='black')
    plt.axvline(media_global, color='red', linestyle='--', label=f'Media Global: {media_global:.2f}')
    plt.axvline(media_global + np.sqrt(varianza_global), color='blue', linestyle='--', label=f'+1 Desv: {np.sqrt(varianza_global):.2f}')
    plt.axvline(media_global - np.sqrt(varianza_global), color='blue', linestyle='--', label=f'-1 Desv: {np.sqrt(varianza_global):.2f}')
    plt.title('Histograma con Media y Varianza')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.show()



def equalize_histogram_local(image, tile_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=35.0, tileGridSize=tile_size)
    clahe_image = clahe.apply(image)
    return clahe_image


def display_histograma(imagen, title="Histograma"):
    media_intensidades = np.mean(imagen)
    histograma = cv2.calcHist([imagen],[0],None,[256],[0,256])
    plt.figure()
    plt.title(title)
    plt.xlabel('Intensidadades')
    plt.ylabel('pixels')
    plt.plot(histograma)
    plt.axvline(x=media_intensidades, color='r', label='Media de intensidades')
    plt.legend()
    plt.xlim([0,256])
    plt.show()

# Función para mostrar la imagen
def display_image(image, title="Resultado",imagenOriginal=None):
    cv2.imshow(title,cv2.hconcat([imagenOriginal, image]))
    #cv2.imshow(title,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# Código principal para manipular imágenes
#imagen = cv2.imread('bajo_contraste2.png', cv2.IMREAD_GRAYSCALE)
imagen = cv2.imread('equa.png', cv2.IMREAD_GRAYSCALE)
#imagen = cv2.imread('alto_contraste.jpg', cv2.IMREAD_GRAYSCALE)
imagen = np.array(imagen)
display_histograma(imagen, "Histograma de la imagen original")
# Mostrar la imagen de entrada

#media_varianza_global(imagen)

media_varianza_local(imagen)
#equali  = equalize_histogram_local(imagen, tile_size=(8, 8))
#display_image(equali, "Histograma de la imagen ecualizada",imagen)
#display_histograma(equali, "Histograma de la imagen ecualizada")

