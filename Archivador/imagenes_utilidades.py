import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_imagen(imagen, titulo):
    plt.imshow(imagen, cmap='gray')
    plt.title(titulo)
    plt.show()

def mostrar_imagenes(imagenes, titulos, histograma=False):
    # Número de imágenes
    n = len(imagenes)

    # Definir el tamaño de la cuadrícula (calcular filas y columnas)
    columnas = 2  # Número de columnas deseado (puedes ajustarlo)
    filas = (n + columnas - 1) // columnas
    # Calcula el número de filas necesario

    # Crear la figura
    hist = None

    if histograma:
        hist = [retornar_histograma(imagen) for imagen in imagenes]
        filas = n

    plt.figure(figsize=(5 * columnas, 5 * filas))  # Ajusta el tamaño de la figura según columnas y filas

    for i in range(n):

        if histograma:
            # la posicion es a la izquierda de la imagen
            posImagen = 2 * i + 1
            posHist = 2 * i + 2
        else:
            posImagen = i + 1
            posHist = 0

        plt.subplot(filas, columnas, posImagen)  # Posición del subplot
        plt.imshow(imagenes[i], cmap='gray', vmin=0, vmax=255) # IMPORTANTE SI NO SE PONE VMIN Y VMAX LA PERCEPCION DE LA IMAGEN CAMBIA (SE VE MAS CLARA)
        plt.title(titulos[i])
        plt.axis('off')
        if histograma:
            # la posicion es a la derecha de la imagen
            plt.subplot(filas, columnas, posHist)
            plt.plot(hist[i], color='gray')
            plt.xlabel('intensidad de iluminacion')
            plt.ylabel('cantidad')
            plt.title('Histograma')

    # Ajustar espacio entre imágenes
    plt.tight_layout()
    plt.show()



def mostrar_histograma(imagen, titulo):
    hist = retornar_histograma(imagen)
    plt.plot(hist, color='gray')
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.title(titulo)
    plt.show()
    return hist


def retornar_histograma(imagen):
    # Calcular el histograma usando np.bincount, más eficiente para imágenes de 8 bits
    histograma = np.bincount(imagen.flatten(), minlength=256)
    return histograma


def mostrar_histograma_plt(imagen, titulo):
    plt.hist(imagen.ravel(), 256, [0, 256])
    plt.title(titulo)
    plt.show()


def ecualizacion_histograma(imagen):
    hist = retornar_histograma(imagen)
    L = 256
    n = imagen.shape[0] * imagen.shape[1]
    sn = np.cumsum(hist)
    sk = (L - 1) * sn / n
    sk = np.round(sk).astype(np.uint8)
    img_ecualizada = sk[imagen]

    return img_ecualizada, sk


def ecualizacion_histograma_local(imagen):
    print(f'Imagen shape x: {imagen.shape[0]} y: {imagen.shape[1]}')
    vecindad = 150  # es decir 8-vecindad

    padding = (vecindad) // 2
    # no colocar padding a la izquierda ni arriba, solo a la derecha y abajo
    imagen = cv2.copyMakeBorder(imagen, 0, padding, 0, padding, cv2.BORDER_CONSTANT, value=0)
    mostrar_imagen(imagen, 'Imagen con padding')
    print(f'Imagen shape x: {imagen.shape[0]} y: {imagen.shape[1]}')
    # generar ventanas sin superposicion
    ventana = np.array([imagen[i - padding:i + padding, j - padding:j + padding] for i in range(padding, imagen.shape[0] - padding, vecindad) for j in range(padding, imagen.shape[1] - padding, vecindad)])

    ecualizad = []
    for i in ventana:
        ecualizad.append(ecualizacion_histograma(i)[0])

    imagen_ecualizada = np.zeros_like(imagen)
    index = 0

    for i in range(padding, imagen.shape[0] - padding, vecindad):
        for j in range(padding, imagen.shape[1] - padding, vecindad):
            ventana = ecualizad[index]
            imagen_ecualizada[i - padding:i + padding, j - padding:j + padding] = ventana
            index += 1

    # eliminar padding a la derecha y abajo
    imagen_ecualizada = imagen_ecualizada[0:imagen.shape[0] - padding, 0:imagen.shape[1] - padding]

    return imagen_ecualizada
