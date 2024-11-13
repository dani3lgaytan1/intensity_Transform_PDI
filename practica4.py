#Codigo practica 4

import matplotlib.pyplot as plt
import numpy as np
import cv2


def ecualizar_histograma(imagen):
    # Crear un histograma de la imagen
    histograma = cv2.calcHist([imagen.flatten()],[0],None,[256],[0,256])

    n = imagen.shape[0] * imagen.shape[1]  # Número total de píxeles
    # Crear un histograma acumulado
    histograma_acumulado = np.cumsum(histograma)
    # Normalizar el histograma acumulado
    histograma_acumulado = (histograma_acumulado * 255) / n
    # Redondear los valores del histograma acumulado
    histograma_acumulado = np.round(histograma_acumulado).astype(np.uint8)
    # Aplicar la transformación de intensidades
    imagen_ecualizada = histograma_acumulado[imagen]
    
    return imagen_ecualizada

def ecualizar_histograma2(imagen):
    #usando opencv
    imagen_ecualizada = cv2.equalizeHist(imagen)
    return imagen_ecualizada


def ecualizacion_local(imagen, tam_vecindad):
    # Calcular el padding en función del tamaño de la vecindad
    padding = tam_vecindad // 2
    
    # Añadir padding de 0 alrededor de la imagen para manejar bordes
    padding_imagen = cv2.copyMakeBorder(imagen, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)


    # Crear una imagen de salida
    imagen_salida = np.copy(imagen)

    # Recorrer cada píxel de la imagen original (sin el borde añadido por el padding)
    for i in range(padding, padding_imagen.shape[0] - padding):
        for j in range(padding, padding_imagen.shape[1] - padding):
            # Obtener la vecindad nxn alrededor del píxel actual
            vecindad = padding_imagen[i-padding:i+padding+1, j-padding:j+padding+1]
            
            # Realizar la ecualización del histograma en la vecindad
            vecindad_ecualizada = ecualizar_histograma(vecindad)
            
            # Actualizar el valor del píxel central con el valor ecualizado
            imagen_salida[i-padding, j-padding] = vecindad_ecualizada[padding, padding]

    return imagen_salida

def equalize_histogram_local(image, tile_size=7):
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(tile_size, tile_size))
    clahe_image = clahe.apply(image)
    return clahe_image

def media_global(imagen):
    # Calcular la media global de la imagen
    media_global = np.mean(imagen)
    
    # Crear una imagen de salida donde cada píxel tenga el valor de la media global
    imagen_salida = np.full(imagen.shape, media_global, dtype=imagen.dtype)

    return imagen_salida

def varianza_global(imagen):
    # Calcular la varianza global de la imagen
    varianza_global = np.var(imagen)
    
    # Crear una imagen de salida donde cada píxel tenga el valor de la varianza global
    imagen_salida = np.full(imagen.shape, varianza_global, dtype=imagen.dtype)

    return imagen_salida

def procesamiento_adaptativo_global(imagen, umbral_metodo, umbral_varianza=500):

    # Aplicar ecualización de histograma global si la varianza global es baja
    if umbral_metodo < umbral_varianza:
        print("Aplicando ecualización de histograma global.")
        return ecualizar_histograma2(imagen)

    else:
        print("La varianza es alta. No se necesita ecualización global.")
        return imagen  # Retorna la imagen sin cambios si no requiere ecualización



def mejoramiento_local(imagen, tam_vecindad,a, k0, k1, k2, k3):
    # Calcular la media y varianza global
    media_global = np.mean(imagen)
    #calcular la desviacion estandar
    desviacion_global = np.std(imagen)
    

    padding = tam_vecindad // 2
    
    # Añadir padding de 0 alrededor de la imagen para manejar bordes
    padding_imagen = cv2.copyMakeBorder(imagen, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
   
    # Crear una imagen de salida
    imagen_salida = np.copy(imagen)

    # Recorrer cada píxel de la imagen original (sin el borde añadido por el padding)
    for i in range(padding, padding_imagen.shape[0] - padding):
        for j in range(padding, padding_imagen.shape[1] - padding):
            # Obtener la vecindad nxn alrededor del píxel actual
            vecindad = padding_imagen[i-padding:i+padding+1, j-padding:j+padding+1]
            
            # Calcular la media y varianza de la vecindad
            vecindad_media = np.mean(vecindad)
            vecindad_desviacion= np.std(vecindad)

            # condicion para aplicar el mejoramiento estadistico : 
            if(k1*desviacion_global >= vecindad_desviacion >= k0*desviacion_global and k3*media_global >= vecindad_media >= k2*media_global):
                # Aplicar la transformación de intensidades
                nuevo_valor = a * imagen[i-padding, j-padding]
                imagen_salida[i-padding, j-padding] = np.clip(nuevo_valor, 0, 255)
            else:
                imagen_salida[i-padding, j-padding] = imagen[i-padding, j-padding]
           

    return imagen_salida.astype(np.uint8)

def mejorar_imagen_con_media_varianza(imagen, varianza_deseada):
    # Calcular la media y la varianza global de la imagen
    media_global = np.mean(imagen)
    varianza_global = np.var(imagen)

    print("Media global: ", media_global)
    print("Varianza global: ", varianza_global)


    # Calcular el valor de alpha para ajustar el contraste
    alpha = np.sqrt(varianza_deseada / varianza_global)
    
    # Aplicar la transformación
    imagen_mejorada = alpha * (imagen - media_global) + media_global

    # Asegurarse de que los valores estén en el rango [0, 255]
    imagen_mejorada = np.clip(imagen_mejorada, 0, 255).astype(np.uint8)

    return imagen_mejorada

def calcular_histograma(imagen):
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    return histograma

def display_resultado(imagen_original, imagen_resultado,title="Resultado",title2="Resultado", window_title="Resultado"):
    # Calcular histogramas
    histograma_original = calcular_histograma(imagen_original)
    histograma_resultado = calcular_histograma(imagen_resultado)

    # Crear figura con 2x2 subgráficos
    plt.figure(figsize=(12, 10))

 # Título principal de la figura
    plt.suptitle(window_title, fontsize=16)

    # Imagen Original
    plt.subplot(2, 2, 1)
    plt.imshow(imagen_original, cmap='gray')
    plt.title("Imagen Original")
    plt.axis('off')  # Ocultar ejes

    # Histograma de la Imagen Original
    plt.subplot(2, 2, 2)
    plt.plot(histograma_original, color='blue')
    plt.title("Histograma Original")
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.xlim([0, 256])

    # Imagen Mejorada
    plt.subplot(2, 2, 3)
    plt.imshow(imagen_resultado, cmap='gray')
    plt.title(title)
    plt.axis('off')  # Ocultar ejes

    # Histograma de la Imagen Mejorada
    plt.subplot(2, 2, 4)
    plt.plot(histograma_resultado, color='green')
    plt.title(title2)
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.xlim([0, 256])

    # Mostrar la figura completa
    plt.tight_layout()
    plt.show()

def mostrar_comparacion(imagenes_originales, titulos_filas):


    plt.figure(figsize=(12, 12))
    
    for idx, imagen_original in enumerate(imagenes_originales):
        # Imagen original
        plt.subplot(len(imagenes_originales), 4, idx * 4 + 1)
        plt.imshow(imagen_original, cmap='gray')
        plt.title(titulos_filas[idx])
        plt.axis('off')
        
        # Imagen ecualizada
        imagen_ecualizada = ecualizar_histograma2(imagen_original)
        plt.subplot(len(imagenes_originales), 4, idx * 4 + 2)
        plt.imshow(imagen_ecualizada, cmap='gray')
        plt.title("Ecualizada")
        plt.axis('off')
        
        # Histograma original
        hist_original = calcular_histograma(imagen_original)
        plt.subplot(len(imagenes_originales), 4, idx * 4 + 3)
        plt.plot(hist_original, color='blue')
        plt.title(titulos_filas[idx])
        plt.xlim([0, 256])
        
        # Histograma ecualizado
        hist_ecualizado = calcular_histograma(imagen_ecualizada)
        plt.subplot(len(imagenes_originales), 4, idx * 4 + 4)
        plt.plot(hist_ecualizado, color='green')
        plt.title("Histograma ecualizado")
        plt.xlim([0, 256])
        
        # Etiquetas de fila
        plt.subplot(len(imagenes_originales), 4, idx * 4 + 1)
        plt.ylabel(titulos_filas[idx], rotation=0, labelpad=40, verticalalignment='center')
    
    plt.tight_layout()
    plt.show()


# Código principal para manipular imágenes
if __name__ == "__main__":
    #Menu de opciones

    #elegir imagen de entrada 

    # Mostrar el menú de transformación
    print("Elige un tipo de imagen:")
    print("1. Alto Contraste")
    print("2. Bajo Contraste")
    print("3. Poca Iluminación")
    print("4. Alto Iluminacion")
    print("5. Salir")

    # Leer la opción del usuario

    opcion = int(input("Opción: "))

    # Leer la imagen de entrada
    if opcion == 1:
        image = cv2.imread('altoC2.jpg', cv2.IMREAD_GRAYSCALE)
        #image = cv2.imread('alto_contraste2.jpg', cv2.IMREAD_GRAYSCALE)
    elif opcion == 2:
        image = cv2.imread('bajoC1.jpg', cv2.IMREAD_GRAYSCALE)
    elif opcion == 3:
        image = cv2.imread('poca_iluminacion.jpg', cv2.IMREAD_GRAYSCALE)
    elif opcion == 4:
        image = cv2.imread('equa.png', cv2.IMREAD_GRAYSCALE)
    elif opcion == 5:
        #salir del programa
        image = None
    else:
        print("Opción no válida.")
        image = None

    # Verificar si se seleccionó una imagen válida

    if image is not None:
        # Mostrar la imagen de entrada
             # Mostrar el menú de transformación
        print("Elige un metodo:")
        print("1. Ecualización de histograma global")
        print("2. Ecualización de histograma local")
        print("3. Media global y varianza global")
        print("4. Mejoramiento estadístico local")
        print("5. Todas ecualizacion de histograma global")
        print("6. Salir")

        # Leer la opción del usuario
        opcion = int(input("Opción: "))
        # Verificar la opción seleccionada


      
        if opcion == 1:
            hist_ecualizado=ecualizar_histograma2(image)
            display_resultado(image, hist_ecualizado,"Imagen Ecualizada","Histograma Ecualizado Global","Ecualizacion Global")
            
        elif opcion == 2:
            #preguntar por el tamaño de la vecindad 
            vecindad = int(input("tamaño de la vecindad: "))
            if vecindad > 0:
                hist_local = ecualizacion_local(image, vecindad)
                display_resultado(image, hist_local,"Imagen Ecualizada","Histograma Ecualizado Local","Ecualizacion Local")
            else:
                print("El tamaño de la vecindad debe ser mayor a 0.")

        elif opcion == 3:
            varianza_deseada = 1500  # Puedes experimentar con diferentes valores

# Mejorar la imagen usando la media y varianza global
            imagen_mejorada = mejorar_imagen_con_media_varianza(image, varianza_deseada)
          
        
            display_resultado(image,imagen_mejorada,"Imagen Media y Varianza Global","Histograma Media varianza Global","Media y Varianza Global")

        elif opcion == 4:
            # Leer los parámetros para el mejoramiento estadístico local
            a = float(input("Valor de a: "))
            k0 = float(input("Valor de k0: "))
            k1 = float(input("Valor de k1: "))
            k2 = float(input("Valor de k2: "))
            k3 = float(input("Valor de k3: "))
            vecindad = int(input("Tamaño de la vecindad: "))
            if vecindad > 0:
                #hist_mejoramiento=  mejoramiento_local(image, tam_vecindad=7,a=3, k0=0, k1=0.8, k2=0, k3=0.7)
                hist_mejoramiento = mejoramiento_local(image, vecindad, a, k0, k1, k2, k3)
                display_resultado(image, hist_mejoramiento,"Imagen mejoramiento E.Local","Histograma E.Local","Mejoramiento E.Local")
            else:
                print("El tamaño de la vecindad debe ser mayor a 0.")

        elif opcion == 5:

            image = cv2.imread('poca_iluminacion.jpg', cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread('altoC2.jpg', cv2.IMREAD_GRAYSCALE)
            image3 = cv2.imread('bajoC1.jpg', cv2.IMREAD_GRAYSCALE)
            image4 = cv2.imread('ilumi1.jpg', cv2.IMREAD_GRAYSCALE)
            imagenes_originales = [image, image2, image3, image4]
            titulos_filas = ["Poca Iluminación", "Alto Contraste", "Bajo Contraste", "Alto Iluminación"]
            mostrar_comparacion(imagenes_originales, titulos_filas)

        else:
            print("Opción no válida.")
    
    else:
        print("Imagen no válida.")
