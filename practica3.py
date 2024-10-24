
import cv2
import matplotlib.pyplot as plt
import  numpy as np


def negative_transform(img):
     
    img = img.astype(np.float32) # Convertir la imagen a flotante
    negativo = 255 - img # Aplicar la transformación negativa L-1-r  255 - img 
    negativo = np.uint8(negativo) # Convertir la imagen a entero
    
    return negativo
     

def log_transform(img, c=1):

    img = img.astype(np.float32) # Convertir la imagen a flotante
    loga = c * np.log(1 + img) # Aplicar la transformación logarítmica 
    maximo = np.amax(loga) # Encontrar el valor máximo de la imagen 
    loga = np.uint8(loga/maximo  * 255) # Normalizar la imagen 
    #los valores no se dan en enteros, por lo que se hace una conversión a enteros

    return loga


def gamma_transform(img, c=1, gamma=1):
    
        img = img.astype(np.float32) # Convertir la imagen a flotante
        gammaI = c * np.power(img,gamma)  # Aplicar la transformación gamma 
        maximo = np.amax(gammaI) # Encontrar el valor máximo de la imagen 
        gammaI = np.uint8(gammaI/maximo  * 255) # Normalizar la imagen 
        #los valores no se dan en enteros, por lo que se hace una conversión a enteros
    
        return gammaI


def estiramiento_contraste(img):
    
        img = img.astype(np.float32) # Convertir la imagen a flotante
        minimo = np.min(img) # Encontrar el valor mínimo de la imagen 
        maximo = np.max(img) # Encontrar el valor máximo de la imagen 
        contraste_estirado = 255 * ((img - minimo) / (maximo - minimo)) # Aplicar la transformación de estiramiento de contraste
        contraste_estirado = np.uint8(contraste_estirado) # Convertir la imagen a entero
        
        return contraste_estirado



def intensity_slicing(image, r_min, r_max, highlight_value=255, preserve_outside=True):

    # Crear una copia de la imagen para trabajar
    output_image = np.zeros_like(image)
    
    # Aplicar el resaltado en el rango de interés
    mask = (image >= r_min) & (image <= r_max)
    
    if preserve_outside:#que se vea la vena
        # Preservar los valores fuera del rango, pero resaltar los valores dentro del rango
        output_image = np.where(mask, highlight_value, image)
    else:#que no se vea la vena 
        # Resaltar los valores dentro del rango, y los valores fuera del rango se establecen a 0
        output_image[mask] = highlight_value
    
    return output_image


# Función para aplicar un slicing de planos de bits
def bit_plane_slicing(image, num_bits=8):
    bit_planes = [np.zeros(image.shape, dtype=np.uint8) for _ in range(num_bits)]
    
    for i in range(num_bits):
        bit_value = 2 ** i
        bit_planes[i] = (image & bit_value) >> i
        bit_planes[i] = bit_planes[i] * 255  # Escalar los valores para visualización
    return bit_planes



# Función para mostrar la imagen
def display_image(image, title="Imagen",imagenOriginal=None):
    cv2.imshow('RESULTADO',cv2.hconcat([imagenOriginal, image]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Código principal para manipular imágenes
if __name__ == "__main__":
    #Menu de opciones

    #elegir imagen de entrada 

    # Mostrar el menú de transformación
    print("Elige un tipo de imagen:")
    print("1. Alto Contraste")
    print("2. Bajo Contraste")
    print("3. Poca Iluminación")

    # Leer la opción del usuario

    opcion = int(input("Opción: "))

    # Leer la imagen de entrada
    if opcion == 1:
        image = cv2.imread('alto_contraste2.jpg', cv2.IMREAD_GRAYSCALE)
    elif opcion == 2:
        image = cv2.imread('bajo_contraste2.png', cv2.IMREAD_GRAYSCALE)
    elif opcion == 3:
        image = cv2.imread('poca_iluminacion.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        print("Opción no válida.")
        image = None

    # Verificar si se seleccionó una imagen válida

    if image is not None:
        # Mostrar la imagen de entrada
             # Mostrar el menú de transformación
        print("Elige un tipo transformación:")
        print("1. Negativo")
        print("2. Logarítmica")
        print("3. Gamma")
        print("4. Estiramiento de contraste")
        print("5. Slicing de intensidad")
        print("6. Slicing de planos de bits")
        print("7. Salir")

        # Leer la opción del usuario
        opcion = int(input("Opción: "))
        # Verificar la opción seleccionada

        image = np.array(image)
      
        if opcion == 1:
            # Transformación negativa
            negativo = negative_transform(image)
            display_image(negativo, "Negativo",image)
        elif opcion == 2:
            # Transformación logarítmica
            loga = log_transform(image)
            display_image(loga, "Logarítmica",image)
        elif opcion == 3:
            # Transformación gamma
            gammaI = gamma_transform(image,1,2)
            display_image(gammaI, "Gamma",image)
        elif opcion == 4:
            # Estiramiento de contraste
            contraste_estirado = estiramiento_contraste(image)
            display_image(contraste_estirado, "Estiramiento de contraste",image)
        elif opcion == 5:
            # Slicing de intensidad
            r_min = int(input("Valor mínimo del rango de intensidad: "))
            r_max = int(input("Valor máximo del rango de intensidad: "))
            preserve_outside = bool(int(input("Preservar valores fuera del rango (1) o no (0): ")))
            intensity_sliced = intensity_slicing(image, r_min, r_max,255, preserve_outside)
            display_image(intensity_sliced, "Slicing de intensidad",image)
        elif opcion == 6:
            # Slicing de planos de bits
            #mostrar imagen original
            #cv2.imshow('Imagen Original',image)
           
            bit_planes_sliced = bit_plane_slicing(image)

            # Definir el tamaño de la figura (en pulgadas)
            plt.figure(figsize=(10, 5))  # Ajusta el tamaño según lo que necesites

            # Mostrar la imagen original
            plt.subplot(1, 4, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Imagen Original')
            plt.axis('off')

            # Mostrar los planos de bits seleccionados (5, 6, 7)
            for i in range(3):
                plt.subplot(1, 4, i + 2)  # Las posiciones van de 2 a 4 en este caso
                plt.imshow(bit_planes_sliced[i +5 ], cmap='gray')
                plt.title(f'Bit Plane {5+i }')
                plt.axis('off')

                # Mostrar el resultado
            plt.tight_layout()
            plt.show()
        else:
            print("Opción no válida.")
    
    else:
        print("Imagen no válida.")






