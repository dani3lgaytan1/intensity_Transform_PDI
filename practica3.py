
import cv2
import matplotlib.pyplot as plt
import  numpy as np

original = cv2.imread('resaltar.png', cv2.IMREAD_GRAYSCALE)


aux_Original = np.array(original)


def imadjust(F,range_in=(0,1),range_out=(0,1),gamma=1):
    G = (((F - range_in[0]) / (range_in[1] - range_in[0])) ** gamma) * (range_out[1] - range_out[0]) + range_out[0]
    return G


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
        minimo = np.amin(img) # Encontrar el valor mínimo de la imagen 
        maximo = np.amax(img) # Encontrar el valor máximo de la imagen 
        contraste_estirado = 255 * (img - minimo) / (maximo - minimo) # Aplicar la transformación de estiramiento de contraste
        contraste_estirado = np.uint8(contraste_estirado) # Convertir la imagen a entero
        
        return contraste_estirado



def intensity_slicing(image, r_min, r_max, highlight_value=255, preserve_outside=True):
    """
    Realiza una rebanada de nivel de intensidad sobre la imagen.
    
    Args:
        image (numpy array): La imagen de entrada (en escala de grises).
        r_min (int): Límite inferior del rango de intensidad.
        r_max (int): Límite superior del rango de intensidad.
        highlight_value (int): Valor de resaltado para los píxeles dentro del rango.
        preserve_outside (bool): Si es True, preserva los valores fuera del rango; si es False, los establece a 0.
    
    Returns:
        numpy array: La imagen con rebanada de nivel de intensidad aplicada.
    """
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

#negativo = imadjust(aux_Original, (0, 1), (1, 0)) # Negativo de la imagen

#print(negativo)



#loga = log_transform(aux_Original, c=1)

#gamma = gamma_transform(aux_Original, c=1, gamma=0.3)


#print(aux_Original)
#resaltar = intensity_slicing(aux_Original,150,230, highlight_value=255, preserve_outside=False)

 # Aplicar el slicing de los bit planes
bit_planes_sliced = bit_plane_slicing(aux_Original)

print(bit_planes_sliced)

    # Mostrar cada bit plane
for i, plane in enumerate(bit_planes_sliced):
        cv2.imshow(f'Bit Plane {i}', plane)

#negativo  = negative_transform(aux_Original)    

#contraste_estirado = estiramiento_contraste(aux_Original)

#cv2.imshow('original', original)

#cv2.imshow('contraste_estirado', contraste_estirado)

#cv2.imshow('resaltar', resaltar )    
#cv2.imshow('modificada',cv2.hconcat([original, contraste_estirado]))
#brillo_up = imadjust(original, gamma=0.3) # Aumentar brillo
#brillo_down = imadjust(original, gamma=1.7) # Disminuir


# Mostrar imágenes
#_, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(10, 10))

#ax0.imshow(original, cmap="gray")
#ax0.set_title("original")
#ax0.set_axis_off()

#ax1.imshow(negativo, cmap="gray")
#ax1.set_title("negativo")
#ax1.set_axis_off()

#ax2.imshow(brillo_up, cmap="gray")
#ax2.set_title("brillo_up")
#ax2.set_axis_off()

#ax3.imshow(brillo_down, cmap="gray")
#ax3.set_title("brillo_down")
#ax3.set_axis_off()




# Código principal para manipular imágenes
if __name__ == "__main__":
    # Cargar la imagen
    image_path = 'tu_imagen_bn.jpg'  # Coloca el nombre de tu imagen en blanco y negro
    image = load_image_grayscale(image_path)
    



    if image is not None:
        # Mostrar la imagen original
        display_image(image, "Imagen Original")
        
        # Invertir los colores (negativo)
        inverted_image = invert_image(image)
        display_image(inverted_image, "Imagen Invertida (Negativo)")
        
        # Ajustar el contraste
        contrast_image = adjust_contrast(image)
        display_image(contrast_image, "Contraste Ajustado")
        
        # Aplicar bit-plane slicing
        bit_planes = bit_plane_slicing(image)
        for i, plane in enumerate(bit_planes):
            display_image(plane, f"Plano de Bits {i}")

        # Guardar una de las imágenes procesadas como ejemplo
        save_image(inverted_image, "imagen_invertida.jpg")
    
     cv2.waitKey(0)
    cv2.destroyAllWindows()



