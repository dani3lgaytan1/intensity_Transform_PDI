import cv2
import matplotlib.pyplot as plt
# Cargar las imágenes
imagen1 = cv2.imread('100.jpg')
imagen2 = cv2.imread('100_1.jpg')


# Load the image
#image = cv2.imread('image.jpg')

# Resize the image
#resized_image = cv2.resize(image, (width, height))

# Asegúrate de que las imágenes sean del mismo tamaño
imagen1 = cv2.resize(imagen1, (imagen2.shape[1], imagen2.shape[0]))



# Restar las imágenes
resultado = cv2.subtract(imagen1, imagen2)

suma = cv2.add(imagen1, imagen2)

multiplicacion = cv2.multiply(imagen1, imagen2) 

division = cv2.divide(imagen1, imagen2)



# Mostrar el resultado
plt.imshow( resultado)

plt.imshow( suma)   

plt.imshow( multiplicacion)

plt.imshow( division)   


plt.show()


