import cv2
import matplotlib.pyplot as plt
# Cargar las imágenes
imagen1 = cv2.imread('100.jpg')
imagen2 = cv2.imread('100_1.jpg')

# Asegúrate de que las imágenes sean del mismo tamaño
imagen1 = cv2.resize(imagen1, (imagen2.shape[1], imagen2.shape[0]))

# Restar las imágenes
resultado = cv2.subtract(imagen1, imagen2)

# Mostrar el resultado
plt.imshow( resultado)

plt.show()


