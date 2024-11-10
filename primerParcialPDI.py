#CODIGOS DE ESTUDIO DE OPENcv Y PDI MI LOKO

import cv2
import numpy as np




#cargar imagenes :





#Crear un circulo y cuadrado 

cuadrado = np.zeros((400,600),dtype="uint8")
cuadrado [100:300,200:400] = 255

circulo = np.zeros((400,600),dtype="uint8")
circulo = cv2.circle(circulo,(300,200),125,255,-1)


notImagen  = cv2.bitwise_not(circulo)

AND  = cv2.bitwise_and(cuadrado,notImagen)


OR  = cv2.bitwise_or (cuadrado,circulo)

XOR  = cv2.bitwise_xor (cuadrado,circulo)

cv2.imshow("not",notImagen)

cv2.imshow("AND",AND)
cv2.imshow("OR",OR)



cv2.imshow("Cuadrado",cuadrado)
cv2.imshow("Circulo",circulo)
cv2.waitKey(0)
cv2.destroyAllWindows()




#Operaciones de conjuntos de imagenes 