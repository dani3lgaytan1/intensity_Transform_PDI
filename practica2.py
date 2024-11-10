import numpy as np
import cv2


#transformaciones geometricas : 


def traslacion(imagen,tx,ty):
    rows,cols = imagen.shape
    M = np.float32([[1,0,tx],
                    [0,1,ty]])
    dst = cv2.warpAffine(imagen,M,(cols,rows))
    return dst



def rotation(imagen,angulo):

    rows,cols = imagen.shape
   # Obtener la matriz de rotación
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angulo, 1)
    dst = cv2.warpAffine(imagen,M,(cols,rows))
    return dst


def escala(imagen,sx,sy):
    imagen1 = cv2.resize(imagen, (int(imagen.shape[1]*sx), int(imagen.shape[0]*sy)), interpolation = cv2.INTER_AREA)
    
    rows,cols = imagen.shape
    M = np.float32([[sx,0,0],
                    [0,sy,0]])
    dst = cv2.warpAffine(imagen,M,(cols,rows))
    return dst

def escala2(imagen):


    img_enlarged = cv2.resize(imagen, None,
    fx=0.5, fy=0.5,
    interpolation=cv2.INTER_CUBIC)
    cv2.imshow('grande', img_enlarged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#crear 3 imagenes binarias 

#crear un circulo 

def shear_x(imagen):
    rows, cols = imagen.shape
    M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    sheared_img = cv2.warpPerspective(imagen, M, (int(cols*1.5), int(rows*1.5)))
    return sheared_img

def shear_y(imagen):
    rows,cols = imagen.shape
    M = np.float32([[1,0,0],[0.5,1,0],[0,0,1]])
    sheared_img = cv2.warpPerspective(imagen,M,(int(cols*1.5),int(rows*1.5)))
    return sheared_img


cuadrado = np.zeros((400,600),dtype="uint8")
cuadrado [100:300,200:400] = 255

circulo = np.zeros((400,600),dtype="uint8")
circulo = cv2.circle(circulo,(300,200),125,255,-1)

triangulo = np.zeros((400, 600), dtype="uint8")
pts = np.array([[300, 100], [200, 300], [400, 300]], np.int32)
pts = pts.reshape((-1, 1, 2))
triangulo = cv2.fillPoly(triangulo, [pts], 255)



notTriangulo = cv2.bitwise_not(triangulo)
notCuadrado = cv2.bitwise_not(cuadrado)

AND = cv2.bitwise_and(circulo,cuadrado)

OR = cv2.bitwise_or(triangulo,cuadrado)

XOR = cv2.bitwise_xor(triangulo,cuadrado)

AND2 = cv2.bitwise_and(circulo,notCuadrado )


#cv2.imshow("AND",AND)

#traslacionAND = traslacion(cuadrado,100,70)

#rotacion = rotation(cuadrado,30)


escalae = escala(cuadrado,0.5,0.5)

#cv2.imshow("rotacion",rotacion)
cv2.imshow("escala",escalae)




escala2(cuadrado)


#cv2.imshow("TRIANGULO",triangulo)   
#cv2.imshow("CIRCULO",circulo)
cv2.imshow("CUADRADO",cuadrado)
cv2.waitKey(0)
cv2.destroyAllWindows()

