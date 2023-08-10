import cv2 as cv
import numpy as np
import os #sirve para crear archivos con codigo python
import imutils

modelo = 'FotosWilliam' #nombre de como se va a llamar la carpeta
ruta1 = 'C:/Users/w/OneDrive/Desktop/proyectoReconocimientoDeImagenes/reconocimientoFacial1CapturaDatos'
rutaCompleta = ruta1 + '/' + modelo

#si no existe la ruta, que la cree
if not os.path.exists(rutaCompleta):
    os.makedirs(rutaCompleta)



# ruidos ya prestablecidos cara frontal
ruidos = cv.CascadeClassifier('C:/Users/w/OneDrive/Desktop/proyectoReconocimientoDeImagenes/entrenamientosOpencvRuidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

#activar camara
camara = cv.VideoCapture(0)

# id de cada imagen
id = 0

while True:
    respuesta, captura = camara.read()
    if respuesta == False:
        print("No se puede capturar la camara")
        break
    # bajarle la resolucion en caso de que tenga muy alta calidad
    captura = imutils.resize(captura, width = 640)
    #hacemos escala de grises
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    # deteccion del rostro con los ruidos
    caras = ruidos.detectMultiScale(grises, 1.3, 3)

    # id de la captura de cada imagen
    idCaptura = captura.copy()

    #hacemos el rectangulo 
    for (x,y,e1,e2) in caras:
        cv.rectangle(captura,(x,y),(x+e1,y+e2),(0,0,255),2)
        #sacar fragmentos de nuestro rostro y almarcenarlos en la carpeta
        rostroCapturado = idCaptura[y:y+e2 , x:x+e1]
        # modificar el tamaño de las fotos
        rostroCapturado = cv.resize(rostroCapturado,(160,160), interpolation=cv.INTER_CUBIC)
        # modificar el nombre de cada fotos
        cv.imwrite(rutaCompleta + '/imagen_{}.jpg'.format(id), rostroCapturado)
        # aumentar el id
        id += 1


    cv.imshow("Resultado Rostro", captura)
    
    if id == 351:
        break

camara.release()
cv.destroyAllWindows()
