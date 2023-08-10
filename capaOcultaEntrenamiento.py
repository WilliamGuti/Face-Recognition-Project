import cv2 
import numpy as np
import os
from time import time

#traer la carpta donde esta la informacion
dataRuta = 'C:/Users/w/OneDrive/Desktop/proyectoReconocimientoDeImagenes/reconocimientoFacial1CapturaDatos/Data'
listaData = os.listdir(dataRuta)

# añadir etiquetas para diferenciar
ids = []
rostrosData = []
id = 0

tiempoInicial = time()

for fila in listaData:
    rutaCompleta =  dataRuta + '/' + fila
    print('Iniciando Lectura...')
    for archivo in os.listdir(rutaCompleta):
        print('imagenes: ', fila + '/' + archivo )
        ids.append(id)
        rostrosData.append(cv2.imread(rutaCompleta + '/' + archivo, 0)) # se puede escalar a grises con el 0 para no hacer lo de bar binry
        
    id += 1

    tiempoFinalLectura = time()
    tiempoTotalLectura = tiempoFinalLectura - tiempoInicial
    print('Tiempo Total De Lectura: ' , tiempoTotalLectura) 
    

#entrenamiento de reconocimiento facial
entrenamientoModelo1 = cv2.face.EigenFaceRecognizer_create()
print("Iniciando Entrenamiento ... Espere")
entrenamientoModelo1.train(rostrosData, np.array(ids))
tiempoFinalLectura = time()
tiempoTotalLectura = tiempoFinalLectura - tiempoInicial
print("tiepo de entrenamiento total: " , tiempoTotalLectura)
entrenamientoModelo1.write('EntrenamientoEigenFaceRecognizer.xml')
print('Entrenamiento Completado')

        
