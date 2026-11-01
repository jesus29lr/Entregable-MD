#   entrenamiento.py

import cv2
import os
import numpy as np

#   UBICACION DE LOS DATOS
ruta_datos = "data"

#   NOMBRE DE LAS PERSONAS
lista_personas = sorted(os.listdir(ruta_datos))

print("LISTA DE PERSONAS: ", lista_personas)

etiquetas = []
datos_rostros = []
etiqueta = 0

print("LEYENDO LAS IMAGENES")

for nombre_persona in lista_personas :
    ruta_persona = os.path.join(ruta_datos, nombre_persona)

    for nombre_archivo in os.listdir(ruta_persona):
        ruta_imagen = os.path.join(ruta_persona, nombre_archivo)

        print("ROSTRO: ", ruta_imagen)

        imagen_rostro = cv2.imread(ruta_imagen, 0)

        etiquetas.append(etiqueta)
        datos_rostros.append(imagen_rostro)

    etiqueta = etiqueta + 1

#   CREAR AL MODELO
reconocedor = cv2.face.LBPHFaceRecognizer_create()
print ("ENTRENANDO AL MODELO")

#   ENTRENAR AL MODELO
reconocedor.train(datos_rostros, np.array(etiquetas))

#   GUARDAR EL MODELO
reconocedor.write('modelo.xml')
print("MODELO GUARDADO CORRECTAMENTE")