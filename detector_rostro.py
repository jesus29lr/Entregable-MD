#   LIBRERIAS
import cv2
import os

#   UBICACION DE LOS DATOS
ruta_datos = "data"

#   NOMBRE DE LAS PERSONAS
lista_personas = sorted(os.listdir(ruta_datos))
print ("PERSONAS REGISTRADAS: ", lista_personas)

#   CREAR AL MODELO
reconocedor = cv2.face.LBPHFaceRecognizer_create()
reconocedor.read("modelo.xml")
print("MODELO CARGADO CORRECTAMENTE")

#CARGAR EL DETECTOR DE ROSTROS
detector = cv2.CascadeClassifier("rostros.xml")

if detector.empty():
    print ("Error: NO SE PUDO CARGAR EL DETECTOR DE ROSTROS")
    exit()

# ACTIVAR LA CAMARA
camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#   VERIFICAR LA CAMARA
if not camara.isOpened():
    print("Error: NO SE PUDO ABRIR LA CAMARA")
    exit()
print("CAMARA ACTIVADA CORRECTAMENTE")

#   MANTENER LA CAMARA PRENDIDA
while True:
    resultado, imagen = camara.read()

    if not resultado:
        print("ERROR: AL CAPTURAR LA IMAGEN")
        break

    #   CONVERITR A ESCALA DE GRISES
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    imagen_auxiliar = gris.copy()

    rostros = detector.detectMultiScale(
        gris,
        scaleFactor= 1.3,
        minNeighbors=5,
    )

    #   RECORRER LOS ROSTROS DETECTADOS
    for(x, y, ancho, alto) in rostros:
        rostro = imagen_auxiliar[
            y:y + alto,
            x:x + ancho
        ]

        rostro = cv2.resize(rostro, (150, 150))

        resultado_prediccion = reconocedor.predict(rostro)

        etiqueta = resultado_prediccion[0]
        distancia = resultado_prediccion[1]

        if distancia < 75 and etiqueta < len (lista_personas):
            nombre = lista_personas[etiqueta]
            color = (0, 255, 0)

        else:
            nombre = "DESCONOCIDO"
            color = (0, 0, 255)

        #   MOSTRAR EL NOMBRE
        cv2.putText(
            imagen,
            nombre,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        #   DIBUJAR EL RECTANGULO
        cv2.rectangle(
            imagen,
            (x, y),
            (x + ancho, y + alto),
            color,
            2
        )

    #   MOSTRAR LA CAMARA
    cv2.imshow("RECONOCIMIENTO FACIAL", imagen)

    #   SALIR PRESIONANDO Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()

print("RECONOCIMIENTO FACIAL FINALIZADO")