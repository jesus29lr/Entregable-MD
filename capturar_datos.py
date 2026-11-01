#   LIBRERIAS
import cv2
import imutils
import os

#   NOMBRE A REGISTRAR
nombre = input("ESCRIBE TU NOMBRE: ").strip()

carpeta = os.path.join("data", nombre)

if os.path.exists(carpeta):
    print("LA PERSONA YA EXISTE")
else:
    os.makedirs(carpeta)
    print("NUEVA PERSONA REGISTRADA")

# ACTIVAR LA CAMARA
camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#   VERIFICAR LA CAMARA
if not camara.isOpened():
    print("Error: NO SE PUDO ABRIR LA CAMARA")
    exit()

print("CAMARA ACTIVADA CORRECTAMENTE")

#CARGAR EL DETECTOR DE ROSTROS
detector = cv2.CascadeClassifier("rostros.xml")

if detector.empty():
    print ("Error: NO SE PUDO CARGAR EL DETECTOR DE ROSTROS")
    camara.release()
    exit()

contador = 0

#   MANTENER LA CAMARA PRENDIDA
while True:
    resultado, imagen = camara.read()

    if not resultado:
        print("ERROE: AL CAPTURAR LA IMAGEN ")
        break

    #   INVERTIR IMAGEN
    imagen_espejo = cv2.flip(imagen, 1)

    #   REDIMENSIONAR LA IMAGEN
    imagen = imutils.resize(imagen, width=640)

    #   CONVERITR A ESCALA DE GRISES
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    rostros = detector.detectMultiScale(
        gris,
        scaleFactor= 1.1,
        minNeighbors=5,
        minSize=(120,120),
        maxSize=(1000, 1000)
    )

    #   DIBUJAR RECTANGULOS
    for(x, y, w, h) in rostros:
        cv2.rectangle(
            imagen,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        #   RECORTAR LA IMAGEN - SOLO ROSTRO
        rostro = imagen[y:y + h, x:x + w].copy()

        rostro = cv2.resize(rostro, (150, 150))

        ruta = os.path.join(
            carpeta,
            f"{nombre}_{contador}.jpg"
        )

        cv2.imwrite(ruta, rostro)

        print(f"Fotografia {contador + 1} guardada")

        contador += 1

    #   MOSTRAR EL CONTADOR
    cv2.putText(
        imagen,
        f"Fotos: {contador}/300",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    #   MOSTRAR LA CAMARA
    cv2.imshow("DETECTOR DE ROSTROS", imagen)

    #   SALIR PRESIONANDO Q
    if contador >= 300 or cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()