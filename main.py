## Recoleccion de datos para entrenamiento - Traductor de lenguaje de señas LSA (Lenguajes de señas Argentino)
import cv2
import mediapipe as mp
import numpy as np


#Configuracion de la camara web
cap = cv2.VideoCapture(0)

#----Bucle principal para capturar y mostrar la imagen de la camara--------
while True:
    #Captura de la imagen de la camara
    ret, frame = cap.read()
    if not ret:
        break

    #Mostrar la imagen en una ventana
    cv2.imshow('Traductor de lenguaje de señas', frame)

    #Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#----Fin del bucle principal------------------------------------------------

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()