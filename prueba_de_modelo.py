import time
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import tensorflow as tf
from collections import deque
import pyttsx3

# ==== Configuración inicial ======
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "hand_landmarker.task"
FACE_MODEL_PATH = SCRIPT_DIR / "face_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

if not FACE_MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo de rostro en: {FACE_MODEL_PATH}")

SECUENCIAS_LENGTH = 30

# Cargar el modelo entrenado (.keras o .h5) y las clases del encoder
MODELO_FILE = SCRIPT_DIR / 'modelo_entrenado.keras'
if not MODELO_FILE.exists():
    MODELO_FILE = SCRIPT_DIR / 'modelo_entrenado.h5'

modelo = tf.keras.models.load_model(MODELO_FILE)
clases = np.load(SCRIPT_DIR / 'clases.npy')

# Variables de la máquina de estados e inferencia
buffer = deque(maxlen=SECUENCIAS_LENGTH)

# Variables para almacenar predicciones y letras acumuladas
prediccion_actual = ""
texto_acumulado = ""


# Variables para texto en pantalla
fuente = cv2.FONT_HERSHEY_SIMPLEX
escala = 0.8
grosor = 21

# Definir la cámara de video
cap = cv2.VideoCapture(1)

# ==== Configuración de variables de MediaPipe ========
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

# Opciones para configurar MediaPipe
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

options_face = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    num_faces=1
)

#configuraccion de pyttsx3
engine = pyttsx3.init()
def lectura_letras(conjunto_letras):
    engine.say(conjunto_letras)
    engine.runAndWait()

# Funcióes de dibujo y extracción de landmarks
def dibujar_landmarks(frame, hand_landmarks, alto, ancho):
    for hand_landmark in hand_landmarks:
        for indice, landmark in enumerate(hand_landmark):
            x = int(landmark.x * ancho)
            y = int(landmark.y * alto)
            if indice in [4, 8, 12, 16, 20]:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), 1)
                cv2.line(frame, (x, y),
                         (int(hand_landmark[indice - 1].x * ancho), int(hand_landmark[indice - 1].y * alto)),
                         (255, 255, 255), 1)
            else:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), 1)

            if indice in [9, 13, 17]:
                cv2.line(frame, (x, y),
                         (int(hand_landmark[indice - 4].x * ancho), int(hand_landmark[indice - 4].y * alto)),
                         (255, 255, 255), 1)

            if indice in [5, 17]:
                cv2.line(frame, (x, y), (int(hand_landmark[0].x * ancho), int(hand_landmark[0].y * alto)),
                         (255, 255, 255), 1)
            if indice in [1, 2, 3, 6, 7, 10, 11, 14, 15, 18, 19]:
                cv2.line(frame, (x, y),
                         (int(hand_landmark[indice - 1].x * ancho), int(hand_landmark[indice - 1].y * alto)),
                         (255, 255, 255), 1)


def dibujar_landmarks_rostro(frame, face_landmarks, alto, ancho):
    for face_landmark in face_landmarks:
        for indice, landmark in enumerate(face_landmark):
            x = int(landmark.x * ancho)
            y = int(landmark.y * alto)
            if indice in [0, 1, 234, 454, 152]:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


def extraer_datos_mano(resultados_mano):
    datos_mano = []
    if resultados_mano and resultados_mano.hand_landmarks:
        for hand_landmark in resultados_mano.hand_landmarks:
            mano = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmark]
            datos_mano.append(mano)
    return normalizar_manos(datos_mano)


def extraer_datos_rostro(rostro_resultados):
    datos_rostro = []
    if rostro_resultados and rostro_resultados.face_landmarks:
        for face_landmark in rostro_resultados.face_landmarks:
            rostro = [
                (landmark.x, landmark.y, landmark.z)
                for indice, landmark in enumerate(face_landmark)
                if indice in [0, 1, 234, 454, 152]
            ]
            datos_rostro.append(rostro)
    return normalizar_rostro(datos_rostro)


def normalizar_manos(datos_mano):
    if len(datos_mano) == 1:
        datos_mano.append([(0.0, 0.0, 0.0)] * 21)
    elif len(datos_mano) == 0:
        datos_mano.append([(0.0, 0.0, 0.0)] * 21)
        datos_mano.append([(0.0, 0.0, 0.0)] * 21)
    return datos_mano


def normalizar_rostro(datos_rostro):
    if len(datos_rostro) == 0:
        datos_rostro.append([(0.0, 0.0, 0.0)] * 5)
    return datos_rostro


def normalizar_manos_rostro(resultados_mano, resultados_rostro):
    resultados_manos_rostro = []
    for mano in resultados_mano:
        wrist = mano[0]
        if wrist == (0, 0, 0):
            resultados_manos_rostro.append(wrist)
        else:
            punto_cara = resultados_rostro[0][4]
            diferencia = (wrist[0] - punto_cara[0], wrist[1] - punto_cara[1], wrist[2] - punto_cara[2])
            resultados_manos_rostro.append(diferencia)
    return resultados_manos_rostro


def normalizar_posicion_manos(resultados_mano):
    resultados_normalizados = []
    for mano in resultados_mano:
        wrist = mano[0]
        mano_normalizada = [
            (landmark[0] - wrist[0], landmark[1] - wrist[1], landmark[2] - wrist[2])
            for landmark in mano
        ]
        resultados_normalizados.append(mano_normalizada)
    return resultados_normalizados


def constructor_de_vectores(datos_mano, datos_rostro, distancia_mano_rostro):
    vector_mano_1 = np.array(datos_mano[0]).flatten()
    vector_mano_2 = np.array(datos_mano[1]).flatten()
    vector_rostro = np.array(datos_rostro[0]).flatten()
    vector_distancia = np.array(distancia_mano_rostro).flatten()
    return np.concatenate((vector_mano_1, vector_mano_2, vector_rostro, vector_distancia))


# ==== Comienzo del loop principal ====
with HandLandmarker.create_from_options(options) as landmarker, FaceLandmarker.create_from_options(
        options_face) as face_landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            break

        alto, ancho, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        resultados_mano = landmarker.detect(mp_image)
        resultados_rostro = face_landmarker.detect(mp_image)

        # Dibujar landmarks en el frame
        if resultados_rostro and resultados_rostro.face_landmarks:
            dibujar_landmarks_rostro(frame, resultados_rostro.face_landmarks, alto, ancho)
        if resultados_mano and resultados_mano.hand_landmarks:
            dibujar_landmarks(frame, resultados_mano.hand_landmarks, alto, ancho)

        # Extracción y construcción del vector de características (147,)
        datos_mano_crudo = extraer_datos_mano(resultados_mano)
        datos_mano_normalizada = normalizar_posicion_manos(datos_mano_crudo)
        datos_rostro = extraer_datos_rostro(resultados_rostro)
        distancia_mano_rostro = normalizar_manos_rostro(datos_mano_crudo, datos_rostro)

        vector_final = constructor_de_vectores(datos_mano_normalizada, datos_rostro, distancia_mano_rostro)
        
        buffer.append(vector_final)
        
        if len(buffer) == SECUENCIAS_LENGTH:
            # 1. Convertir buffer a array (30, 147) y expandir dimensión a (1, 30, 147)
            secuencia_array = np.array(buffer)
            input_data = np.expand_dims(secuencia_array, axis=0)

            # 2. Predecir
            predicciones = modelo.predict(input_data, verbose=0)
            indice_max = np.argmax(predicciones[0])
            prediccion_actual = str(clases[indice_max])
            

        # UI: Dibujar la predicción actual y el string acumulado en el frame
        cv2.putText(frame, f"Prediccion: {prediccion_actual}", (10, 30), fuente, escala, (255, 0, 0), grosor)
        cv2.putText(frame, f"Palabra: {texto_acumulado}", (10, 70), fuente, escala, (0, 255, 0), grosor)


        # Captura de teclado
        tecla = cv2.waitKey(1) & 0xFF

        #Almacenamiento de la predicción actual en un archivo de texto
        if tecla == 32:  # Tecla Espacio
            if prediccion_actual:
                texto_acumulado += prediccion_actual
                print(f"Letra agregada. Texto actual: {texto_acumulado}")

        # Reproducir la predicción actual en voz alta
        if tecla == 13:  # Tecla Enter
            lectura_letras(texto_acumulado)
            
        # Agregar un espacio al texto acumulado
        if tecla == 9:  # Tecla Tab
            texto_acumulado += " "
            print(f"Espacio agregado. Texto actual: {texto_acumulado}")
            
        # Borrar el texto acumulado
        if tecla == 8:  # Tecla Backspace
            texto_acumulado = ""
            print("Texto acumulado borrado.")
        
        # Borrar una letra del texto acumulado
        if tecla == 15:  # Shift out
            texto_acumulado = texto_acumulado[:-1]
            print(f"Última letra borrada. Texto actual: {texto_acumulado}")
            

        cv2.imshow("Traductor de lenguaje de senas", frame)

        # Salir con tecla ESC (ASCII 27)
        if tecla == 27:
            break

cap.release()
cv2.destroyAllWindows()