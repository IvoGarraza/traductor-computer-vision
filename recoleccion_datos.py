import time
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

#====Configuracion inicial======
# Ruta del modelo de MediaPipe Hand Landmarker
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "hand_landmarker.task"
# Ruta del modelo de MediaPipe Face Landmarker
FACE_MODEL_PATH = SCRIPT_DIR / "face_landmarker.task"
# Configuración del dataset
N_SECUENCIAS = 100
SECUENCIAS_LENGTH = 30
LETRA = ""

#Estados
estado = 'inicio'
secuencia_actual = 0
frame_actual = 0

#variables para texto en pantalla
ubicacion = (50, 50)  # Coordenada (x, y)
fuente = cv2.FONT_HERSHEY_SIMPLEX
escala = 1.0
color = (0, 144, 255)  # Color Verde en BGR
grosor = 2

# Carpeta principal del dataset
DATASET_DIR = SCRIPT_DIR / "dataset"

# Crear carpeta de la letra
#LETRA_DIR = DATASET_DIR / LETRA
#LETRA_DIR.mkdir(parents=True, exist_ok=True)

# Crear las carpetas de las secuencias
#for secuencia in range(N_SECUENCIAS):
#    secuencia_dir = LETRA_DIR / str(secuencia)
#    secuencia_dir.mkdir(parents=True, exist_ok=True)


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

if not FACE_MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo de rostro en: {FACE_MODEL_PATH}")

# Definir la camara de video
cap = cv2.VideoCapture(0)


#==== Configuracon de variables de MediaPipe========
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode
# HandLandmarker y sus opciones
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# FaceLandmarker y sus opciones
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

#Función para dibujar los puntos de las manos
def dibujar_landmarks(frame, hand_landmarks, alto, ancho):
    for hand_landmark in hand_landmarks:
        #print(hand_landmark)
        for indice, landmark in enumerate(hand_landmark):
            # 'indice' valdrá 0, 1, 2, 3... 
            # 'landmark' contendrá las coordenadas correspondientes
            x = int(landmark.x * ancho)
            y = int(landmark.y * alto)
            # Condicion para dibujar circulos de colores distintos
            if indice == 4 or indice == 8 or indice == 12 or indice == 16 or indice == 20:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), 1) #circulo en las puntas Rojo 
                cv2.line(frame, (x, y), (int(hand_landmark[indice - 1].x * ancho), int(hand_landmark[indice - 1].y * alto)), (255, 255, 255), 1) #Dibujo de las primeras falanges
            else:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), 1) #Circulo Verde
            
            if indice == 9 or indice == 13 or indice == 17:
                cv2.line(frame, (x, y), (int(hand_landmark[indice - 4].x * ancho), int(hand_landmark[indice - 4].y * alto)), (255, 255, 255), 1) #Dibujo de nudillos
                
            if indice == 5 or indice == 17:
                cv2.line(frame, (x, y), (int(hand_landmark[0].x * ancho), int(hand_landmark[0].y * alto)), (255, 255, 255), 1) #dibujo de nudillos a la muñeca
            if indice == 7 or indice == 11 or indice == 15 or indice == 19 or indice == 6 or indice == 10 or indice == 14 or indice == 18 or indice == 3 or indice == 2 or indice == 1:
                cv2.line(frame, (x, y), (int(hand_landmark[indice - 1].x * ancho), int(hand_landmark[indice - 1].y * alto)), (255, 255, 255), 1) #Dibujo de las segundas falanges

#Función para dibujar los puntos del rostro
def dibujar_landmarks_rostro(frame, face_landmarks, alto, ancho):
    for face_landmark in face_landmarks:
        for indice, landmark in enumerate(face_landmark):
            x = int(landmark.x * ancho)
            y = int(landmark.y * alto)
            if indice in [0, 1, 234, 454, 152]:  # Ejemplo de índices de puntos de referencia del rostro
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1) # Dibuja un punto verde para los puntos de referencia seleccionados

#Opciones para configurar las manos de mediapipe
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

#Opciones para configurar el rostro de mediapipe
options_face = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    num_faces=1
)

#Función para extraer los puntos de la mano y acomodarlos en forma bidimensional
def extraer_datos_mano(resultados_mano):
    datos_mano = []
    if resultados_mano and resultados_mano.hand_landmarks:
        for hand_landmark in resultados_mano.hand_landmarks:
            mano = []
            for landmark in hand_landmark:
                mano.append((landmark.x, landmark.y, landmark.z))
            datos_mano.append(mano)
    datos_normalizados = normalizar_manos(datos_mano)
    return datos_normalizados

#Función para extraer los datos de rostro y colocarlos de forma bidimensional
def extraer_datos_rostro(rostro_resultados):
    datos_rostro = []
    for face_landmark in rostro_resultados.face_landmarks:
        rostro = []
        for indice, landmark in enumerate(face_landmark):
            if indice in [0, 1, 234, 454, 152]:# Ejemplo de índices de puntos de referencia del rostro
                rostro.append((landmark.x, landmark.y, landmark.z))
        datos_rostro.append(rostro)
    return normalizar_rostro(datos_rostro)

#Normalización de manos para tener siempre 2 manos
def normalizar_manos(datos_mano):
    if len(datos_mano) == 1:
        datos_mano.append([(0.0, 0.0, 0.0)] * 21)  # Agregar una mano vacía si solo hay una mano detectada
    elif len(datos_mano) == 0:
        datos_mano.append([(0.0, 0.0, 0.0)] * 21)
        datos_mano.append([(0.0, 0.0, 0.0)] * 21)   # Agregar dos manos vacías si no se detecta ninguna mano
    return datos_mano

#Normalizacion de rostro para tener siempre 1 rostro
def normalizar_rostro(datos_rostro):
    if len(datos_rostro) == 0:
        datos_rostro.append([(0.0, 0.0, 0.0)] * 5)  # Agregar un rostro vacío si no se detecta ningún rostro
    return datos_rostro

#Función para construir el vector de las posiciones de las 2 manos, el rostro y la distancia mano-rostro
def constructor_de_vectores(datos_mano, datos_rostro, distancia_mano_rostro):
    vector_mano_1 = np.array(datos_mano[0]).flatten()  # Mano izquierda
    vector_mano_2 = np.array(datos_mano[1]).flatten()  # Mano derecha
    vector_rostro = np.array(datos_rostro[0]).flatten()     # Rostro
    vector_distancia = np.array(distancia_mano_rostro).flatten()
    vector_final = np.concatenate((vector_mano_1, vector_mano_2, vector_rostro, vector_distancia))  # Concatenar todos los vectores
    return vector_final

def normalizar_manos_rostro(resultados_mano, resultados_rostro):
    resultados_manos_rostro = []
    for mano in resultados_mano:
        wrist = mano[0]
        if wrist == (0,0,0):
            resultados_manos_rostro.append(wrist)
        else:
            punto_cara = resultados_rostro[0][4]
            diferencia = (wrist[0] - punto_cara[0], wrist[1] - punto_cara[1], wrist[2] - punto_cara[2])
            resultados_manos_rostro.append(diferencia)
    return resultados_manos_rostro


def normalizar_posicion_manos(resultados_mano):
    #Función para restar muñeca en relación con los demás landmarks
    resultados_normalizados = []
    for mano in resultados_mano:
        wrist = mano[0] # Variable de landmark de la muñeca
        mano_normalizada = []
        for landmark in mano:
            landmark_normalizado = (landmark[0]- wrist[0], landmark[1]- wrist[1], landmark[2]- wrist[2]) # Se resta la cada landmark de la muñeca a demas landmarks
            mano_normalizada.append(landmark_normalizado)
        resultados_normalizados.append(mano_normalizada)
    return resultados_normalizados



#Comienzo de loop principal
with HandLandmarker.create_from_options(options) as landmarker, FaceLandmarker.create_from_options(options_face) as face_landmarker:
    while True:
        ret, frame = cap.read()
        alto, ancho, _ = frame.shape #obtenemos la altura y ancho del frame
        print(f"Resolución del frame: {ancho}x{alto}")
        
        if not ret:
            print("No se pudo acceder a la cámara.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # == Cambio de BGR a RGB ==
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        resultados_mano = landmarker.detect(mp_image) # Detectamos la mano
        resultados_rostro = face_landmarker.detect(mp_image) # Detectamos el rostro
        
        # Dibujo de los landmarks del rostro si se detectan
        if resultados_rostro and resultados_rostro.face_landmarks:
            alto, ancho, _ = frame.shape
            dibujar_landmarks_rostro(frame, resultados_rostro.face_landmarks, alto, ancho)
        # Dibujo de los landmarks de la mano si se detectan
        if resultados_mano and resultados_mano.hand_landmarks:
            alto, ancho, _ = frame.shape
            dibujar_landmarks(frame, resultados_mano.hand_landmarks, alto, ancho) # Llamada a la funcion para dibujar los landmarks en la pantalla
        #Puntos de las manos crudos, sin normalizar
        datos_mano_crudo = extraer_datos_mano(resultados_mano)  # Llamada a la función para extraer datos de la mano sin normalizar
        #Puntos de las manos normalizados
        datos_mano_normalizada= normalizar_posicion_manos(datos_mano_crudo) #Función para normalizar las manos
        #Puntos del rostro (5 puntos)
        datos_rostro = extraer_datos_rostro(resultados_rostro)  # Llamada a la función para extraer datos del rostro
        #Distancia de manos y rostro
        distancia_mano_rostro = normalizar_manos_rostro(datos_mano_crudo,datos_rostro)
        #print('Distancia mano cara:', distancia_mano_rostro)
        #Vector final con los datos de las manos y los rostros normalizados
        vector_final = constructor_de_vectores(datos_mano_normalizada, datos_rostro, distancia_mano_rostro)  # Llamada a la función para construir el vector final
        #print(vector_final.shape)
        #cv2.putText(frame,estado, ubicacion, fuente, escala, color,grosor) #muestra de texto en pantalla
        #cv2.putText(frame, 'secuencia:' + str(secuencia_actual), (50,80), fuente, escala, color, grosor)
        #cv2.putText(frame, "frames:" + str(frame_actual), (50,110), fuente, escala, color, grosor)
        #cv2.imshow("Traductor de lenguaje de señas", frame)

        # --- 1. DIBUJAR TEXTOS SEGÚN EL ESTADO ---
        if estado == 'inicio':
            # Mostramos el mensaje que pediste
            cv2.putText(frame, 'Apreta la letra a grabar', (50, 50), fuente, escala, color, grosor)
        elif estado == 'esperando':
            cv2.putText(frame, estado, ubicacion, fuente, escala, color, grosor)
            cv2.putText(frame, 'secuencia:' + str(secuencia_actual), (50, 80), fuente, escala, color, grosor)
            cv2.putText(frame, "frames:" + str(frame_actual), (50, 110), fuente, escala, color, grosor)
        elif estado == 'grabando':
            cv2.putText(frame, estado, ubicacion, fuente, escala, color, grosor)
            cv2.putText(frame, 'secuencia:' + str(secuencia_actual), (50, 80), fuente, escala, color, grosor)
            cv2.putText(frame, "frames:" + str(frame_actual), (50, 110), fuente, escala, color, grosor)


        # --- 2. MOSTRAR EL FRAME ---
        cv2.imshow("Traductor de lenguaje de senas", frame)

        # --- 3. CAPTURAR LA TECLA (Una sola vez por ciclo) ---
        tecla = cv2.waitKey(1) & 0xFF
        #Para grabar la letra CH hay que hardcodearla
        # --- 4. LÓGICA DE TRANSICIÓN DE ESTADOS ---
        if estado == 'inicio':
            if tecla != 255:
                caracter = chr(tecla)
                
                # Validamos que la tecla sea estrictamente una letra o número
                if caracter.isalnum():
                    LETRA = caracter.lower()  # Guarda en minúscula para uniformidad

                    # Creación de carpetas solo si es un carácter válido
                    LETRA_DIR = DATASET_DIR / LETRA
                    LETRA_DIR.mkdir(parents=True, exist_ok=True)
                    for secuencia in range(N_SECUENCIAS):
                        secuencia_dir = LETRA_DIR / str(secuencia)
                        secuencia_dir.mkdir(parents=True, exist_ok=True)
                    
                    estado = 'esperando'  # Pasamos al siguiente estado

        elif estado == 'esperando':
            color = (0, 144, 255)
            if tecla == ord(" ") and secuencia_actual < N_SECUENCIAS:
                estado = 'grabando'

        elif estado == 'grabando':
            color = (0, 255, 0)
            np.save(LETRA_DIR / str(secuencia_actual) / f"{frame_actual}.npy", vector_final)
            frame_actual += 1
            if frame_actual == SECUENCIAS_LENGTH:
                estado = 'esperando'
                secuencia_actual += 1
                frame_actual = 0

        # --- 5. SALIR DEL PROGRAMA ---
        if tecla == 27:  # 27 es el código ASCII de la tecla Escape
            break

cap.release()
cv2.destroyAllWindows()