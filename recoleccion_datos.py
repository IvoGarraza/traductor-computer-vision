import time
import cv2
import mediapipe as mp
from pathlib import Path

#====Configuracion inicial======
# Ruta del modelo de MediaPipe Hand Landmarker
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "hand_landmarker.task"
# Ruta del modelo de MediaPipe Face Landmarker
FACE_MODEL_PATH = SCRIPT_DIR / "face_landmarker.task"


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

resultados_mano = None # Variable global para almacenar los resultados de la detección de manos
resultados_rostro = None # Variable global para almacenar los resultados de la detección de rostros

def procesar_resultado(result, output_image, timestamp_ms: int):
    global resultados_mano
    resultados_mano = result # Guardamos el resultado completo aquí para usarlo después
    if result.hand_landmarks:
        print(f"¡Se detectaron {len(result.hand_landmarks)} manos!")

def procesar_resultado_face(result, output_image, timestamp_ms: int):
    global resultados_rostro
    resultados_rostro = result # Guardamos el resultado completo aquí para usarlo después
    if result.face_landmarks:
        print(f"¡Se detectaron {len(result.face_landmarks)} rostros!")

def dibujar_landmarks(frame, hand_landmarks, alto, ancho):
    for hand_landmark in hand_landmarks:
        print(hand_landmark)
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

def dibujar_landmarks_rostro(frame, face_landmarks, alto, ancho):
    for face_landmark in face_landmarks:
        for indice, landmark in enumerate(face_landmark):
            x = int(landmark.x * ancho)
            y = int(landmark.y * alto)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1) # Dibuja un punto azul para cada punto de referencia del rostro

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=procesar_resultado,
    num_hands=2
)

options_face = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=procesar_resultado_face,
    num_faces=1
)

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
        landmarker.detect_async(mp_image, int(time.time() * 1000))
        face_landmarker.detect_async(mp_image, int(time.time() * 1000))
        
        # Dibujo de los landmarks del rostro si se detectan
        if resultados_rostro and resultados_rostro.face_landmarks:
            alto, ancho, _ = frame.shape
            dibujar_landmarks_rostro(frame, resultados_rostro.face_landmarks, alto, ancho)
        # Dibujo de los landmarks de la mano si se detectan
        if resultados_mano and resultados_mano.hand_landmarks:
            alto, ancho, _ = frame.shape
            # Llamamos a la función matemática que creaste
            dibujar_landmarks(frame, resultados_mano.hand_landmarks, alto, ancho)
            
        cv2.imshow("Traductor de lenguaje de señas", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()