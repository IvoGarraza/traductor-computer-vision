import random

import numpy as np
from pathlib import Path
import os

# Ruta del archivo .npy
SCRIPT_DIR = Path(__file__).resolve().parent
FILE_PATH = SCRIPT_DIR / "dataset" / "b" / "3"

#cantidad_carpetas = Path.iterdir(SCRIPT_DIR / "dataset" / "a" )
#cantidad_carpetas = len(os.listdir(FILE_PATH))
carpeta = 0
for carpeta in range(100):
    direccion_carpeta = SCRIPT_DIR / "dataset" / "b" / str(carpeta)
    cantidad_carpetas = len(os.listdir(direccion_carpeta))
    if cantidad_carpetas == 30:
        print("Analizando carpeta:", carpeta)
        nombre_archivo1 = os.listdir(direccion_carpeta)[0]
        #nombre_archivo1= random.choice(os.listdir(direccion_carpeta))
        ruta_completa1 = direccion_carpeta / nombre_archivo1
        carga_npy1 = np.load(ruta_completa1)
        print(carga_npy1.shape)
        #nombre_archivo2= random.choice(os.listdir(direccion_carpeta))
        nombre_archivo2 = os.listdir(direccion_carpeta)[29]
        ruta_completa2 = direccion_carpeta / nombre_archivo2
        carga_npy2 = np.load(ruta_completa2)
        print(carga_npy2.shape)
        son_iguales = np.array_equal(carga_npy1, carga_npy2)
        print(son_iguales)
        carpeta += 1
        pass
    else:
        print("Falta un archivo en la carpeta:", carpeta)
        break
# Cargar archivo
#data = np.load(FILE_PATH)

# Mostrar información
#print(cantidad_carpetas)
#print("Archivo:", FILE_PATH)
#print("Tipo:", type(data))
#print("Shape:", data.shape)
#print("Dtype:", data.dtype)
#print("\nContenido:")
#print(data)

