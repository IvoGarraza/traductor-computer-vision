import os
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
file_path = SCRIPT_DIR / "dataset"

carpetas_letras = os.listdir(file_path)
array_x = []
array_y = []

for letra in carpetas_letras:
    letra_dir = file_path / letra
    letras = os.listdir(letra_dir)
    for secuencias in letras:
        #print("secuencias:",secuencias)
        secuencias_dir = sorted(os.listdir(letra_dir / secuencias), key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else x)
        #print(secuencias_dir)
        frames = []
        for frame in secuencias_dir:
            ruta_frame = np.load(letra_dir / secuencias / frame)
            frames.append(ruta_frame)
        frames_array = np.array(frames)
        #print('array_temporal:',frames_array.shape)
        array_x.append(frames_array)
        array_y.append(letra)
X = np.array(array_x)
Y = np.array(array_y)
print(X.shape)  # esperás (n_muestras_totales, 30, 147)
print(Y.shape)  # esperás (n_muestras_totales,)
print(Y[:5])    # mirá que las etiquetas sean texto, tipo ['a', 'a', 'a', 'a', 'a']