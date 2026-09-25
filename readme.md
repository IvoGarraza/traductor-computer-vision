# TRADUCTOR DE LENGUAJE DE SEÑAS ARGENTINO

Esta app en python utiliza mediapipe para la deteccion de manos
y se entreno un modelo de IA con los diferentes gestos correspondientes al lenguaje de señas argentino

## Instrucciones de uso
# Paso 1
Instalar las dependencias necesarias:

```console
pip install requeriments.txt
```
# Paso 2 
Recolectar datos con recoleccion_datos.py

```console
python recoleccion_datos.py
```
Grabar todas las secuencias del alfabeto con presionando espacio para comenzar a grabar y corriendola 100 veces por cada letra

# Paso 3
Verificar toda la informacion recolectada con test.py
```console
python test.py
```
# Paso 4
Preparar dataset para el entrenamiento con cargar_dataset.py
```console
python cargar_dataset.py
```

# Paso 5
