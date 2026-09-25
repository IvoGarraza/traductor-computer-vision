import sklearn.preprocessing as prep
import sklearn.model_selection as ms
import tensorflow as tf
import numpy as np

from cargar_dataset import Y, X #Variable Y, X del script 'cargar_dataset.py'


encoder = prep.LabelEncoder() #funcion encoder de scikit-learn
Y_codificado = encoder.fit_transform(Y) #Y Transformamos el formato de texto a [0,1]
np.save('clases.npy', encoder.classes_)
X_one_hot = X                        #X No es necesario transformacion
#print(Y_codificado.shape)
Y_one_hot = tf.keras.utils.to_categorical(Y_codificado)
#print(Y_one_hot)
x_train, x_test, y_train, y_test = ms.train_test_split(X_one_hot, Y_one_hot,  test_size=0.3, random_state=42)
print(y_train.shape[1])
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=x_train.shape[1:3]))
model.add(tf.keras.layers.GRU(64))       # capa 1: GRU, con input_shape
model.add(tf.keras.layers.Dense(units=32, activation='relu'))    # capa 2: Dense intermedia
model.add(tf.keras.layers.Dense(units=y_train.shape[1], activation='softmax'))  # capa 3: Dense de salida
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#Tupla para validacion
validation_data = (x_test, y_test)

history = model.fit(x_train, y_train, epochs=40, batch_size=16, validation_data=validation_data)
tf.keras.models.save_model(model, 'modelo_entrenado.keras') # Guardar el modelo entrenado en un archivo .h5
