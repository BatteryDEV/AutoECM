import tensorflow as tf
import keras.api._v2.keras as keras
from keras import layers
from keras import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import pickle


print("Tensorflow version " + tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
X = X/255.0

# Looking for NaN values
nan_X = np.isnan(np.min(X))
print(f"Is there NaN element in X? \n {nan_X}")
nan_y = np.isnan(np.min(y))
print(f"Is there NaN element in Y? \n {nan_y}")

model = keras.models.Sequential()
model.add(layers.Conv2D(128, (3,3), strides=(1,1), padding="valid", activation='relu', input_shape = (128,128,3)))
model.add(layers.MaxPool2D(2,2))
# model.add(layers.Conv2D(128, 3, activation='relu'))
# model.add(layers.MaxPool2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(8))

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # Applies the Softmax
optim = keras.optimizers.Adam(learning_rate=0.00001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

#training
batch_size = 26
epochs = 10

model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

