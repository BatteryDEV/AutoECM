# CNN Baseline model 
# Largely inspired by Kaggle MNIST approach:
# https://www.kaggle.com/code/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1/notebook
# The architecture is only slightly adapted to account for bigger images.
# Many lines of code are directly copeid form the above mentioned example.

import pandas as pd
import numpy as np

import datetime
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Lambda, MaxPooling2D, Rescaling # convolution layers
from keras.layers import Dense, Dropout, Flatten # core layers

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

import json 

from utils import plot_cm, calcualte_classification_report


def cnn_architecture(input_shape, nb_classes, adaptive_based_on_val=True):
    """CNN architecture
    Parameters
    ----------
    input_shape: tuple
        Shape of input data
    nb_classes: int
        Number of classes
    adaptive_based_on_val_loss: bool
        If True, the model os compiled with an optimizer and loss that is monitored for 
        early stopping. If False, the model is compiled with a another optimizer.

    Returns
    -------
    model: keras model
        CNN model
    """
    model=Sequential()
    model.add(Rescaling(1./127.5, offset=-1, input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(Conv2D(filters=64, kernel_size = (6,6), activation="relu", strides=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())    

    model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
        
    model.add(Flatten())
    model.add(Dense(512,activation="relu"))
        
    model.add(Dense(nb_classes, activation="softmax"))
    if adaptive_based_on_val:
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    else:
        ada_delta_ = keras.optimizers.Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.03)
        model.compile(loss="categorical_crossentropy", optimizer=ada_delta_, metrics=["accuracy"])
    
    model.summary()

    return model

def train_model(model, train_ds, val_ds, adaptive_based_on_val):
    if adaptive_based_on_val:
        es = keras.callbacks.EarlyStopping(
                monitor="val_accuracy", # metrics to monitor
                patience=10, # how many epochs before stop
                verbose=1,
                mode="max", # we need the maximum accuracy.
                restore_best_weights=True, # 
                )

        rp = keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.2,
                patience=3,
                verbose=1,
                mode="max",
                min_lr=0.00001,
                )
        h = model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=[rp, es])
        # Do 5 epochs with a very low learning rate
        # Can be omitted if the model is already well converged
        model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-5), metrics=["accuracy"])
        model = model.fit(val_ds, epochs=5)
    else:
        model = model.fit(train_ds, epochs=140)
    return model

def main(input_shape, nb_classes=9, output_dir="", adaptive_based_on_val=True):
    if adaptive_based_on_val:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            "data/images/train",
            validation_split=0.1,
            subset="training",
            seed=20,
            image_size=input_shape,
            label_mode="categorical",
            shuffle=True,
            color_mode="grayscale",
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            "data/images/train",
            validation_split=0.1,
            subset="validation",
            seed=20,
            image_size=input_shape,
            batch_size=64,
            label_mode="categorical",
            shuffle=False,
            color_mode="grayscale",
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            "data/images/train",
            seed=42,
            image_size=input_shape,
            label_mode="categorical",
            shuffle=True,
            batch_size=1000,
            color_mode="grayscale",
        )
        val_ds = np.nan
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        "data/images/test",
        validation_split=None,
        seed=42,
        image_size=input_shape,
        batch_size=2000,
        label_mode="categorical",
        shuffle=False,
        color_mode="grayscale",
    )
    for image_batch, labels_batch in train_ds:
        y_train = np.array(labels_batch)
    for image_batch, labels_batch in test_ds:
        y_test = np.array(labels_batch)

    # Build model
    model = cnn_architecture(input_shape, nb_classes, adaptive_based_on_val=adaptive_based_on_val)
    # Train model
    model = train_model(model, train_ds, val_ds, adaptive_based_on_val=adaptive_based_on_val)

    y_test_pred = model.predict(test_ds) # Predict class probabilities as 2 => [0.1, 0, 0.9, 0, 0, 0, 0, 0, 0, 0]
    Y_test_pred = np.argmax(y_test_pred, 1) # Decode Predicted labels
    y_train_pred = model.predict(train_ds) # Predict class probabilities as 2 => [0.1, 0, 0.9, 0, 0, 0, 0, 0, 0, 0]
    Y_train_pred = np.argmax(y_train_pred, 1) # Decode Predicted labels
    # Y_test_true = np.argmax(y_test, 1) # Decode Predicted labels
    f1_score(y_test, Y_test_pred, average='macro')

    with open('data/le_name_mapping.json', 'r') as f:
        mapping = json.load(f)
        le = LabelEncoder()
    mapping['classes'] = [mapping[str(int(i))] for i in range(9)]
    le.classes_ = np.array(mapping['classes'])

    plot_cm(y_test, Y_test_pred, le, save=0, title='Confusion Matrix: CNN Model', figname='cnn_cm', save_path='figures/CNN/')

    proportion_correct = f1_score(y_test, Y_test_pred, average='macro')
    print('Test Accuracy: {}'.format(proportion_correct))

    # Save model and weights
    model.save(f'{output_dir}/cnn_model.h5')

    # Calculate f1 and save classification report
    calcualte_classification_report(y_train, y_train_pred, y_test, Y_test_pred, le, save=1, output_dir=output_dir)

    return train_ds, val_ds

if __name__ == "__main__":
    input_shape = (58, 58)
    nb_classes = 9
    adaptive_based_on_val = True  
    # if False, then uses ada_delta with decay, else uses stopping and decay based on val_acc

    # Create new folder with results, name is datetime
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"results/clf/cnn/{now_str}"
    os.mkdir(output_dir)

    train_ds, val_ds = main(input_shape, nb_classes, output_dir, adaptive_based_on_val)
