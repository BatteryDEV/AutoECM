# CNN Baseline model
# Largely inspired by Kaggle MNIST approach:
# https://www.kaggle.com/code/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1/notebook
# The architecture is only slightly adapted to account for bigger images.
# Many lines of code are directly copeid form the above mentioned example.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Lambda, MaxPooling2D, Rescaling  # convolution layers
from keras.layers import Dense, Dropout, Flatten  # core layers

import tensorflow as tf
from tensorflow import keras
from keras.layers import BatchNormalization

from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

import json

from utils import plot_cm, calcualte_classification_report

# Set seeds for reproducibility
# Some randomness might still be present depending on the used libraries/hardware/GPU
# https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
# tf.random.set_seed(20)
# np.random.seed(20)


def cnn_architecture(
    input_shape: Tuple[int, int], nb_classes: int, adaptive_based_on_val: bool = True
) -> keras.Model:
    """CNN architecture
    Parameters
    ----------
    input_shape: Tuple
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
    model = Sequential()
    model.add(
        Rescaling(1.0 / 127.5, offset=-1, input_shape=(input_shape[0], input_shape[1], 1))
    )
    model.add(Conv2D(filters=64, kernel_size=(6, 6), activation="relu", strides=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))

    model.add(Dense(nb_classes, activation="softmax"))
    if adaptive_based_on_val:
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
    else:
        ada_delta_ = keras.optimizers.Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.03)
        model.compile(
            loss="categorical_crossentropy", optimizer=ada_delta_, metrics=["accuracy"]
        )

    model.summary()

    return model


def train_model(
    model: keras.Model, train_ds, val_ds, adaptive_based_on_val: bool
) -> Tuple[keras.callbacks.History, keras.Model]:
    if adaptive_based_on_val:
        es = keras.callbacks.EarlyStopping(
            monitor="val_accuracy",  # metrics to monitor
            patience=10,  # how many epochs before stop
            verbose=1,
            mode="max",  # we need the maximum accuracy.
            restore_best_weights=True,  #
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
        # model.compile(
        #     loss="categorical_crossentropy",
        #     optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        #     metrics=["accuracy"],
        # )
        # h = model.fit(val_ds, epochs=5)
    else:
        h = model.fit(train_ds, epochs=140)
    return h, model


def main(
    input_shape: Tuple[int, int],
    nb_classes: int = 9,
    output_dir: str = "",
    adaptive_based_on_val: bool = True,
) -> None:
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
    y_train = []
    for image_batch, labels_batch in train_ds:
        y_train.append(labels_batch)
    y_train = np.concatenate(y_train, axis=0)

    for image_batch, labels_batch in test_ds:
        y_test = np.array(labels_batch)

    # Build model
    model = cnn_architecture(
        input_shape, nb_classes, adaptive_based_on_val=adaptive_based_on_val
    )
    # Train model
    h, model = train_model(
        model, train_ds, val_ds, adaptive_based_on_val=adaptive_based_on_val
    )

    # Predict class probabilities as 2 => [0.1, 0, 0.9, 0, 0, 0, 0, 0, 0, 0]
    y_test_pred = model.predict(test_ds)
    Y_test_pred = np.argmax(y_test_pred, 1)  # Decode Predicted labels
    # Predict class probabilities as 2 => [0.1, 0, 0.9, 0, 0, 0, 0, 0, 0, 0]
    y_train_pred = model.predict(train_ds)
    Y_train_pred = np.argmax(y_train_pred, 1)  # Decode Predicted labels
    Y_test = np.argmax(y_test, 1)  # Decode Predicted labels
    Y_train = np.argmax(y_train, 1)  # Decode Predicted labels
    f1_score(Y_test, Y_test_pred, average="macro")
    f1_score(Y_train, Y_train_pred, average="macro")

    with open("data/le_name_mapping.json", "r") as f:
        mapping = json.load(f)
        le = LabelEncoder()
    mapping["classes"] = [mapping[str(int(i))] for i in range(9)]
    le.classes_ = np.array(mapping["classes"])

    plot_cm(Y_test, Y_test_pred, le, save=1, figname=f"{output_dir}/cm_test")
    plot_cm(Y_train, Y_train, le, save=1, figname=f"{output_dir}/cm_train")
    proportion_correct = f1_score(Y_test, Y_test_pred, average="macro")
    print("Test Accuracy: {}".format(proportion_correct))

    # Save model and weights
    model.save(f"{output_dir}/cnn_model.h5")

    # Calculate f1 and save classification report
    calcualte_classification_report(
        Y_train, Y_train_pred, Y_test, Y_test_pred, le, save=1, output_dir=output_dir
    )
    plt.close("all")
    return


def calculate_stats_multiple_run(output_dir: str, nb_runs: int) -> pd.DataFrame:
    """Calculate stats for multiple runs of the CNN model."""
    y_pred = []
    class_reports = []

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
    for image_batch, labels_batch in test_ds:
        y_test = np.array(labels_batch)
    Y_test = np.argmax(y_test, 1)

    for i in range(nb_runs):
        # read y_pred from file
        output_dir_ = f"{output_dir}/{i}/"
        y_pred_i = np.loadtxt(f"{output_dir_}/pred_test.txt")
        y_pred.append(y_pred_i)
        report_dict = classification_report(Y_test, y_pred_i, output_dict=True)
        class_reports.append(pd.DataFrame(report_dict))
    y_pred = np.array(y_pred)

    cnn_stats = pd.DataFrame(
        columns=[
            "macro avg mean",
            "macro avg std",
            "weighted avg mean",
            "weighted avg std",
        ],
        index=["recall", "f1-score"],
    )
    recalls_mavg = [class_reports[i].loc["recall", "macro avg"] for i in range(nb_runs)]
    recalls_wavg = [
        class_reports[i].loc["recall", "weighted avg"] for i in range(nb_runs)
    ]
    f1s_mavg = [class_reports[i].loc["f1-score", "macro avg"] for i in range(nb_runs)]
    f1s_wavg = [class_reports[i].loc["f1-score", "weighted avg"] for i in range(nb_runs)]
    cnn_stats.loc["recall", "macro avg mean"] = np.mean(recalls_mavg)
    cnn_stats.loc["recall", "macro avg std"] = np.std(recalls_mavg)
    cnn_stats.loc["f1-score", "macro avg mean"] = np.mean(f1s_mavg)
    cnn_stats.loc["f1-score", "macro avg std"] = np.std(f1s_mavg)
    cnn_stats.loc["recall", "weighted avg mean"] = np.mean(recalls_wavg)
    cnn_stats.loc["recall", "weighted avg std"] = np.std(recalls_wavg)
    cnn_stats.loc["f1-score", "weighted avg mean"] = np.mean(f1s_wavg)
    cnn_stats.loc["f1-score", "weighted avg std"] = np.std(f1s_wavg)

    return cnn_stats


if __name__ == "__main__":
    input_shape = (58, 58)
    nb_classes = 9
    # if False, then uses ada_delta with decay, else uses stopping and decay based on val_acc
    adaptive_based_on_val = True
    # If true, the model will be trained nb_runs times and the results will be saved in a folder
    # This is useful to assess randomness in the model which can come from GPU usage
    single_run = False
    nb_runs = 30

    # Create new folder with results, name is datetime
    if single_run:
        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"results/clf/cnn/{now_str}"
        os.mkdir(output_dir)

        main(input_shape, nb_classes, output_dir, adaptive_based_on_val)
    if 0:
        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"results/clf/cnn/{now_str}"
        os.mkdir(output_dir)
        for i in range(nb_runs):
            print(f"Run {i} of {nb_runs}")
            output_dir_ = f"{output_dir}/{i}"
            os.mkdir(output_dir_)
            main(input_shape, nb_classes, output_dir_, adaptive_based_on_val)

        # The cnn_stats from 30 runs are very similar to the ones from the run shown in the paper
        # output_dir = f"results/clf/cnn/2023-03-14_12-29-00"
        cnn_stats = calculate_stats_multiple_run(output_dir, nb_runs)
        print(cnn_stats)

    # Delete all
    print("Done")
