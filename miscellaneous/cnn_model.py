import tensorflow as tf
import pathlib
print("Tensorflow version " + tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
import pandas as pd
import mlflow
import mlflow.tensorflow
import uuid

import tensorflow as tf
from tensorflow import keras

from cnn_architectures import baseline_cnn_model
from cnn_architectures import transferlearn_cnn_model
from eis_preprocessing import preprocess_data
from eis_preprocessing import eis_label_encoder

import os
from os import path


# Read all data
here = os.path.dirname(os.path.realpath(__file__))
# df for labels
d_path = './data/'
df = preprocess_data(d_path + "train_data_newspl.csv")
df_test = preprocess_data(d_path + "test_data_newspl.csv")

# Label encoder
le, mapping = eis_label_encoder(le_f = 'models/labels.json')

# Add image path
# image_folders_path = path.join(path.dirname(here[0]),"images/")
image_folders_path = "images/train/"

eis_data = pd.DataFrame()
eis_data["Circuit"] = df["Circuit"]

eis_data["image_name"] = (
    image_folders_path + eis_data["Circuit"] + "/" + eis_data.index.astype(str) + ".png"
)

# Create category if from circuits
eis_data["category_id"] = le.transform(eis_data["Circuit"])
eis_data.head()

feature_cols = []

model = transferlearn_cnn_model(num_classes=9, initial_lr=0.02, features=feature_cols)
model.summary()

# Create tf.Data type for tpu
train_ds = tf.keras.utils.image_dataset_from_directory(
    "images/train",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=64,
    label_mode="categorical",
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "images/train",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=64,
    label_mode="categorical",
    shuffle=False,
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()

# Train model
with mlflow.start_run():
    # Set checkpoint for best model
    model_name = str(uuid.uuid4())
    checkpoint_filepath_acc = path.join(
        path.dirname(here[0]), "models", f"{model_name}_acc.h5"
    )
    checkpoint_filepath_top_k = path.join(
        path.dirname(here[0]), "models", f"{model_name}_top_k.h5"
    )
    model_checkpoint_acc = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_acc,
        save_weights_only=False,
        monitor="val_accuracy",
        mode="auto",
        save_best_only=True,
    )

    model_checkpoint_top_k = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_top_k,
        save_weights_only=False,
        monitor="val_top_k_categorical_accuracy",
        mode="auto",
        save_best_only=True,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.001, patience=100
        ),
        keras.callbacks.ReduceLROnPlateau(factor=0.01, min_delta=0.001, patience=20),
        model_checkpoint_acc,
        model_checkpoint_top_k,
    ]

    # step_size_train = train_gen.n//train_gen.batch_size
    # step_size_val = val_gen.n//val_gen.batch_size

    h = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

# checkpoint_filepath_top_k = "/models/cf0335d4-05a9-4b34-9b64-f38e1bb4a095_top_k.h5"
# model = tf.keras.models.load_model(checkpoint_filepath_top_k)
# for layer in model.layers:
#   if layer.name == "input_2":
#     layer._name = "image"
#   if layer.name == "pred":
#     layer._name = "category_id"
# model.summary()

full_ds = tf.keras.utils.image_dataset_from_directory(
    "images/test",
    validation_split=None,
    subset=None,
    seed=42,
    image_size=(224, 224),
    batch_size=64,
    label_mode="categorical",
    shuffle=False,
)

pred = model.predict(full_ds)

class_pred = pred.argmax(axis=1)

