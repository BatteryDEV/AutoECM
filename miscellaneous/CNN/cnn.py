import tensorflow as tf
from azure.storage.blob import BlobClient
import pathlib
print("Tensorflow version " + tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

import pandas as pd

import mlflow
import mlflow.tensorflow

import uuid


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
from os import path

def upload_datastorage(file, directory, container, cs):
        try:
            idx_file=0
            file_name=pathlib.Path(file).name
            blob=BlobClient.from_connection_string(
                conn_str=cs,
                container_name=container,
                blob_name=f"{directory}/{file_name}"
            )
        except Exception:
            raise

        try:
            with open(file, "rb") as data:
                blob.upload_blob(data, overwrite=True)
        except Exception:
            raise

here = os.path.dirname(os.path.realpath(__file__))

eis_data = pd.read_csv("train_data_images.csv", index_col=0)

# Add image path
# image_folders_path = path.join(path.dirname(here[0]),"images/")
image_folders_path = "images/"

eis_data["image_name"] = (
    image_folders_path + eis_data["Circuit"] + "/" + eis_data.index.astype(str) + ".jpg"
)
eis_data.head()

# Create category if from circuits
classes_dict = {}
for i, cl in enumerate(eis_data["Circuit"].unique()):
    classes_dict[cl] = i
eis_data["category_id"] = eis_data["Circuit"].apply(lambda x: classes_dict[x])
eis_data.head()

"""# Note to the readers
We were changing the model strucrture to make it more understandable but we ran out of time. This is the model that is working. We are also including the model that we were revising for the structure. Look at  **NotWorking_NiceStructure_GPU_CNN_Trainning.ipynb**
"""

# Create CNN model with additional features
# Warning!! This is not the best model structure
def create_cnn_model(num_classes: int, initial_lr: float = 0.01, features: list = []):

    feature_inputs = []

    for feature in features:
        feature_inputs.append(layers.Input(shape=(1), name=feature))

    image_inputs = layers.Input(shape=(224, 224, 3), name="image")
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        input_tensor=image_inputs,
        weights="imagenet",
        include_top=False,
    )

    # Freeze the pretrained weights
    base_model.trainable = True

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(32, activation="relu", kernel_regularizer="l2")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    if feature_inputs:
        x = layers.Concatenate(axis=1)([x] + feature_inputs)

    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(
        0.1,
    )(x)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.3
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(top_dropout_rate)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dropout(top_dropout_rate)(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="category_id")(x)

    inputs = []
    if feature_inputs:
        inputs = [image_inputs] + feature_inputs
    else:
        inputs = image_inputs

    # Compile
    model = tf.keras.Model(inputs, outputs, name="MobileNetV2")
    optimizer = tf.keras.optimizers.Adamax(learning_rate=initial_lr)
    top_k_metric = keras.metrics.TopKCategoricalAccuracy(k=3)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", top_k_metric],
    )
    return model


feature_cols = []

model = create_cnn_model(num_classes=9, initial_lr=0.02, features=feature_cols)
model.summary()

# Create tf.Data type for tpu
train_ds = tf.keras.utils.image_dataset_from_directory(
    "images",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=64,
    label_mode="categorical",
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "images",
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

    h = model.fit(train_ds, validation_data=val_ds, epochs=1000, callbacks=callbacks)

# checkpoint_filepath_top_k = "/models/cf0335d4-05a9-4b34-9b64-f38e1bb4a095_top_k.h5"
# model = tf.keras.models.load_model(checkpoint_filepath_top_k)
# for layer in model.layers:
#   if layer.name == "input_2":
#     layer._name = "image"
#   if layer.name == "pred":
#     layer._name = "category_id"
# model.summary()

full_ds = tf.keras.utils.image_dataset_from_directory(
    "images",
    validation_split=None,
    subset=None,
    seed=42,
    image_size=(224, 224),
    batch_size=64,
    label_mode="categorical",
    shuffle=False,
)

pred = model.predict(full_ds)

eis_data["pred"] = pred.argmax(axis=1)
eis_data.head()

eis_data.to_csv("results.csv")

upload_datastorage("results.csv", "dataset", "", "")
