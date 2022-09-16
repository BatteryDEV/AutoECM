import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def baseline_cnn_model(num_classes, initial_lr, features):
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
    return model

# Create CNN model with additional features
def transferlearn_cnn_model(num_classes: int, initial_lr: float = 0.01, features: list = []):

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