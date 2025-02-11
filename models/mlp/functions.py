import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def train_model(model, trainX, trainY, epochs, metrics_collector):
    for epoch in range(epochs):
        start_time = time.time()
        metrics_collector.collect_system_metrics()

        history = model.fit(
            trainX,
            trainY,
            epochs=1,
            batch_size=1,
            verbose=2
        )

        batch_time = time.time() - start_time

        print(f"Batch time: {batch_time}")
        metrics_collector.collect_training_metrics(
            batch_time=batch_time,
            loss=history.history["loss"][-1],
            epoch=epoch
        )


def prepare_data():
    # load the dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_train = x_train / 255.0
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_test = x_test / 255.0

    num_classes = 10

    y_train = tf.one_hot(y_train.astype(np.int32), depth=num_classes)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=num_classes)

    return x_train, y_train, x_test, y_test


def create_model():

    model = Sequential(
        [
            Flatten(input_shape=(28, 28)),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
