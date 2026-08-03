import tensorflow as tf
from tensorflow.keras import layers


class VGGNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.features = tf.keras.Sequential(
            [
                layers.Conv2D(64, kernel_size=3, padding="same", activation="relu", input_shape=(224, 224, 3)),
                layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=2, strides=2),
                layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
                layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=2, strides=2),
                layers.Conv2D(256, kernel_size=3, padding="same", activation="relu"),
                layers.Conv2D(256, kernel_size=3, padding="same", activation="relu"),
                layers.Conv2D(256, kernel_size=3, padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=2, strides=2),
                layers.Conv2D(512, kernel_size=3, padding="same", activation="relu"),
                layers.Conv2D(512, kernel_size=3, padding="same", activation="relu"),
                layers.Conv2D(512, kernel_size=3, padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=2, strides=2),
                layers.Conv2D(512, kernel_size=3, padding="same", activation="relu"),
                layers.Conv2D(512, kernel_size=3, padding="same", activation="relu"),
                layers.Conv2D(512, kernel_size=3, padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=2, strides=2),
            ]
        )
        self.classifier = tf.keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(4096, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(4096, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(1000),
            ]
        )

    def call(self, inputs):
        x = self.features(inputs)
        return self.classifier(x)
