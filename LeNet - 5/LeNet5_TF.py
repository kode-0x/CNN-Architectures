import tensorflow as tf
from tensorflow.keras import layers


class LeNet5(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.features = tf.keras.Sequential(
            [
                layers.Conv2D(6, kernel_size=5, activation="tanh", input_shape=(32, 32, 1)),
                layers.AveragePooling2D(pool_size=2, strides=2),
                layers.Conv2D(16, kernel_size=5, activation="tanh"),
                layers.AveragePooling2D(pool_size=2, strides=2),
            ]
        )
        self.classifier = tf.keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(120, activation="tanh"),
                layers.Dense(84, activation="tanh"),
                layers.Dense(10),
            ]
        )

    def call(self, inputs):
        x = self.features(inputs)
        return self.classifier(x)
