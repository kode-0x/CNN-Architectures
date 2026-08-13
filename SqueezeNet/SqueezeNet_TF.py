import tensorflow as tf
from tensorflow.keras import layers


class FireModule(tf.keras.layers.Layer):
    def __init__(self, squeeze, expand_1x1, expand_3x3):
        super().__init__()
        self.squeeze = layers.Conv2D(squeeze, kernel_size=1, activation="relu")
        self.expand_1x1 = layers.Conv2D(expand_1x1, kernel_size=1, activation="relu")
        self.expand_3x3 = layers.Conv2D(expand_3x3, kernel_size=3, padding="same", activation="relu")

    def call(self, inputs):
        x = self.squeeze(inputs)
        return tf.concat([self.expand_1x1(x), self.expand_3x3(x)], axis=-1)


class SqueezeNet(tf.keras.Model):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = tf.keras.Sequential(
            [
                layers.Conv2D(96, kernel_size=7, strides=2, activation="relu"),
                layers.MaxPooling2D(pool_size=3, strides=2),
                FireModule(16, 64, 64),
                FireModule(16, 64, 64),
                FireModule(32, 128, 128),
                layers.MaxPooling2D(pool_size=3, strides=2),
                FireModule(32, 128, 128),
                FireModule(48, 192, 192),
                FireModule(48, 192, 192),
                FireModule(64, 256, 256),
                layers.MaxPooling2D(pool_size=3, strides=2),
                FireModule(64, 256, 256),
                layers.Dropout(0.5),
                layers.Conv2D(num_classes, kernel_size=1, activation="relu"),
            ]
        )
        self.avgpool = layers.GlobalAveragePooling2D()

    def call(self, inputs):
        x = self.features(inputs)
        return self.avgpool(x)
