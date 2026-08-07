import tensorflow as tf
from tensorflow.keras import layers


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, filters, strides=1):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.downsample = None
        if strides != 1 or in_channels != filters:
            self.downsample = tf.keras.Sequential(
                [
                    layers.Conv2D(filters, 1, strides=strides, use_bias=False),
                    layers.BatchNormalization(),
                ]
            )

    def call(self, inputs):
        identity = inputs
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = layers.add([x, identity])
        x = self.relu(x)
        return x


class ResNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.features = tf.keras.Sequential(
            [
                layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPooling2D(pool_size=3, strides=2, padding="same"),
            ]
        )
        self.layer1 = tf.keras.Sequential([BasicBlock(64, 64), BasicBlock(64, 64)])
        self.layer2 = tf.keras.Sequential([BasicBlock(64, 128, strides=2), BasicBlock(128, 128)])
        self.layer3 = tf.keras.Sequential([BasicBlock(128, 256, strides=2), BasicBlock(256, 256)])
        self.layer4 = tf.keras.Sequential([BasicBlock(256, 512, strides=2), BasicBlock(512, 512)])
        self.classifier = tf.keras.Sequential([layers.GlobalAveragePooling2D(), layers.Dense(1000)])

    def call(self, inputs, training=None, mask=None):
        x = self.features(inputs, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        return self.classifier(x, training=training)
