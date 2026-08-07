import tensorflow as tf
from tensorflow.keras import layers


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.conv2 = layers.Conv2D(growth_rate, 3, padding="same", use_bias=False)

    def call(self, inputs, training=False):
        x = self.bn1(inputs, training=training)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.conv2(x)
        return x


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.dense_layers = [DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)]

    def call(self, inputs, training=False):
        features = [inputs]
        x = inputs
        for layer in self.dense_layers:
            out = layer(x, training=training)
            features.append(out)
            x = tf.concat(features, axis=-1)
        return x


class Transition(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv = layers.Conv2D(out_channels, 1, use_bias=False)
        self.pool = layers.AveragePooling2D(pool_size=2, strides=2)

    def call(self, inputs, training=False):
        x = self.bn(inputs, training=training)
        x = self.relu(x)
        x = self.conv(x)
        return self.pool(x)


class DenseNet(tf.keras.Model):
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
        self.block1 = DenseBlock(64, 32, 6)
        self.transition1 = Transition(64 + 6 * 32, 128)
        self.block2 = DenseBlock(128, 32, 12)
        self.transition2 = Transition(128 + 12 * 32, 256)
        self.block3 = DenseBlock(256, 32, 24)
        self.transition3 = Transition(256 + 24 * 32, 512)
        self.block4 = DenseBlock(512, 32, 16)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.classifier = tf.keras.Sequential([layers.GlobalAveragePooling2D(), layers.Dense(1000)])

    def call(self, inputs, training=False, mask=None):
        x = self.features(inputs, training=training)
        x = self.block1(x, training=training)
        x = self.transition1(x, training=training)
        x = self.block2(x, training=training)
        x = self.transition2(x, training=training)
        x = self.block3(x, training=training)
        x = self.transition3(x, training=training)
        x = self.block4(x, training=training)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return self.classifier(x, training=training)
