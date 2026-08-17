import tensorflow as tf
from tensorflow.keras import layers


class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super().__init__()
        self.b1_conv = layers.Conv2D(out_1x1, 1, padding="same", use_bias=False)
        self.b1_bn = layers.BatchNormalization()

        self.b2_red = layers.Conv2D(red_3x3, 1, padding="same", use_bias=False)
        self.b2_red_bn = layers.BatchNormalization()
        self.b2_conv = layers.Conv2D(out_3x3, 3, padding="same", use_bias=False)
        self.b2_bn = layers.BatchNormalization()

        self.b3_red = layers.Conv2D(red_5x5, 1, padding="same", use_bias=False)
        self.b3_red_bn = layers.BatchNormalization()
        self.b3_conv = layers.Conv2D(out_5x5, 5, padding="same", use_bias=False)
        self.b3_bn = layers.BatchNormalization()

        self.b4_pool = layers.MaxPooling2D(pool_size=3, strides=1, padding="same")
        self.b4_conv = layers.Conv2D(out_pool, 1, padding="same", use_bias=False)
        self.b4_bn = layers.BatchNormalization()

        self.relu = layers.ReLU()

    def call(self, inputs, training=False):
        b1 = self.relu(self.b1_bn(self.b1_conv(inputs), training=training))

        b2 = self.relu(self.b2_red_bn(self.b2_red(inputs), training=training))
        b2 = self.relu(self.b2_bn(self.b2_conv(b2), training=training))

        b3 = self.relu(self.b3_red_bn(self.b3_red(inputs), training=training))
        b3 = self.relu(self.b3_bn(self.b3_conv(b3), training=training))

        b4 = self.b4_pool(inputs)
        b4 = self.relu(self.b4_bn(self.b4_conv(b4), training=training))

        return tf.concat([b1, b2, b3, b4], axis=-1)


class GoogLeNet(tf.keras.Model):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem_conv1 = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)
        self.stem_bn1 = layers.BatchNormalization()
        self.stem_pool1 = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")
        self.stem_conv2 = layers.Conv2D(64, 1, padding="same", use_bias=False)
        self.stem_bn2 = layers.BatchNormalization()
        self.stem_conv3 = layers.Conv2D(192, 3, padding="same", use_bias=False)
        self.stem_bn3 = layers.BatchNormalization()
        self.stem_pool2 = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")
        self.relu = layers.ReLU()

        self.inception3a = InceptionModule(64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(128, 128, 192, 32, 96, 64)
        self.maxpool3 = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")

        self.inception4a = InceptionModule(192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(256, 160, 320, 32, 128, 128)
        self.maxpool4 = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")

        self.inception5a = InceptionModule(256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(384, 192, 384, 48, 128, 128)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.4)
        self.fc = layers.Dense(num_classes)

    def _stem(self, x, training):
        x = self.relu(self.stem_bn1(self.stem_conv1(x), training=training))
        x = self.stem_pool1(x)
        x = self.relu(self.stem_bn2(self.stem_conv2(x), training=training))
        x = self.relu(self.stem_bn3(self.stem_conv3(x), training=training))
        x = self.stem_pool2(x)
        return x

    def call(self, inputs, training=False):
        x = self._stem(inputs, training)

        x = self.inception3a(x, training=training)
        x = self.inception3b(x, training=training)
        x = self.maxpool3(x)

        x = self.inception4a(x, training=training)
        x = self.inception4b(x, training=training)
        x = self.inception4c(x, training=training)
        x = self.inception4d(x, training=training)
        x = self.inception4e(x, training=training)
        x = self.maxpool4(x)

        x = self.inception5a(x, training=training)
        x = self.inception5b(x, training=training)

        x = self.avgpool(x)
        x = self.dropout(x, training=training)
        x = self.fc(x)
        return x
