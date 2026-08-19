import tensorflow as tf
from tensorflow.keras import layers


class DepthwiseSeparableConv(tf.keras.layers.Layer):
    def __init__(self, out_channels, stride=1):
        super().__init__()
        self.dw_conv = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding="same", use_bias=False)
        self.dw_bn = layers.BatchNormalization()
        self.dw_relu = layers.ReLU()
        self.pw_conv = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.pw_bn = layers.BatchNormalization()
        self.pw_relu = layers.ReLU()

    def call(self, inputs, training=False):
        x = self.dw_relu(self.dw_bn(self.dw_conv(inputs), training=training))
        x = self.pw_relu(self.pw_bn(self.pw_conv(x), training=training))
        return x


class MobileNetV1(tf.keras.Model):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem_conv = layers.Conv2D(32, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.stem_bn = layers.BatchNormalization()
        self.stem_relu = layers.ReLU()

        _block_cfg = [
            (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
            (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2), (1024, 1),
        ]
        for i, (out_ch, stride) in enumerate(_block_cfg):
            setattr(self, f"dw_block_{i}", DepthwiseSeparableConv(out_ch, stride=stride))

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.stem_relu(self.stem_bn(self.stem_conv(inputs), training=training))
        for i in range(13):
            x = getattr(self, f"dw_block_{i}")(x, training=training)
        x = self.avgpool(x)
        return self.fc(x)
