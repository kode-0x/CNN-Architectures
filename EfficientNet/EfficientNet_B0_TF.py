import tensorflow as tf
from tensorflow.keras import layers


class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, in_channels, reduced_channels):
        super().__init__()
        self.se_reduce = layers.Dense(reduced_channels, activation="swish")
        self.se_expand = layers.Dense(in_channels, activation="sigmoid")
        self.gap = layers.GlobalAveragePooling2D()

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.se_reduce(x)
        x = self.se_expand(x)
        x = tf.reshape(x, (-1, 1, 1, tf.shape(x)[-1]))
        return inputs * x


class MBConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        mid_channels = in_channels * expand_ratio
        reduced_channels = max(1, int(in_channels * se_ratio))

        self.expand_conv = None
        if expand_ratio != 1:
            self.expand_conv = layers.Conv2D(mid_channels, 1, use_bias=False)
            self.expand_bn = layers.BatchNormalization()

        self.dw_conv = layers.DepthwiseConv2D(kernel_size, strides=stride,
                                               padding="same", use_bias=False)
        self.dw_bn = layers.BatchNormalization()
        self.se = SqueezeExcitation(mid_channels, reduced_channels)
        self.project_conv = layers.Conv2D(out_channels, 1, use_bias=False)
        self.project_bn = layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = inputs
        if self.expand_conv is not None:
            x = tf.nn.swish(self.expand_bn(self.expand_conv(x), training=training))
        x = tf.nn.swish(self.dw_bn(self.dw_conv(x), training=training))
        x = self.se(x)
        x = self.project_bn(self.project_conv(x), training=training)
        if self.use_residual:
            x = x + inputs
        return x


class EfficientNetB0(tf.keras.Model):
    _STAGE_CFG = [
        (1,  16, 1, 3, 1),
        (6,  24, 2, 3, 2),
        (6,  40, 2, 5, 2),
        (6,  80, 3, 3, 2),
        (6, 112, 3, 5, 1),
        (6, 192, 4, 5, 2),
        (6, 320, 1, 3, 1),
    ]

    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem_conv = layers.Conv2D(32, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.stem_bn = layers.BatchNormalization()

        in_channels = 32
        for stage_idx, (expand_ratio, out_channels, num_layers, kernel_size, stride) in enumerate(self._STAGE_CFG):
            for layer_idx in range(num_layers):
                block = MBConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride if layer_idx == 0 else 1,
                    expand_ratio=expand_ratio,
                )
                setattr(self, f"mbconv_s{stage_idx}_l{layer_idx}", block)
                in_channels = out_channels
        self._stage_cfg_snapshot = self._STAGE_CFG

        self.head_conv = layers.Conv2D(1280, kernel_size=1, use_bias=False)
        self.head_bn = layers.BatchNormalization()
        self.avgpool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.2)
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = tf.nn.swish(self.stem_bn(self.stem_conv(inputs), training=training))
        for stage_idx, (_, _, num_layers, _, _) in enumerate(self._STAGE_CFG):
            for layer_idx in range(num_layers):
                block = getattr(self, f"mbconv_s{stage_idx}_l{layer_idx}")
                x = block(x, training=training)
        x = tf.nn.swish(self.head_bn(self.head_conv(x), training=training))
        x = self.avgpool(x)
        x = self.dropout(x, training=training)
        return self.fc(x)
