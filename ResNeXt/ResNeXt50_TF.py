import tensorflow as tf
from tensorflow.keras import layers


class ResNeXtBottleneck(tf.keras.layers.Layer):
    expansion = 2

    def __init__(self, in_channels, out_channels, cardinality=32, stride=1):
        super().__init__()
        width = out_channels * self.expansion
        self.conv1 = layers.Conv2D(width, 1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(width, 3, strides=stride, padding="same",
                                   groups=cardinality, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(out_channels * 4, 1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()

        self.downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = tf.keras.Sequential([
                layers.Conv2D(out_channels * 4, 1, strides=stride, use_bias=False),
                layers.BatchNormalization(),
            ])

    def call(self, inputs, training=False):
        identity = inputs
        x = self.relu(self.bn1(self.conv1(inputs), training=training))
        x = self.relu(self.bn2(self.conv2(x), training=training))
        x = self.bn3(self.conv3(x), training=training)
        if self.downsample is not None:
            identity = self.downsample(inputs, training=training)
        x = self.relu(x + identity)
        return x


def make_layer(in_channels, out_channels, num_blocks, cardinality=32, stride=1):
    block_list = [ResNeXtBottleneck(in_channels, out_channels, cardinality, stride)]
    for _ in range(1, num_blocks):
        block_list.append(ResNeXtBottleneck(out_channels * 4, out_channels, cardinality, stride=1))
    return block_list


class ResNeXt50(tf.keras.Model):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = tf.keras.Sequential([
            layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2, padding="same"),
        ])
        _stage_cfg = [(64, 64, 3, 1), (256, 128, 4, 2), (512, 256, 6, 2), (1024, 512, 3, 2)]
        for stage_idx, (in_ch, out_ch, num_blocks, stride) in enumerate(_stage_cfg):
            for block_idx, block in enumerate(make_layer(in_ch, out_ch, num_blocks, stride=stride)):
                setattr(self, f"stage{stage_idx}_block{block_idx}", block)
        self._stage_block_counts = [3, 4, 6, 3]

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def _forward_blocks(self, x, stage_idx, training):
        for block_idx in range(self._stage_block_counts[stage_idx]):
            x = getattr(self, f"stage{stage_idx}_block{block_idx}")(x, training=training)
        return x

    def call(self, inputs, training=False):
        x = self.stem(inputs, training=training)
        x = self._forward_blocks(x, 0, training)
        x = self._forward_blocks(x, 1, training)
        x = self._forward_blocks(x, 2, training)
        x = self._forward_blocks(x, 3, training)
        x = self.avgpool(x)
        return self.fc(x)
