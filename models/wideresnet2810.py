import tensorflow as tf
from keras import layers, models
from models.experimental_model import ExperimentalModel


class WideResNet28_10Model(ExperimentalModel):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10, strategy_name=''):
        super().__init__(input_shape, num_classes, strategy_name)
        self.name = 'WideResNet28_10'

    def _residual_block(self, x, out_filters, stride, dropout_rate=0.0):
        in_filters = x.shape[-1]
        shortcut = x

        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(out_filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(x)

        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(out_filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)

        if stride != 1 or in_filters != out_filters:
            shortcut = layers.Conv2D(out_filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(shortcut)

        x = layers.add([x, shortcut])
        return x

    def _network_block(self, x, num_blocks, out_filters, stride, dropout_rate):
        for i in range(num_blocks):
            x = self._residual_block(x, out_filters, stride if i == 0 else 1, dropout_rate)
        return x

    def _build_model(self):
        depth = 28
        width = 10
        dropout_rate = 0.0

        assert (depth - 4) % 6 == 0
        num_blocks_per_stage = (depth - 4) // 6
        filters = [16, 16 * width, 32 * width, 64 * width]

        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(filters[0], kernel_size=3, padding='same', use_bias=False)(inputs)

        x = self._network_block(x, num_blocks_per_stage, filters[1], stride=1, dropout_rate=dropout_rate)
        x = self._network_block(x, num_blocks_per_stage, filters[2], stride=2, dropout_rate=dropout_rate)
        x = self._network_block(x, num_blocks_per_stage, filters[3], stride=2, dropout_rate=dropout_rate)

        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.GlobalAveragePooling2D()(x)
        # Use float32 for final layer for numerical stability in mixed precision
        outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        return model
