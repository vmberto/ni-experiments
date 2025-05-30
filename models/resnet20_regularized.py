import tensorflow as tf
from keras import layers, models, regularizers
from models.experimental_model import ExperimentalModel


class ResNet20RegularizedModel(ExperimentalModel):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10, strategy_name='',
                 l2_reg=1e-4, dropout_rate=0.2):
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        super().__init__(input_shape, num_classes, strategy_name)
        self.name = 'ResNet20Regularized'

    def _residual_block(self, x, filters, stride=1):
        shortcut = x

        x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False,
                          kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False,
                          kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)

        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False,
                                     kernel_regularizer=regularizers.l2(self.l2_reg))(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        return x

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(16, 3, padding='same', use_bias=False,
                          kernel_regularizer=regularizers.l2(self.l2_reg))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # 3 stages of residual blocks: filters = 16, 32, 64
        for filters, stride in zip(
            [16]*3 + [32]+[32]*2 + [64]+[64]*2,
            [1]*3 + [2]+[1]*2 + [2]+[1]*2
        ):
            x = self._residual_block(x, filters, stride)

        x = layers.GlobalAveragePooling2D()(x)

        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate)(x)

        outputs = layers.Dense(self.num_classes, activation='softmax',
                               kernel_regularizer=regularizers.l2(self.l2_reg))(x)

        return models.Model(inputs, outputs)
