import tensorflow as tf
from keras.applications import ResNet101
from keras.models import Sequential
from models.experimental_model import ExperimentalModel


class ResNet101Model(ExperimentalModel):

    def __init__(self, input_shape=(32, 32, 3), num_classes=10, strategy_name=''):
        super().__init__(input_shape, num_classes, strategy_name)
        self.name = 'ResNet101'

    def _build_model(self):
        resnet50 = ResNet101(
            weights=None,
            include_top=True,
            input_shape=self.input_shape,
            classes=10,
        )
        model = Sequential(
            [
                tf.keras.layers.Input(self.input_shape),
                resnet50,
            ]
        )
        return model