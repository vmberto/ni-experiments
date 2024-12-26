import tensorflow as tf
from keras.applications import Xception
from keras.models import Sequential
from models.experimental_model import ExperimentalModel


class XceptionModel(ExperimentalModel):

    def __init__(self, input_shape=(32, 32, 3), num_classes=10, strategy_name=''):
        super().__init__(input_shape, num_classes, strategy_name)
        self.name = 'Xception'

    def _build_model(self):
        xception = Xception(
            weights=None,
            include_top=True,
            input_shape=self.input_shape,
            classes=10,
        )
        model = Sequential(
            [
                tf.keras.layers.Input(self.input_shape),
                xception,
            ]
        )
        return model
