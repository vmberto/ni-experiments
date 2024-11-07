import tensorflow as tf
from keras.applications import EfficientNetB7
from keras.models import Sequential
from models.experimental_model import ExperimentalModel


class EfficientNetB7Model(ExperimentalModel):

    def __init__(self, input_shape=(32, 32, 3), num_classes=10, approach_name=''):
        super().__init__(input_shape, num_classes, approach_name)
        self.name = 'EfficientNetB7'

    def _build_model(self):
        resnet152 = EfficientNetB7(
            weights=None,
            include_top=True,
            input_shape=self.input_shape,
            classes=10,
        )
        model = Sequential(
            [
                tf.keras.layers.Input(self.input_shape),
                resnet152,
            ]
        )
        return model
