import tensorflow as tf
from keras.applications import VGG16
from keras.models import Sequential
from models.experimental_model import ExperimentalModel


class VGG16Model(ExperimentalModel):

    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super().__init__(input_shape, num_classes)
        self.model_name = 'resnet50'

    def _build_model(self):
        vgg16 = VGG16(
            weights=None,
            include_top=True,
            input_shape=self.input_shape,
            classes=10,
        )
        model = Sequential(
            [
                tf.keras.layers.Input(self.input_shape),
                tf.keras.layers.Rescaling(scale=1.0 / 255, offset=-1),
                vgg16,
            ]
        )
        return model
