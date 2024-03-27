import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Sequential
from .experimental_model import ExperimentalModel


class ResNet50Model(ExperimentalModel):

    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super().__init__(input_shape, num_classes)

    def _build_model(self):
        model = Sequential([
            ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(self.num_classes, activation='softmax'),
        ])
        return model
