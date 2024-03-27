import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Sequential
from .experimental_model import ExperimentalModel


class ResNet50Model(ExperimentalModel):

    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        super().__init__(input_shape, num_classes)

    def _build_model(self):
        model = Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax'),
        ])
        return model
