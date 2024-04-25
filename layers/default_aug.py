import keras.layers as layers
from keras.models import Sequential


def get_default_aug_layers():
    return Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2)
    ], name="DefaultAug")