from models.xception import XceptionModel
from experiment import experiment
import keras_cv.layers as layers
import multiprocessing
import os

INPUT_SHAPE = (72, 72, 3)


def run():
    execution_name = 'DefaultAug'

    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ]

    xception = XceptionModel(input_shape=INPUT_SHAPE, approach_name=execution_name)

    experiment(xception, data_augmentation_layers)


if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()
    print("finished")
