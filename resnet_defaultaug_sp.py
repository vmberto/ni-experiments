from layers.salt_and_pepper import RandomSaltAndPepper
from keras_cv.core import UniformFactorSampler
from models.resnet import ResNet50Model
from experiment import experiment
import keras_cv.layers as layers
import multiprocessing
import os

INPUT_SHAPE = (72, 72, 3)


def run():
    execution_name = os.path.splitext(os.path.basename(__file__))[0]

    factor = UniformFactorSampler(0, .5)

    data_augmentation_layers = [
        RandomSaltAndPepper(factor),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ]

    resnet = ResNet50Model(input_shape=INPUT_SHAPE, execution_name=execution_name)

    experiment(execution_name, resnet, data_augmentation_layers)


if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()
    print("finished")
