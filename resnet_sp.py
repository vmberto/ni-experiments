from layers.salt_and_pepper import RandomSaltAndPepper
from keras_cv.core import UniformFactorSampler
from models.resnet import ResNet50Model
from experiment import experiment
import multiprocessing
import os

INPUT_SHAPE = (72, 72, 3)


def run():
    execution_name = 'Salt&Pepper'

    factor = UniformFactorSampler(0, .5)

    data_augmentation_layers = [RandomSaltAndPepper(factor)]

    resnet = ResNet50Model(input_shape=INPUT_SHAPE, approach_name=execution_name)

    experiment(resnet, data_augmentation_layers)


if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()
    print("finished")
