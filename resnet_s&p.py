import multiprocessing
from experiment import experiment
from keras.callbacks import EarlyStopping
from models.resnet import ResNet50Model
from dataset.cifar import get_cifar10, get_cifar10_corrupted
from layers.salt_and_pepper import RandomSaltAndPepper
import keras_cv.layers as layers
from keras_cv.core import NormalFactorSampler, UniformFactorSampler

BATCH_SIZE = 128
IMG_SIZE = 72
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
SEED = 42


def run():
    execution_name = 'Resnet S&P Augment'

    # @TODO: TESTAR COM UNIFORM FACTOR SAMPLER
    factor = UniformFactorSampler(0, .9)
    # factor = NormalFactorSampler(mean=0.3, stddev=0.1, min_value=.0, max_value=.9)

    data_augmentation_layers = [
        RandomSaltAndPepper(factor),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ]

    resnet = ResNet50Model(input_shape=INPUT_SHAPE)

    experiment(execution_name, resnet, data_augmentation_layers)


if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()
    print("finished")
