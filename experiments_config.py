from models.resnet import ResNet50Model
from models.xception import XceptionModel
from layers.salt_and_pepper import RandomSaltAndPepper
from layers.default_aug import get_default_aug_layers
import keras.layers as layers
from keras_cv.core import UniformFactorSampler

EPOCHS = 100
BATCH_SIZE = 128
INPUT_SHAPE = (72, 72, 3)
KFOLD_N_SPLITS = 15
factor = UniformFactorSampler(0, .5)
DefaultAug = get_default_aug_layers()


CONFIGS = [
    {
        "approach_name": 'Baseline',
        "data_augmentation_layers": [],
        "model": ResNet50Model,
        "active": False,
    },
    {
        "approach_name": 'Salt&Pepper',
        "data_augmentation_layers": [RandomSaltAndPepper(factor)],
        "model": ResNet50Model,
        "active": False,
    },
    {
        "approach_name": 'Gaussian',
        "data_augmentation_layers": [layers.GaussianNoise(.1)],
        "model": ResNet50Model,
        "active": False,
    },
    {
        "approach_name": 'DefaultAug',
        "data_augmentation_layers": [
            DefaultAug
        ],
        "model": ResNet50Model,
        "active": True,
    },
    {
        "approach_name": 'DefaultAug+S&P',
        "data_augmentation_layers": [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
            RandomSaltAndPepper(factor),
        ],
        "model": ResNet50Model,
        "active": False,
    },
    {
        "approach_name": 'DefaultAug+Gaussian',
        "data_augmentation_layers": [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
            layers.GaussianNoise(.1),
        ],
        "model": ResNet50Model,
        "active": False,
    },
    {
        "approach_name": 'Baseline',
        "data_augmentation_layers": [],
        "model": XceptionModel,
        "active": False,
    },
    {
        "approach_name": 'Salt&Pepper',
        "data_augmentation_layers": [RandomSaltAndPepper(factor)],
        "model": XceptionModel,
        "active": False,
    },
    {
        "approach_name": 'Gaussian',
        "data_augmentation_layers": [layers.GaussianNoise(.1)],
        "model": XceptionModel,
        "active": False,
    },
    {
        "approach_name": 'DefaultAug',
        "data_augmentation_layers": [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        "model": XceptionModel,
        "active": False,
    },
    {
        "approach_name": 'DefaultAug+S&P',
        "data_augmentation_layers": [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
            RandomSaltAndPepper(factor),
        ],
        "model": XceptionModel,
        "active": False,
    },
    {
        "approach_name": 'DefaultAug+Gaussian',
        "data_augmentation_layers": [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
            layers.GaussianNoise(.1),
        ],
        "model": XceptionModel,
        "active": False,
    },
]
