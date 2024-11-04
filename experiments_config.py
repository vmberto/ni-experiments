from layers.default_aug import get_default_aug_layers as DefaultAug
from models.resnet import ResNet50Model
from models.xception import XceptionModel
from layers.salt_and_pepper import RandomSaltAndPepper
import keras.layers as layers
from keras_cv.core import UniformFactorSampler

EPOCHS = 100
BATCH_SIZE = 128
INPUT_SHAPE = (72, 72, 3)
KFOLD_N_SPLITS = 10
factor = UniformFactorSampler(0, .5)

CONFIGS = [
    {
        "approach_name": 'Baseline',
        "data_augmentation_layers": [],
        "model": ResNet50Model,
        "mixed": False,
        "active": False,
    },
    {
        "approach_name": 'Salt&Pepper',
        "data_augmentation_layers": [RandomSaltAndPepper(factor)],
        "model": ResNet50Model,
        "mixed": False,
        "active": False,
    },
    {
        "approach_name": 'Gaussian',
        "data_augmentation_layers": [layers.GaussianNoise(.1)],
        "model": ResNet50Model,
        "mixed": False,
        "active": False,
    },
    {
        "approach_name": 'DefaultAug',
        "data_augmentation_layers": DefaultAug(),
        "model": ResNet50Model,
        "mixed": False,
        "active": False,
    },
    {
        "approach_name": 'DefaultAug+S&P',
        "data_augmentation_layers": [
            *DefaultAug(),
            RandomSaltAndPepper(factor),
        ],
        "model": ResNet50Model,
        "mixed": False,
        "active": False,
    },
    {
        "approach_name": 'DefaultAug+Gaussian',
        "data_augmentation_layers": [
            *DefaultAug(),
            layers.GaussianNoise(.1),
        ],
        "model": ResNet50Model,
        "mixed": False,
        "active": False,
    },
    {
        "approach_name": 'Baseline',
        "data_augmentation_layers": [],
        "model": XceptionModel,
        "mixed": False,
        "active": False,
    },
    {
        "approach_name": 'Salt&Pepper',
        "data_augmentation_layers": [RandomSaltAndPepper(factor)],
        "model": XceptionModel,
        "mixed": False,
        "active": False,
    },
    {
        "approach_name": 'Gaussian',
        "data_augmentation_layers": [layers.GaussianNoise(.1)],
        "model": XceptionModel,
        "mixed": False,
        "active": False,
    },
    {
        "approach_name": 'DefaultAug',
        "data_augmentation_layers": DefaultAug(),
        "model": XceptionModel,
        "mixed": False,
        "active": False,
    },
    {
        "approach_name": 'DefaultAug+S&P',
        "data_augmentation_layers": [
            *DefaultAug(),
            RandomSaltAndPepper(factor),
        ],
        "model": XceptionModel,
        "mixed": False,
        "active": False,
    },
    {
        "approach_name": 'DefaultAug+Gaussian',
        "data_augmentation_layers": [
            *DefaultAug(),
            layers.GaussianNoise(.1),
        ],
        "model": XceptionModel,
        "mixed": False,
        "active": False,
    },
    {
        "approach_name": 'Mixed',
        "data_augmentation_layers": [
            None,
            DefaultAug(),
            [
                *DefaultAug(),
                layers.GaussianNoise(.1),
            ],
            [
                *DefaultAug(),
                RandomSaltAndPepper(factor),
            ],
        ],
        "model": ResNet50Model,
        "mixed": True,
        "active": True,
    },
    {
        "approach_name": 'Mixed',
        "data_augmentation_layers": [
            None,
            DefaultAug(),
            [
                *DefaultAug(),
                layers.GaussianNoise(.1),
            ],
            [
                *DefaultAug(),
                RandomSaltAndPepper(factor),
            ],
        ],
        "model": XceptionModel,
        "mixed": True,
        "active": True,
    },
]
