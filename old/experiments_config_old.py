from layers.default_aug import get_default_aug_layers as DefaultAug
from models.resnet50 import ResNet50Model
from models.xception import XceptionModel
from old.salt_and_pepper import RandomSaltAndPepper
import keras.layers as layers
from keras_cv.core import UniformFactorSampler

EPOCHS = 100
BATCH_SIZE = 128
INPUT_SHAPE = (32, 32, 3)
KFOLD_N_SPLITS = 10
SALTPEPPER_FACTOR = UniformFactorSampler(0, .5)


CONFIGS = [
    {
        "strategy_name": 'Baseline',
        "data_augmentation_layers": [],
        "model": ResNet50Model,
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'Salt&Pepper',
        "data_augmentation_layers": [RandomSaltAndPepper(SALTPEPPER_FACTOR)],
        "model": ResNet50Model,
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'Gaussian',
        "data_augmentation_layers": [layers.GaussianNoise(.1)],
        "model": ResNet50Model,
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'DefaultAug',
        "data_augmentation_layers": DefaultAug(),
        "model": ResNet50Model,
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'DefaultAug+S&P',
        "data_augmentation_layers": [
            *DefaultAug(),
            RandomSaltAndPepper(SALTPEPPER_FACTOR),
        ],
        "model": ResNet50Model,
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'DefaultAug+Gaussian',
        "data_augmentation_layers": [
            *DefaultAug(),
            layers.GaussianNoise(.1),
        ],
        "model": ResNet50Model,
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'Mixed',
        "data_augmentation_layers": [
            None,
            DefaultAug(),
            [
                *DefaultAug(),
                layers.GaussianNoise(.1),
            ],
            [
                *DefaultAug(),
                RandomSaltAndPepper(SALTPEPPER_FACTOR),
            ],
        ],
        "model": ResNet50Model,
        "mixed": True,
        "active": False,
    },
    {
        "strategy_name": 'Baseline',
        "data_augmentation_layers": [],
        "model": XceptionModel,
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'Salt&Pepper',
        "data_augmentation_layers": [RandomSaltAndPepper(SALTPEPPER_FACTOR)],
        "model": XceptionModel,
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'Gaussian',
        "data_augmentation_layers": [layers.GaussianNoise(.1)],
        "model": XceptionModel,
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'DefaultAug',
        "data_augmentation_layers": DefaultAug(),
        "model": XceptionModel,
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'DefaultAug+S&P',
        "data_augmentation_layers": [
            *DefaultAug(),
            RandomSaltAndPepper(SALTPEPPER_FACTOR),
        ],
        "model": XceptionModel,
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'DefaultAug+Gaussian',
        "data_augmentation_layers": [
            *DefaultAug(),
            layers.GaussianNoise(.1),
        ],
        "model": XceptionModel,
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'Mixed',
        "data_augmentation_layers": [
            None,
            DefaultAug(),
            [
                *DefaultAug(),
                layers.GaussianNoise(.1),
            ],
            [
                *DefaultAug(),
                RandomSaltAndPepper(SALTPEPPER_FACTOR),
            ],
        ],
        "model": XceptionModel,
        "mixed": True,
        "active": False,
    },
]
