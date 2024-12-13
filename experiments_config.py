from layers.default_aug import get_default_aug_layers as DefaultAug
from models.resnet50 import ResNet50Model
from models.resnet101 import ResNet101Model
from layers.salt_and_pepper import RandomSaltAndPepper
from keras import layers
from keras_cv import core

EPOCHS = 100
BATCH_SIZE = 128
INPUT_SHAPE = (72, 72, 3)
KFOLD_N_SPLITS = 10
SALTPEPPER_FACTOR = core.UniformFactorSampler(0, .5)


MODEL_ARCHITECTURES = [
    ResNet50Model,
    ResNet101Model,
]


CONFIGS = [
    {
        "strategy_name": 'Baseline',
        "data_augmentation_layers": [],
        "mixed": False,
        "active": True,
    },
    {
        "strategy_name": 'Salt&Pepper',
        "data_augmentation_layers": [RandomSaltAndPepper(SALTPEPPER_FACTOR)],
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'Gaussian',
        "data_augmentation_layers": [layers.GaussianNoise(.1)],
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'DefaultAug',
        "data_augmentation_layers": DefaultAug(),
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'DefaultAug+S&P',
        "data_augmentation_layers": [
            *DefaultAug(),
            RandomSaltAndPepper(SALTPEPPER_FACTOR),
        ],
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'DefaultAug+Gaussian',
        "data_augmentation_layers": [
            *DefaultAug(),
            layers.GaussianNoise(.1),
        ],
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
        "mixed": True,
        "active": False,
    },
]
