from dataset.cifar10dataset import Cifar10Dataset
from experiment import experiment
from lib.consts import CIFAR10_CORRUPTIONS
from layers.default_aug import get_default_aug_layers as DefaultAug
from models.resnet50 import ResNet50Model
from models.resnet101 import ResNet101Model
from layers.random_salt_and_pepper import RandomSaltAndPepper
from keras import layers

BATCH_SIZE = 128
INPUT_SHAPE = (72, 72, 3)
KFOLD_N_SPLITS = 10
SALT_PEPPER_FACTOR = .5
Dataset = Cifar10Dataset

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
        "data_augmentation_layers": [RandomSaltAndPepper(SALT_PEPPER_FACTOR)],
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
            RandomSaltAndPepper(SALT_PEPPER_FACTOR),
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
                RandomSaltAndPepper(SALT_PEPPER_FACTOR),
            ],
        ],
        "mixed": True,
        "active": False,
    },
]

experiment(Dataset, KFOLD_N_SPLITS, CONFIGS, MODEL_ARCHITECTURES, CIFAR10_CORRUPTIONS)
