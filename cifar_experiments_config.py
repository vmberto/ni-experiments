from dataset.cifar10dataset import Cifar10Dataset
from experiment import experiment
from lib.consts import CIFAR10_CORRUPTIONS
import keras_cv
from models.resnet18 import ResNet18Model
from layers.random_salt_and_pepper import RandomSaltAndPepper
from layers.custom_gaussian_noise import CustomGaussianNoise
from keras import layers
from models.resnet50 import ResNet50Model


KFOLD_N_SPLITS = 10
SALT_PEPPER_FACTOR = .3
GAUSSIAN_STDDEV = .2
Dataset = Cifar10Dataset
RandAugment = keras_cv.layers.RandAugment(value_range=(0, 1), augmentations_per_image=3, magnitude=0.3, rate=1)
MODEL_ARCHITECTURES = [
    ResNet50Model,
    # ResNet18Model,
]


CONFIGS = [
    {
        "strategy_name": 'Baseline',
        "data_augmentation_layers": [],
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'Salt&Pepper',
        "data_augmentation_layers": [RandomSaltAndPepper(max_factor=SALT_PEPPER_FACTOR)],
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'Gaussian',
        "data_augmentation_layers": [CustomGaussianNoise(max_stddev=GAUSSIAN_STDDEV)],
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'RandAugment',
        "data_augmentation_layers": [RandAugment],
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'RandAugment+S&P',
        "data_augmentation_layers": [
            RandAugment,
            RandomSaltAndPepper(max_factor=SALT_PEPPER_FACTOR),
        ],
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'RandAugment+Gaussian',
        "data_augmentation_layers": [
            RandAugment,
            CustomGaussianNoise(max_stddev=GAUSSIAN_STDDEV),
        ],
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'Mixed',
        "data_augmentation_layers": [
            [RandAugment],
            [
                RandAugment,
                CustomGaussianNoise(max_stddev=GAUSSIAN_STDDEV),
            ],
            [
                RandAugment,
                RandomSaltAndPepper(max_factor=SALT_PEPPER_FACTOR),
            ],
        ],
        "mixed": True,
        "active": False,
    },



    # Finetuning
    {
        "strategy_name": 'RandAugment+S&P/Fixed.2',
        "data_augmentation_layers": [
            RandAugment,
            RandomSaltAndPepper(factor=.2),
        ],
        "mixed": False,
        "active": False,
    },
    {
        "strategy_name": 'RandAugment+S&P/Fixed.3',
        "data_augmentation_layers": [
            RandAugment,
            RandomSaltAndPepper(factor=.3),
        ],
        "mixed": False,
        "active": True,
    },
    {
        "strategy_name": 'RandAugment+S&P/Fixed.4',
        "data_augmentation_layers": [
            RandAugment,
            RandomSaltAndPepper(factor=.4),
        ],
        "mixed": False,
        "active": True,
    },
    {
        "strategy_name": 'RandAugment+S&P/Distribution.2',
        "data_augmentation_layers": [
            RandAugment,
            RandomSaltAndPepper(max_factor=.2),
        ],
        "mixed": False,
        "active": True,
    },
    {
        "strategy_name": 'RandAugment+S&P/Distribution.3',
        "data_augmentation_layers": [
            RandAugment,
            RandomSaltAndPepper(max_factor=.3),
        ],
        "mixed": False,
        "active": True,
    },
    {
        "strategy_name": 'RandAugment+S&P/Distribution.4',
        "data_augmentation_layers": [
            RandAugment,
            RandomSaltAndPepper(max_factor=.4),
        ],
        "mixed": False,
        "active": True,
    },





    # Finetuning Gaussian
    {
        "strategy_name": 'RandAugment+Gaussian/Fixed.2',
        "data_augmentation_layers": [
            RandAugment,
            layers.GaussianNoise(.2),
        ],
        "mixed": False,
        "active": True,
    },
    {
        "strategy_name": 'RandAugment+Gaussian/Fixed.3',
        "data_augmentation_layers": [
            RandAugment,
            layers.GaussianNoise(.3),
        ],
        "mixed": False,
        "active": True,
    },
    {
        "strategy_name": 'RandAugment+Gaussian/Fixed.4',
        "data_augmentation_layers": [
            RandAugment,
            layers.GaussianNoise(.4),
        ],
        "mixed": False,
        "active": True,
    },
    {
        "strategy_name": 'RandAugment+Gaussian/Distribution.2',
        "data_augmentation_layers": [
            RandAugment,
            CustomGaussianNoise(max_stddev=.2),
        ],
        "mixed": False,
        "active": True,
    },
    {
        "strategy_name": 'RandAugment+Gaussian/Distribution.3',
        "data_augmentation_layers": [
            RandAugment,
            CustomGaussianNoise(max_stddev=.3),
        ],
        "mixed": False,
        "active": True,
    },
    {
        "strategy_name": 'RandAugment+Gaussian/Distribution.4',
        "data_augmentation_layers": [
            RandAugment,
            CustomGaussianNoise(max_stddev=.4),
        ],
        "mixed": False,
        "active": True,
    },

]

experiment(Dataset, KFOLD_N_SPLITS, CONFIGS, MODEL_ARCHITECTURES, CIFAR10_CORRUPTIONS)
