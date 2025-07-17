from dataset.cifar10dataset import Cifar10Dataset
from scripts.experiment_pipeline import experiment
from lib.consts import CIFAR10_CORRUPTIONS
import keras_cv
from layers.random_salt_and_pepper import RandomSaltAndPepper
from layers.custom_gaussian_noise import CustomGaussianNoise

from models.wideresnet2810 import WideResNet28_10Model
from models.cct import CCTModel
from models.resnet20 import ResNet20Model


KFOLD_N_SPLITS = 10
SALT_PEPPER_FACTOR = .3
GAUSSIAN_STDDEV = .2

INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 128
EPOCHS = 200
DATASET = Cifar10Dataset(INPUT_SHAPE, BATCH_SIZE)

RandAugment = keras_cv.layers.RandAugment(value_range=(0, 1), augmentations_per_image=3, magnitude=0.3, rate=1)
MODEL_ARCHITECTURES = [
    ResNet20Model,
    WideResNet28_10Model,
    CCTModel,
]

CONFIGS = [
    {
        "strategy_name": 'Baseline',
        "data_augmentation_layers": [],
        "curriculum_learning": False,
        "active": False,
    },
    {
        "strategy_name": 'RandAugment',
        "data_augmentation_layers": [RandAugment],
        "curriculum_learning": False,
        "active": False,
    },
    {
        "strategy_name": 'RandAugment+S&P',
        "data_augmentation_layers": [
            RandAugment,
            RandomSaltAndPepper(max_factor=SALT_PEPPER_FACTOR),
        ],
        "curriculum_learning": False,
        "active": False,
    },
    {
        "strategy_name": 'RandAugment+Gaussian',
        "data_augmentation_layers": [
            RandAugment,
            CustomGaussianNoise(max_stddev=GAUSSIAN_STDDEV),
        ],
        "curriculum_learning": False,
        "active": False,
    },
    {
        "strategy_name": 'Curriculum Learning',
        "data_augmentation_layers": [
            [RandAugment],
            [
                RandAugment,
                CustomGaussianNoise(max_stddev=GAUSSIAN_STDDEV / 2),
            ],
            [
                RandAugment,
                CustomGaussianNoise(max_stddev=GAUSSIAN_STDDEV),
            ],
        ],
        "es_patience_stages": [3, 5, 8],
        "curriculum_learning": True,
        "active": False,
    },
    {
        "strategy_name": 'Gaussian',
        "data_augmentation_layers": [
            CustomGaussianNoise(max_stddev=GAUSSIAN_STDDEV),
        ],
        "curriculum_learning": False,
        "active": True,
    },
    {
        "strategy_name": 'Salt&Pepper',
        "data_augmentation_layers": [
            RandomSaltAndPepper(max_factor=SALT_PEPPER_FACTOR),
        ],
        "curriculum_learning": False,
        "active": True,
    },
]


if __name__ == '__main__':
    experiment(DATASET, EPOCHS, KFOLD_N_SPLITS, CONFIGS, MODEL_ARCHITECTURES, CIFAR10_CORRUPTIONS)
