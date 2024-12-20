from dataset.agnewsdataset import AGNewsDataset
from experiment import experiment
from lib.consts import AGNEWS_CORRUPTIONS
from layers.default_text_aug_layer import get_default_aug_layers as DefaultAug
from models.lstm import LSTMModel
from layers.random_salt_and_pepper import RandomSaltAndPepper
from keras import layers


EPOCHS = 100
BATCH_SIZE = 128
KFOLD_N_SPLITS = 10
SALT_PEPPER_FACTOR = .3
Dataset = AGNewsDataset


MODEL_ARCHITECTURES = [
    LSTMModel,
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
        "active": True,
    },
    {
        "strategy_name": 'Gaussian',
        "data_augmentation_layers": [layers.GaussianNoise(.1)],
        "mixed": False,
        "active": True,
    },
    {
        "strategy_name": 'DefaultAug',
        "data_augmentation_layers": [DefaultAug()],
        "mixed": False,
        "active": True,
    },
    {
        "strategy_name": 'DefaultAug+S&P',
        "data_augmentation_layers": [DefaultAug(), RandomSaltAndPepper(SALT_PEPPER_FACTOR)],
        "mixed": False,
        "active": True,
    },
    {
        "strategy_name": 'DefaultAug+Gaussian',
        "data_augmentation_layers": [
            DefaultAug(),
            layers.GaussianNoise(.1),
        ],
        "mixed": False,
        "active": True,
    },
    # {
    #     "strategy_name": 'Mixed',
    #     "data_augmentation_layers": [
    #         None,
    #         DefaultAug(),
    #         [
    #             *DefaultAug(),
    #             layers.GaussianNoise(.1),
    #         ],
    #         [
    #             *DefaultAug(),
    #             RandomSaltAndPepper(SALT_PEPPER_FACTOR),
    #         ],
    #     ],
    #     "mixed": False,
    #     "active": True,
    # },
]

experiment(Dataset, KFOLD_N_SPLITS, CONFIGS, MODEL_ARCHITECTURES, AGNEWS_CORRUPTIONS)
