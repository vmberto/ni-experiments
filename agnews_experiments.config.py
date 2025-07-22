from dataset.agnewsdataset import AGNewsDataset
from scripts.experiment_pipeline import experiment
from layers.text_randaugment import get_text_randaugment_layers
from lib.consts import AGNEWS_CORRUPTIONS
from layers.random_salt_and_pepper import RandomSaltAndPepper
from layers.custom_gaussian_noise import CustomGaussianNoise
from models.cnn_text import CNNTextModel
from models.bilstm import BiLSTMModel
from models.tiny_transformer import TinyTransformerModel


EPOCHS = 200
KFOLD_N_SPLITS = 10
SALT_PEPPER_FACTOR = .2
GAUSSIAN_STDDEV = .2

MAX_SEQUENCE_LENGTH=128
VOCAB_SIZE=20000
BATCH_SIZE=128

Dataset = AGNewsDataset(max_sequence_length=MAX_SEQUENCE_LENGTH, vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE)
RandAugment = get_text_randaugment_layers()


MODEL_ARCHITECTURES = [
    CNNTextModel,
    BiLSTMModel,
    TinyTransformerModel
]


CONFIGS = [
    {
        "strategy_name": 'Baseline',
        "data_augmentation_layers": [],
        "curriculum_learning": False,
        "active": True,
    },
    {
        "strategy_name": 'Salt&Pepper',
        "data_augmentation_layers": [RandomSaltAndPepper(SALT_PEPPER_FACTOR)],
        "curriculum_learning": False,
        "active": True,
    },
    {
        "strategy_name": 'Gaussian',
        "data_augmentation_layers": [CustomGaussianNoise(GAUSSIAN_STDDEV)],
        "curriculum_learning": False,
        "active": True,
    },
    {
        "strategy_name": 'RandAugment',
        "data_augmentation_layers": [RandAugment],
        "curriculum_learning": False,
        "active": True,
    },
    {
        "strategy_name": 'RandAugment+S&P',
        "data_augmentation_layers": [RandAugment, RandomSaltAndPepper(SALT_PEPPER_FACTOR)],
        "curriculum_learning": False,
        "active": True,
    },
    {
        "strategy_name": 'RandAugment+Gaussian',
        "data_augmentation_layers": [
            RandAugment,
            CustomGaussianNoise(GAUSSIAN_STDDEV),
        ],
        "curriculum_learning": False,
        "active": True,
    },
    {
        "strategy_name": 'Curriculum Learning',
        "data_augmentation_layers": [
            [],
            [RandomSaltAndPepper(.1)],
            [RandAugment, RandomSaltAndPepper(SALT_PEPPER_FACTOR)],
        ],
        "es_patience_stages": [5, 8, 10],
        "curriculum_learning": True,
        "active": True,
    },
]

experiment(Dataset, EPOCHS, KFOLD_N_SPLITS, CONFIGS, MODEL_ARCHITECTURES, AGNEWS_CORRUPTIONS)
