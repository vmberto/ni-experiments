from dataset.mimiidataset import MIMIIDataset
from lib.consts import MIMII_OOD_MACHINES
from mtsa import Hitachi
from mimii_experiment import experiment
from audiomentations import Compose, AddGaussianNoise, HighPassFilter, LowPassFilter, TimeMask


KFOLD_N_SPLITS = 10
SALT_PEPPER_FACTOR = .3
GAUSSIAN_STDDEV = .2

BATCH_SIZE = 128
EPOCHS = 50

DATASET = MIMIIDataset(BATCH_SIZE)
MODEL_ARCHITECTURES = [
    Hitachi,
]

RandAugment = Compose([
    HighPassFilter(min_cutoff_freq=100, max_cutoff_freq=300, p=0.2),
    LowPassFilter(min_cutoff_freq=4000, max_cutoff_freq=7000, p=0.2),
    TimeMask(min_band_part=0.01, max_band_part=0.05, fade=True, p=0.3),
])

CONFIGS = [
    {
        "strategy_name": 'Baseline',
        "data_augmentation_layers": None,
        "curriculum_learning": False,
        "active": True,
    },
    {
        "strategy_name": 'RandAugment',
        "data_augmentation_layers": RandAugment,
        "curriculum_learning": False,
        "active": True,
    },
    {
        "strategy_name": 'RandAugment+S&P',
        "data_augmentation_layers": [
        ],
        "curriculum_learning": False,
        "active": False,
    },
    {
        "strategy_name": 'RandAugment+Gaussian',
        "data_augmentation_layers": [
        ],
        "curriculum_learning": False,
        "active": False,
    },
    {
        "strategy_name": 'Curriculum Learning',
        "data_augmentation_layers": [
        ],
        "es_patience_stages": [3, 5, 8],
        "curriculum_learning": True,
        "active": False,
    },
]


if __name__ == '__main__':
    experiment(DATASET, EPOCHS, KFOLD_N_SPLITS, CONFIGS, MODEL_ARCHITECTURES, MIMII_OOD_MACHINES)
