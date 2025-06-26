from audiomentations import AddGaussianNoise

from dataset.mimiidataset import MIMIIDataset
from layers.audio_randaugment import AudioRandAugment
from lib.consts import MIMII_OOD_MACHINES
from mtsa import Hitachi
from mimii_experiment import experiment


MIMII_MACHINES = [
    'valve',
    'pump',
    'slider',
    'fan',
]

KFOLD_N_SPLITS = 10
BATCH_SIZE = 512
EPOCHS = 600

DATASET = MIMIIDataset(BATCH_SIZE)
MODEL_ARCHITECTURES = [
    Hitachi,
]

RandAugment = AudioRandAugment(n_ops=2, magnitude_range=(0.35, 0.85))
RandAugmentAndGaussianNoise = AudioRandAugment(n_ops=2, magnitude_range=(0.35, 0.85), include_gaussian_noise=True)



CONFIGS = [
    {
        "strategy_name": 'Baseline',
        "data_augmentation_layers": None,
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
        "strategy_name": 'RandAugment+Gaussian',
        "data_augmentation_layers": [RandAugmentAndGaussianNoise],
        "curriculum_learning": False,
        "active": True,
    },
    # {
    #     "strategy_name": 'Curriculum Learning',
    #     "data_augmentation_layers": [
    #         RandAugment, [RandAugment, PartialGaussianNoise], [RandAugment, GaussianNoise]
    #     ],
    #     "es_patience_stages": [10, 10, 10],
    #     "curriculum_learning": True,
    #     "active": False,
    # },
]


if __name__ == '__main__':
    experiment(DATASET, EPOCHS, KFOLD_N_SPLITS, CONFIGS, MODEL_ARCHITECTURES, MIMII_MACHINES, MIMII_OOD_MACHINES)
