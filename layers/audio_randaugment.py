from random import sample
from audiomentations import (
    AddGaussianNoise, GainTransition, PolarityInversion, HighPassFilter,
    BandPassFilter, BandStopFilter, Compose
)

class AudioRandAugment:
    def __init__(self, n_ops: int = 2, magnitude_range: tuple = (0.5, 1.0), include_gaussian_noise=False):
        self.magnitude_range = magnitude_range
        self.include_gaussian_noise = include_gaussian_noise
        self.transform_pool = self._get_base_transforms()

        if n_ops > len(self.transform_pool):
            print(f"[⚠️] Requested n_ops={n_ops} but only {len(self.transform_pool)} transforms available. Clipping.")
            self.n_ops = len(self.transform_pool)
        else:
            self.n_ops = n_ops

    def _get_base_transforms(self):
        min_mag, max_mag = self.magnitude_range
        transforms = []

        if self.include_gaussian_noise:
            transforms.append(
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.2, p=1.0)
            )

        transforms.append(GainTransition(
            min_gain_db=-18.0 * min_mag,
            max_gain_db=18.0 * max_mag,
            min_duration=0.1 * min_mag,
            max_duration=0.5 * max_mag,
            p=1.0
        ))
        transforms.append(PolarityInversion(p=1.0))
        transforms.append(HighPassFilter(
            min_cutoff_freq=80 * min_mag,
            max_cutoff_freq=400 * max_mag,
            p=1.0
        ))
        transforms.append(BandPassFilter(
            min_center_freq=300 * min_mag,
            max_center_freq=3000 * max_mag,
            min_bandwidth_fraction=0.2 * min_mag,
            max_bandwidth_fraction=0.4 * max_mag,
            p=1.0
        ))
        transforms.append(BandStopFilter(
            min_center_freq=800 * min_mag,
            max_center_freq=3000 * max_mag,
            min_bandwidth_fraction=0.1 * min_mag,
            max_bandwidth_fraction=0.3 * max_mag,
            p=1.0
        ))

        return transforms

    def __call__(self, samples, sample_rate):
        selected_transforms = sample(self.transform_pool, self.n_ops)
        pipeline = Compose(selected_transforms)
        return pipeline(samples=samples, sample_rate=sample_rate)
