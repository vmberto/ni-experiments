import itertools
import optuna
import os
from audiomentations import AddGaussianNoise
from dataset.mimiidataset import MIMIIDataset
from layers.audio_randaugment import AudioRandAugment
from mtsa import calculate_aucroc, Hitachi, files_train_test_split
from lib.logger import print_execution, print_evaluation
from lib.consts import IN_DISTRIBUTION_LABEL

# 🎛️ All transform names
ALL_TRANSFORMS = [
    'GainTransition', 'PolarityInversion', 'HighPassFilter', 'LowPassFilter',
    'BandPassFilter', 'BandStopFilter', 'PitchShift', 'Shift', 'TimeMask', 'TimeStretch'
]

# 📦 Generate subsets of size 2 to 6
SUBSET_POOL = []
for k in range(2, 7):
    SUBSET_POOL.extend(list(itertools.combinations(ALL_TRANSFORMS, k)))


# 🧠 Training & evaluation logic
def custom_train_and_evaluate(config, dataset, x_train, y_train, x_test, y_test, fold_number, epochs=500):
    model = Hitachi(use_MFCC=True, mono=True, learning_rate=1e-3, batch_size=512, verbose=1, epochs=epochs)
    print_execution(fold_number, f"valve: {config['strategy_name']}", model.name)
    x_train_aug, y_train_aug = dataset.get(x_train, y_train, augmentation_layer=config["data_augmentation_layers"])
    model.fit(x_train_aug)
    print_evaluation(fold_number, config["strategy_name"], model.name, IN_DISTRIBUTION_LABEL)
    return calculate_aucroc(model, x_test, y_test)


# 🔧 RandAugment builder
def make_audio_randaugment(n_ops, magnitude_range, subset=None, include_gaussian_noise=False):
    class CustomAudioRandAugment(AudioRandAugment):
        def _get_base_transforms(self):
            min_mag, max_mag = self.magnitude_range
            pool = super()._get_base_transforms()
            return [t for t in pool if t.__class__.__name__ in subset] if subset else pool

    return CustomAudioRandAugment(
        n_ops=n_ops,
        magnitude_range=magnitude_range,
        include_gaussian_noise=include_gaussian_noise
    )


# 🎯 Optuna objective factory
def make_objective(dataset, x_train, y_train, x_test, y_test):
    def objective(trial):
        use_all = trial.suggest_categorical("use_all", [True, False])
        include_gaussian_inside = trial.suggest_categorical("include_gaussian_inside", [True, False])
        use_extra_gaussian = trial.suggest_categorical("use_extra_gaussian", [True, False])

        transform_subset = ALL_TRANSFORMS if use_all else list(SUBSET_POOL[trial.suggest_int("subset_index", 0, len(SUBSET_POOL) - 1)])

        min_mag = trial.suggest_float("min_mag", 0.2, 0.5)
        max_mag = trial.suggest_float("max_mag", min_mag + 0.2, 1.0)
        n_ops = trial.suggest_int("n_ops", 1, len(transform_subset))  # will be capped inside the class

        base_aug = make_audio_randaugment(
            n_ops=n_ops,
            magnitude_range=(min_mag, max_mag),
            subset=transform_subset,
            include_gaussian_noise=include_gaussian_inside
        )

        if use_extra_gaussian:
            def combined(samples, sample_rate):
                samples = base_aug(samples, sample_rate)
                return AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.2, p=1.0)(
                    samples=samples, sample_rate=sample_rate
                )
            aug_pipeline = combined
        else:
            aug_pipeline = base_aug

        config = {
            "strategy_name": f"SmartCombo_trial_{trial.number}_inGN{include_gaussian_inside}_afterGN{use_extra_gaussian}",
            "data_augmentation_layers": aug_pipeline
        }

        return custom_train_and_evaluate(config, dataset, x_train, y_train, x_test, y_test, fold_number=1)

    return objective


# 🚀 Search entrypoint
def run_autoaugment_search(dataset, x_train, y_train, x_test, y_test):
    study = optuna.create_study(direction="maximize")
    study.optimize(make_objective(dataset, x_train, y_train, x_test, y_test), n_trials=20)

    print("\n🏆 Top 3 Trials:")
    top_trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:3]
    for i, trial in enumerate(top_trials):
        print(f"\n🥇 Rank #{i+1}")
        print(f"AUC: {trial.value}")
        print(f"Params: {trial.params}")
        print(f"Strategy: SmartCombo_trial_{trial.number}_inGN{trial.params.get('include_gaussian_inside', '?')}_afterGN{trial.params.get('use_extra_gaussian', '?')}")


# 🧪 Baseline trial: RandAugment only
def run_baseline_trial(dataset, x_train, y_train, x_test, y_test):
    print("\n🧪 Baseline Trial: RandAugment Only")
    aug = make_audio_randaugment(n_ops=2, magnitude_range=(0.5, 1.0), subset=None, include_gaussian_noise=False)
    config = {
        "strategy_name": "RandAugment_Only_Baseline",
        "data_augmentation_layers": aug
    }
    auc = custom_train_and_evaluate(config, dataset, x_train, y_train, x_test, y_test, fold_number=0)
    print(f"📊 RandAugment Only Baseline AUC: {auc:.4f}")


# 🏁 Main
if __name__ == '__main__':
    BATCH_SIZE = 512
    dataset = MIMIIDataset(BATCH_SIZE)

    x_train, x_test, y_train, y_test = files_train_test_split(
        f'{os.getcwd()}/../dataset/mimii_dataset/valve/id_00'
    )

    run_baseline_trial(dataset, x_train, y_train, x_test, y_test)
    run_autoaugment_search(dataset, x_train, y_train, x_test, y_test)

