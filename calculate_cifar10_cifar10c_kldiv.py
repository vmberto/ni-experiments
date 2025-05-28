from dataset.cifar10dataset import Cifar10Dataset
from layers.custom_gaussian_noise import CustomGaussianNoise
from lib.consts import CIFAR10_CORRUPTIONS
from keras import callbacks

from lib.metrics import calculate_kl_divergence
from models.autoencoder import Autoencoder
import pandas as pd
from cifar_experiments_config import (
    KFOLD_N_SPLITS,
    RandAugment,
    SALT_PEPPER_FACTOR,
    GAUSSIAN_STDDEV,
    INPUT_SHAPE,
    BATCH_SIZE,
    RandomSaltAndPepper
)

# Estratégias com nomes e augmentations
SOURCES = {
    "baseline": [],
    "randaug_snp": [RandAugment, RandomSaltAndPepper(max_factor=SALT_PEPPER_FACTOR)],
    "randaug_gaussian": [RandAugment, CustomGaussianNoise(max_stddev=GAUSSIAN_STDDEV)],
}

# Função para categorizar severidade com base em percentis
def categorize_by_percentiles(values):
    q25 = values.quantile(0.25)
    q75 = values.quantile(0.75)

    def label(kld):
        if kld <= q25:
            return 'Lowest'
        elif kld <= q75:
            return 'Mid-Range'
        else:
            return 'Highest'

    return values.apply(label)

def main():
    dataset = Cifar10Dataset((32, 32, 3), BATCH_SIZE)
    x_train, x_test, splits = dataset.prepare_cifar10_kfold_for_autoencoder(KFOLD_N_SPLITS)
    input_shape = x_train.shape[1:]
    test_ds = dataset.get_dataset_for_autoencoder(x_test)

    results = []

    for source_name, augmentations in SOURCES.items():
        for fold, (train_index, val_index) in splits:
            train_fold_ds = dataset.get_dataset_for_autoencoder(x_train[train_index], augmentations)
            val_fold_ds = dataset.get_dataset_for_autoencoder(x_train[val_index], augmentations)

            autoencoder = Autoencoder(input_shape)
            autoencoder.compile(optimizer='adam', loss='mean_squared_error')

            autoencoder.fit(
                train_fold_ds,
                epochs=100,
                shuffle=True,
                validation_data=val_fold_ds,
                callbacks=[
                    callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)
                ]
            )

            encoder = autoencoder.encoder

            # Comparação clean vs. com augmentations
            test_augmented_ds = dataset.get_dataset_for_autoencoder(x_test, augmentations)
            latent_clean = encoder.predict(test_ds)
            latent_augmented = encoder.predict(test_augmented_ds)

            kld = calculate_kl_divergence(latent_clean, latent_augmented)
            results.append({
                "fold": fold,
                "corruption_type": "clean_vs_augmented",
                "source": source_name,
                "kl_divergences": kld
            })

            for corruption_type in CIFAR10_CORRUPTIONS:
                kld = dataset.prepare_cifar10_c_with_distances(encoder, corruption_type, test_augmented_ds)
                results.append({
                    "fold": fold,
                    "corruption_type": corruption_type,
                    "source": source_name,
                    "kl_divergences": kld
                })

            df = pd.DataFrame(results)
            df['severity'] = df.groupby('source')['kl_divergences'].transform(categorize_by_percentiles)
            df.to_csv("output/kl_comparison_summary.csv", index=False)




if __name__ == "__main__":
    main()