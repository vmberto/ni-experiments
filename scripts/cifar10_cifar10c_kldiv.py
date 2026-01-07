from dataset.cifar10dataset import Cifar10Dataset
from lib.consts import CIFAR10_CORRUPTIONS
from keras import callbacks
from models.autoencoder import Autoencoder
import pandas as pd
from pathlib import Path
from datetime import datetime
from cifar_experiments_config import KFOLD_N_SPLITS, INPUT_SHAPE, BATCH_SIZE
import numpy as np
import tensorflow_datasets as tfds
from dataset.ood_characterization import calculate_kl_divergence, WassersteinComparer


def main():
    # Setup
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'autoencoder_results_kldiv_{timestamp}.csv'

    print(f"Experiment: {timestamp}")
    print(f"Results: {results_file}")
    print("=" * 70)

    # Prepare dataset
    dataset = Cifar10Dataset(INPUT_SHAPE, BATCH_SIZE)
    x_train, x_test, splits = dataset.prepare_cifar10_kfold_for_autoencoder(KFOLD_N_SPLITS)
    input_shape = x_train.shape[1:]
    test_ds = dataset.get_dataset_for_autoencoder(x_test)

    results = []

    for fold, (train_index, val_index) in splits:
        print(f"\nFOLD {fold + 1}/{KFOLD_N_SPLITS}")
        print("-" * 70)

        # Prepare data
        train_fold_ds = dataset.get_dataset_for_autoencoder(x_train[train_index])
        val_fold_ds = dataset.get_dataset_for_autoencoder(x_train[val_index])

        # Train
        autoencoder = Autoencoder(input_shape)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        history = autoencoder.fit(
            train_fold_ds,
            epochs=50,
            shuffle=True,
            validation_data=val_fold_ds,
            callbacks=[
                callbacks.EarlyStopping(
                    patience=5,
                    monitor='val_loss',
                    restore_best_weights=True,
                    verbose=1
                )
            ],
            verbose=1
        )

        encoder = autoencoder.encoder

        print(f"\nEvaluating {len(CIFAR10_CORRUPTIONS)} corruptions...")
        
        # Get clean latent representations (only once per fold)
        print(f"Encoding clean test set...")
        latent_clean = encoder.predict(test_ds)
        for i, corruption_type in enumerate(CIFAR10_CORRUPTIONS, 1):
            print(f"[{i}/{len(CIFAR10_CORRUPTIONS)}] {corruption_type}...", end=' ')

            try:
                # Load corrupted images and get latents
                cifar10_c_ds = tfds.load(f'cifar10_corrupted/{corruption_type}', split='test', as_supervised=True)
                x_corrupted = np.array([image for image, _ in tfds.as_numpy(cifar10_c_ds)])
                x_corrupted = x_corrupted.astype('float32') / 255.0
                
                corrupted_ds = dataset.get_dataset_for_autoencoder(x_corrupted)
                latent_corrupted = encoder.predict(corrupted_ds)
                
                # Calculate multiple OOD metrics
                kl_feature_wise = calculate_kl_divergence(latent_clean, latent_corrupted, method='feature_wise')
                kl_flatten = calculate_kl_divergence(latent_clean, latent_corrupted, method='flatten')
                kl_histogram = calculate_kl_divergence(latent_clean, latent_corrupted, method='histogram')
                
                # Wasserstein distances
                wd_comparer_per_feature = WassersteinComparer(mode='per_feature')
                wd_per_feature = wd_comparer_per_feature.compare(latent_clean, latent_corrupted)
                
                wd_comparer_sliced = WassersteinComparer(mode='sliced', n_projections=128)
                wd_sliced = wd_comparer_sliced.compare(latent_clean, latent_corrupted)

                result = {
                    "fold": fold,
                    "corruption_type": corruption_type,
                    "kl_feature_wise": kl_feature_wise,
                    "kl_flatten": kl_flatten,
                    "kl_histogram": kl_histogram,
                    "wasserstein_per_feature": wd_per_feature,
                    "wasserstein_sliced": wd_sliced,
                    "timestamp": timestamp
                }

                results.append(result)
                print(f"KL_fw={kl_feature_wise:.6f}, WD_pf={wd_per_feature:.6f} ✓")

                # Save incrementally
                pd.DataFrame(results).to_csv(results_file, index=False)

            except Exception as e:
                print(f"ERROR: {e}")
                continue

        print(f"✓ Fold {fold + 1} completed\n")

    # Final message
    print("=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
