from dataset.cifar100dataset import Cifar100Dataset
from lib.consts import CIFAR100_CORRUPTIONS
from keras import callbacks
from models.autoencoder import Autoencoder
import pandas as pd
from pathlib import Path
from datetime import datetime
from cifar100_experiments_config import KFOLD_N_SPLITS, INPUT_SHAPE, BATCH_SIZE


def main():
    # Setup
    output_dir = Path('output/cifar100')
    models_dir = output_dir / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'autoencoder_results_kldiv_{timestamp}.csv'

    print(f"Experiment: {timestamp}")
    print(f"Results: {results_file}")
    print("=" * 70)

    # Prepare dataset
    dataset = Cifar100Dataset(INPUT_SHAPE, BATCH_SIZE)
    x_train, x_test, splits = dataset.prepare_cifar100_kfold_for_autoencoder(KFOLD_N_SPLITS)
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
            epochs=100,
            shuffle=True,
            validation_data=val_fold_ds,
            callbacks=[
                callbacks.EarlyStopping(
                    patience=10,
                    monitor='val_loss',
                    restore_best_weights=True,
                    verbose=1
                )
            ],
            verbose=1
        )

        # Save models
        fold_model_dir = models_dir / f'fold_{fold}'
        fold_model_dir.mkdir(exist_ok=True)

        autoencoder.save(fold_model_dir / 'autoencoder.keras')
        autoencoder.encoder.save(fold_model_dir / 'encoder.keras')
        autoencoder.decoder.save(fold_model_dir / 'decoder.keras')
        pd.DataFrame(history.history).to_csv(
            fold_model_dir / 'training_history.csv',
            index=False
        )

        print(f"✓ Models saved to: {fold_model_dir}")

        # Evaluate
        encoder = autoencoder.encoder

        print(f"\nEvaluating {len(CIFAR100_CORRUPTIONS)} corruptions...")
        for i, corruption_type in enumerate(CIFAR100_CORRUPTIONS, 1):
            print(f"[{i}/{len(CIFAR100_CORRUPTIONS)}] {corruption_type}...", end=' ')

            try:
                kld = dataset.prepare_cifar100_c_with_distances(
                    encoder,
                    corruption_type,
                    test_ds
                )

                result = {
                    "fold": fold,
                    "corruption_type": corruption_type,
                    "kl_divergence": kld,
                    "final_train_loss": history.history['loss'][-1],
                    "final_val_loss": history.history['val_loss'][-1],
                    "epochs_trained": len(history.history['loss']),
                    "timestamp": timestamp
                }

                results.append(result)
                print(f"KL = {kld:.6f} ✓")

                # Save incrementally
                pd.DataFrame(results).to_csv(results_file, index=False)

            except Exception as e:
                print(f"ERROR: {e}")
                continue

        print(f"✓ Fold {fold + 1} completed\n")

    # Summary
    print("=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    if len(results) > 0:
        results_df = pd.DataFrame(results)
        summary = results_df.groupby('corruption_type')['kl_divergence'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(6)

        summary_file = output_dir / f'autoencoder_results_kldiv_{timestamp}_summary.csv'
        summary.to_csv(summary_file)

        print("\nTop 5 Most OOD Corruptions:")
        top = summary.sort_values('mean', ascending=False).head(5)
        for idx, (corruption, row) in enumerate(top.iterrows(), 1):
            print(f"  {idx}. {corruption}: {row['mean']:.6f} ± {row['std']:.6f}")

        print(f"\nResults: {results_file}")
        print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()


