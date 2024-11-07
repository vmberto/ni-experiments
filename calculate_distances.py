import tensorflow as tf
from dataset.cifar import get_dataset_for_autoencoder, prepare_cifar10_kfold_for_autoencoder, prepare_cifar10_c_with_distances
from lib.consts import CORRUPTIONS_TYPES
from lib.gpu import set_memory_growth
from models.autoencoder import Autoencoder
import pandas as pd
import multiprocessing
from experiments_config import BATCH_SIZE, KFOLD_N_SPLITS, EPOCHS


def main():
    set_memory_growth(tf)

    x_train, x_test, splits = prepare_cifar10_kfold_for_autoencoder(KFOLD_N_SPLITS)
    input_shape = x_train.shape[1:]
    test_ds = get_dataset_for_autoencoder(x_test)

    results = []

    for fold, (train_index, val_index) in splits:
        train_fold_ds = get_dataset_for_autoencoder(x_train[train_index])
        val_fold_ds = get_dataset_for_autoencoder(x_train[val_index])

        autoencoder = Autoencoder(input_shape)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        autoencoder.fit(
            train_fold_ds,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            validation_data=val_fold_ds,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)
            ]
        )

        encoder = autoencoder.encoder

        for corruption_type in CORRUPTIONS_TYPES:
            kld = prepare_cifar10_c_with_distances(encoder, corruption_type, test_ds)
            result = {
                "fold": fold,
                "corruption_type": corruption_type,
                "kl_divergences": kld,
            }
            results.append(result)
            pd.DataFrame(results).to_csv('results/autoencoder_results_kldiv.csv')


if __name__ == "__main__":
    try:
        p = multiprocessing.Process(target=main)
        p.start()
        p.join()
    except Exception as e:
        print(f"Error occurred: {e}")
