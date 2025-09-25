from dataset.cifar10dataset import Cifar10Dataset
from lib.consts import CIFAR10_CORRUPTIONS
from keras import callbacks
from models.autoencoder import Autoencoder
import pandas as pd
from cifar_experiments_config import KFOLD_N_SPLITS, INPUT_SHAPE, BATCH_SIZE

def main():
    dataset = Cifar10Dataset(INPUT_SHAPE, BATCH_SIZE)

    x_train, x_test, splits = dataset.prepare_cifar10_kfold_for_autoencoder(KFOLD_N_SPLITS)
    input_shape = x_train.shape[1:]
    test_ds = dataset.get_dataset_for_autoencoder(x_test)

    results = []

    for fold, (train_index, val_index) in splits:
        train_fold_ds = dataset.get_dataset_for_autoencoder(x_train[train_index])
        val_fold_ds = dataset.get_dataset_for_autoencoder(x_train[val_index])

        autoencoder = Autoencoder(input_shape)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        autoencoder.fit(
            train_fold_ds,
            epochs=1,
            shuffle=True,
            validation_data=val_fold_ds,
            callbacks=[
                callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)
            ]
        )

        encoder = autoencoder.encoder

        for corruption_type in CIFAR10_CORRUPTIONS:
            kld = dataset.prepare_cifar10_c_with_distances(encoder, corruption_type, test_ds)
            result = {
                "fold": fold,
                "corruption_type": corruption_type,
                "kl_divergences": kld,
            }
            results.append(result)
            pd.DataFrame(results).to_csv('output/autoencoder_results_kldiv.csv')


if __name__ == "__main__":
    main()
