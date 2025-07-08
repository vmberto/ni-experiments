import pandas as pd
from keras import callbacks

from lib.consts import AGNEWS_CORRUPTIONS
from cifar_experiments_config import KFOLD_N_SPLITS
from models.text_autoencoder import TextAutoencoder
from dataset.agnewsdataset import AGNewsDataset  # You should implement this class

MAX_SEQUENCE_LENGTH=128
VOCAB_SIZE=20000
BATCH_SIZE=128
EMBEDDING_DIM=128

def main():
    dataset = AGNewsDataset(max_sequence_length=MAX_SEQUENCE_LENGTH, vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE)

    x_train, x_test, splits = dataset.prepare_kfold_for_autoencoder(KFOLD_N_SPLITS)
    test_ds = dataset.get_dataset_for_autoencoder(x_test)

    results = []

    for fold, (train_index, val_index) in splits:
        train_fold_ds = dataset.get_dataset_for_autoencoder(x_train[train_index])
        val_fold_ds = dataset.get_dataset_for_autoencoder(x_train[val_index])

        autoencoder = TextAutoencoder(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, max_len=MAX_SEQUENCE_LENGTH)
        autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

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

        for corruption_type in AGNEWS_CORRUPTIONS:
            kld = dataset.prepare_agnews_c_with_distances(encoder, corruption_type, test_ds)
            result = {
                "fold": fold,
                "corruption_type": corruption_type,
                "kl_divergences": kld,
            }
            results.append(result)
            pd.DataFrame(results).to_csv('../output/agnews_autoencoder_kldiv.csv', index=False)


if __name__ == "__main__":
    main()
