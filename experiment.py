from dataset.cifar import get_cifar10_kfold_splits, get_cifar10_dataset, get_cifar10_corrupted
from experiments_config import configs as all_experiments
from utils.metrics import write_fscore_result
from utils.consts import CORRUPTIONS_TYPES
from keras.callbacks import EarlyStopping
from utils.functions import filter_active
from sklearn.metrics import f1_score
import multiprocessing
import numpy as np

KFOLD_N_SPLITS = 10
INPUT_SHAPE = (72, 72, 3)


def experiment():
    x, y, splits = get_cifar10_kfold_splits(KFOLD_N_SPLITS)

    experiments_config = filter_active(all_experiments)

    for index, config in enumerate(experiments_config):
        approach_name = config['approach_name']
        model_config = config['model']
        data_augmentation_layers = config['data_augmentation_layers']

        for fold_number, (train_index, val_index) in splits:
            model = model_config(input_shape=INPUT_SHAPE)
            print("\033[92m{}\033[00m".format(f'Executing Fold {fold_number} for {approach_name} with Model {model.name}'))

            x_train_fold, y_train_fold = x[train_index], y[train_index]
            x_val_fold, y_val_fold = x[val_index], y[val_index]

            train_ds = get_cifar10_dataset(x_train_fold, y_train_fold, data_augmentation_layers)
            val_ds = get_cifar10_dataset(x_val_fold, y_val_fold)

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            _, training_time = model.fit(
                train_ds,
                val_dataset=val_ds,
                epochs=100,
                callbacks=[EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)]
            )

            for corruption in CORRUPTIONS_TYPES:
                print("\033[94m{}\033[00m"
                      .format(
                         f'Evaluating Fold {fold_number} for {approach_name} with Model {model.name} in corruption {corruption}'
                ))

                corrupted_dataset = get_cifar10_corrupted(corruption)
                corrupted_labels = np.concatenate([y for x, y in corrupted_dataset], axis=0)

                predictions = model.predict(corrupted_dataset)

                predicted_labels = np.argmax(predictions, axis=1)
                f1 = f1_score(corrupted_labels, predicted_labels, average='macro')

                write_fscore_result(
                    corruption,
                    approach_name,
                    model.name,
                    f1,
                    training_time,
                )


if __name__ == "__main__":
    p = multiprocessing.Process(target=experiment)
    p.start()
    p.join()
    print("finished")
