from dataset.cifar import get_cifar10_kfold_splits, get_cifar10_dataset, get_cifar10_corrupted
from experiments_config import CONFIGS, KFOLD_N_SPLITS, INPUT_SHAPE, EPOCHS
from lib.metrics import write_fscore_result
from lib.consts import CORRUPTIONS_TYPES
from lib.logger import print_execution, print_evaluation
from keras.callbacks import EarlyStopping
from lib.functions import filter_active
import multiprocessing


def experiment():
    x_train, y_train, x_test, y_test, splits = get_cifar10_kfold_splits(KFOLD_N_SPLITS)

    experiments_config = filter_active(CONFIGS)

    for index, config in enumerate(experiments_config):
        approach_name = config['approach_name']
        model_config = config['model']
        data_augmentation_layers = config['data_augmentation_layers']

        for fold, (train_index, val_index) in splits:
            fold_number = fold + 1
            model = model_config(input_shape=INPUT_SHAPE)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print_execution(fold_number, approach_name, model.name)

            train_ds = get_cifar10_dataset(x_train[train_index], y_train[train_index], data_augmentation_layers)
            val_ds = get_cifar10_dataset(x_train[val_index], y_train[val_index])
            test_ds = get_cifar10_dataset(x_test, y_test)

            _, training_time = model.fit(
                train_ds,
                val_dataset=val_ds,
                epochs=EPOCHS,
                callbacks=[EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)]
            )

            print_evaluation(fold_number, approach_name, model.name, f'in-distribution')

            report = model.predict(test_ds)
            loss, acc = model.evaluate(test_ds)

            write_fscore_result(
                'in-distribution',
                approach_name,
                model.name,
                training_time,
                fold_number,
                loss,
                acc,
                report,
            )

            for corruption in CORRUPTIONS_TYPES:
                print_evaluation(fold_number, approach_name, model.name, f'in {corruption}')

                corrupted_dataset = get_cifar10_corrupted(corruption)
                report = model.predict(corrupted_dataset)
                loss, acc = model.evaluate(corrupted_dataset)

                write_fscore_result(
                    corruption,
                    approach_name,
                    model.name,
                    training_time,
                    fold_number,
                    loss,
                    acc,
                    report,
                )


if __name__ == "__main__":
    p = multiprocessing.Process(target=experiment)
    p.start()
    p.join()
