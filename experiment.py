import os
import itertools
from dataset.cifar import get_cifar10_kfold_splits, get_cifar10_dataset, get_cifar10_corrupted
from experiments_config import CONFIGS, KFOLD_N_SPLITS, INPUT_SHAPE, EPOCHS, MODEL_ARCHITECTURES
from lib.metrics import write_fscore_result
from lib.consts import CORRUPTIONS, IN_DISTRIBUTION_LABEL
from lib.logger import print_execution, print_evaluation
from keras import callbacks
from lib.functions import filter_active
from lib.gpu import set_memory_growth
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_and_evaluate_model(config, model_architecture, train_index, val_index, x_train, y_train, x_test, y_test, fold):
    strategy_name = config['strategy_name']
    data_augmentation_layers = config['data_augmentation_layers']
    mixed = config['mixed']

    model = model_architecture(input_shape=INPUT_SHAPE)

    print_execution(fold, strategy_name, model.name)

    train_ds = get_cifar10_dataset(x_train[train_index], y_train[train_index], data_augmentation_layers, mixed=mixed)
    val_ds = get_cifar10_dataset(x_train[val_index], y_train[val_index])
    test_ds = get_cifar10_dataset(x_test, y_test)

    _, training_time = model.fit(
        train_ds,
        val_dataset=val_ds,
        epochs=EPOCHS,
        callbacks=[callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)]
    )

    print_evaluation(fold, strategy_name, model.name, IN_DISTRIBUTION_LABEL)

    report = model.predict(test_ds)
    loss, acc = model.evaluate(test_ds)

    write_fscore_result(
        IN_DISTRIBUTION_LABEL,
        strategy_name,
        model.name,
        training_time,
        fold,
        loss,
        acc,
        report,
    )

    evaluate_corruptions(model, strategy_name, model.name, training_time, fold)


def evaluate_corruptions(model, strategy_name, model_name, training_time, fold):
    for corruption in CORRUPTIONS:
        print_evaluation(fold, strategy_name, model_name, f'in {corruption}')
        corrupted_dataset = get_cifar10_corrupted(corruption)

        report = model.predict(corrupted_dataset)
        loss, acc = model.evaluate(corrupted_dataset)

        write_fscore_result(
            corruption,
            strategy_name,
            model_name,
            training_time,
            fold,
            loss,
            acc,
            report,
        )


def experiment():
    set_memory_growth(tf)
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs available:", gpus)

    x_train, y_train, x_test, y_test, splits = get_cifar10_kfold_splits(KFOLD_N_SPLITS)
    experiments_config = filter_active(CONFIGS)

    combinations = itertools.product(enumerate(experiments_config), MODEL_ARCHITECTURES, splits)

    for (config_index, config), model_architecture, (fold, (train_index, val_index)) in combinations:
        fold_number = fold + 1
        train_and_evaluate_model(
            config,
            model_architecture,
            train_index,
            val_index,
            x_train,
            y_train,
            x_test,
            y_test,
            fold_number
        )


if __name__ == "__main__":
    experiment()
