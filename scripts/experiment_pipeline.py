import itertools
from datetime import datetime
import os
from keras import callbacks
from lib.metrics import write_fscore_result
from lib.consts import IN_DISTRIBUTION_LABEL
from lib.logger import print_execution, print_evaluation
from lib.helpers import filter_active


def train_model_with_params(params):
    config = params["config"]
    # Determine number of classes from dataset
    num_classes = params.get("num_classes", 10)
    input_shape = params.get("input_shape", (32, 32, 3))
    model = params["model_architecture"](input_shape=input_shape, num_classes=num_classes)

    print_execution(params["fold"], config["strategy_name"], model.name)

    if config.get("curriculum_learning", False):
        _run_curriculum_training(model, params)
    else:
        _run_standard_training(model, params)

    _evaluate_and_log(model, params)


def _run_standard_training(model, params):
    config = params["config"]
    dataset = params["dataset"]

    train_ds = dataset.get(params["x_train"][params["train_index"]],
                           params["y_train"][params["train_index"]],
                           config["data_augmentation_layers"])

    val_ds = dataset.get(params["x_train"][params["val_index"]],
                         params["y_train"][params["val_index"]])

    test_ds = dataset.get(params["x_test"], params["y_test"])

    history, training_time = model.fit(
        train_ds,
        val_dataset=val_ds,
        epochs=params["epochs"],
        callbacks=[callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)]
    )

    params["num_epochs_run"] = len(history.history["loss"])
    params["training_time_total"] = training_time
    params["test_ds"] = test_ds
    params["model"] = model


def _run_curriculum_training(model, params):
    config = params["config"]
    dataset = params["dataset"]

    total_epochs_run = 0
    training_time_total = 0

    val_ds = dataset.get(params["x_train"][params["val_index"]],
                         params["y_train"][params["val_index"]])
    test_ds = dataset.get(params["x_test"], params["y_test"])

    for stage_index, aug_layers in enumerate(config["data_augmentation_layers"]):
        print(f"\n--- Stage {stage_index + 1} ---")
        train_ds = dataset.get(params["x_train"][params["train_index"]],
                               params["y_train"][params["train_index"]],
                               aug_layers)

        history, training_time = model.fit(
            train_ds,
            val_dataset=val_ds,
            epochs=params["epochs"],
            callbacks=[callbacks.EarlyStopping(
                patience=config["es_patience_stages"][stage_index],
                monitor='val_loss',
                restore_best_weights=True,
                verbose=1)]
        )

        total_epochs_run += len(history.history["loss"])
        training_time_total += training_time

    params["num_epochs_run"] = total_epochs_run
    params["training_time_total"] = training_time_total
    params["test_ds"] = test_ds
    params["model"] = model


def _evaluate_and_log(model, params):
    config = params["config"]
    strategy_name = config["strategy_name"]
    fold = params["fold"]
    experiment_dir = params["experiment_dir"]
    training_time = params["training_time_total"]
    test_ds = params["test_ds"]
    num_epochs_run = params["num_epochs_run"]

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
        experiment_dir=experiment_dir,
        epochs_run=num_epochs_run
    )

    for corruption in params["corruptions"]:
        print_evaluation(fold, strategy_name, model.name, f'in {corruption}')
        corrupted_dataset = params["dataset"].get_corrupted(corruption)

        report = model.predict(corrupted_dataset)
        loss, acc = model.evaluate(corrupted_dataset)

        write_fscore_result(
            corruption,
            strategy_name,
            model.name,
            training_time,
            fold,
            loss,
            acc,
            report,
            num_epochs_run,
            experiment_dir,
        )


def experiment(dataset, epochs, kfold_n_splits, configs, model_architectures, corruptions, num_classes=10, input_shape=(32, 32, 3)):
    experiments_config = filter_active(configs)
    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_dir = os.path.join(os.getcwd(), f'output/experiment_complete')

    x_train, y_train, x_test, y_test, splits = dataset.get_kfold_splits(kfold_n_splits)

    combinations = itertools.product(enumerate(experiments_config), splits)

    for model in model_architectures:
        for (config_index, config), (fold, (train_index, val_index)) in combinations:
            params = {
                "config": config,
                "model_architecture": model,
                "dataset": dataset,
                "epochs": epochs,
                "experiment_dir": experiment_dir,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
                "train_index": train_index,
                "val_index": val_index,
                "fold": fold + 1,
                "corruptions": corruptions,
                "num_classes": num_classes,
                "input_shape": input_shape,
            }

            train_model_with_params(params)
