import itertools

from lib.metrics import write_fscore_result
from lib.consts import IN_DISTRIBUTION_LABEL
from lib.logger import print_execution, print_evaluation
from keras import callbacks, mixed_precision
from lib.functions import filter_active


def curriculum_train_and_evaluate_model(config, dataset, model_architecture, train_index, val_index, x_train, y_train, x_test, y_test, fold, corruptions):
    strategy_name = config['strategy_name']
    augmentation_stages = config['data_augmentation_layers']

    model = model_architecture()

    print_execution(fold, strategy_name, model.name)

    val_ds = dataset.get(x_train[val_index], y_train[val_index])
    test_ds = dataset.get(x_test, y_test)

    total_epochs_run = 0
    training_time_total = 0

    for stage_index, aug_layers in enumerate(augmentation_stages):
        print(f"\n--- Stage {stage_index + 1}: Using augmentations: {[str(l) for l in aug_layers]} ---")

        train_ds = dataset.get(x_train[train_index], y_train[train_index], aug_layers)

        history, training_time = model.fit(
            train_ds,
            val_dataset=val_ds,
            epochs=100,
            callbacks=[callbacks.EarlyStopping(patience=config['es_patience_stages'][stage_index], monitor='val_loss', restore_best_weights=True, verbose=1)],
        )
        total_epochs_run += len(history.history['loss'])
        training_time_total += training_time

    print_evaluation(fold, strategy_name, model.name, IN_DISTRIBUTION_LABEL)

    report = model.predict(test_ds)
    loss, acc = model.evaluate(test_ds)

    write_fscore_result(
        IN_DISTRIBUTION_LABEL,
        strategy_name,
        model.name,
        training_time_total,
        fold,
        loss,
        acc,
        report,
        epochs_run=total_epochs_run
    )

    evaluate_corruptions(model, dataset, strategy_name, model.name, training_time_total, fold, corruptions, total_epochs_run)


def train_and_evaluate_model(config, dataset, model_architecture, train_index, val_index, x_train, y_train, x_test, y_test, fold, corruptions):
    strategy_name = config['strategy_name']
    data_augmentation_layers = config['data_augmentation_layers']

    model = model_architecture()

    print_execution(fold, strategy_name, model.name)

    train_ds = dataset.get(x_train[train_index], y_train[train_index], data_augmentation_layers)
    val_ds = dataset.get(x_train[val_index], y_train[val_index])
    test_ds = dataset.get(x_test, y_test)

    history, training_time = model.fit(
        train_ds,
        val_dataset=val_ds,
        epochs=100,
        callbacks=[callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)]
    )

    num_epochs_run = len(history.history['loss'])

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
        epochs_run=num_epochs_run
    )

    evaluate_corruptions(model, dataset, strategy_name, model.name, training_time, fold, corruptions, num_epochs_run)


def evaluate_corruptions(model, dataset, strategy_name, model_name, training_time, fold, corruptions, num_epochs_run):
    for corruption in corruptions:
        print_evaluation(fold, strategy_name, model_name, f'in {corruption}')
        corrupted_dataset = dataset.get_corrupted(corruption)

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
            epochs_run=num_epochs_run
        )


def experiment(dataset, KFOLD_N_SPLITS, CONFIGS, MODEL_ARCHITECTURES, CORRUPTIONS):
    experiments_config = filter_active(CONFIGS)

    x_train, y_train, x_test, y_test, splits = dataset.get_kfold_splits(KFOLD_N_SPLITS)

    combinations = itertools.product(enumerate(experiments_config), MODEL_ARCHITECTURES, splits)

    for (config_index, config), model_architecture, (fold, (train_index, val_index)) in combinations:
        fold_number = fold + 1
        if not config['curriculum_learning']:
            train_and_evaluate_model(
                config,
                dataset,
                model_architecture,
                train_index,
                val_index,
                x_train,
                y_train,
                x_test,
                y_test,
                fold_number,
                CORRUPTIONS,
            )
        else:
            curriculum_train_and_evaluate_model(
                config,
                dataset,
                model_architecture,
                train_index,
                val_index,
                x_train,
                y_train,
                x_test,
                y_test,
                fold_number,
                CORRUPTIONS,
            )
