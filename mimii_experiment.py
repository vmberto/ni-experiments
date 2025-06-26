import itertools

from mtsa import calculate_aucroc

from lib.metrics import write_auc_results
from lib.consts import IN_DISTRIBUTION_LABEL
from lib.logger import print_execution, print_evaluation, print_training_stage
from lib.helpers import filter_active, init_model
from keras import callbacks
import os
from datetime import datetime


def curriculum_train_and_evaluate_model(experiment_dir, config, epochs, dataset, model_architecture, train_index, val_index, xtrain, ytrain, x_test, y_test, fold, mimii_machine, machine_types):
    strategy_name = config['strategy_name']
    augmentation_stages = config['data_augmentation_layers']

    model = init_model(
        model_architecture,
        use_MFCC=True,
        mono=True,
        use_array2mfcc=True,
        isForWaveData=True,
        learning_rate=1e-3,
        batch_size=512,
        verbose=1,
        epochs=epochs,
    )

    print_execution(fold, f'{mimii_machine}: {strategy_name}', model.name)

    for stage_index, aug_layers in enumerate(augmentation_stages):
        print_training_stage(stage_index, aug_layers)
        x_train, y_train = dataset.get(xtrain[train_index], ytrain[train_index], augmentation_layer=aug_layers)
        x_val, y_val = dataset.get(xtrain[val_index], ytrain[val_index])

        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            # callbacks=[callbacks.EarlyStopping(patience=config['es_patience_stages'][stage_index], monitor='val_loss',
            #                                restore_best_weights=True, verbose=1)])
        )

    print_evaluation(fold, strategy_name, model.name, IN_DISTRIBUTION_LABEL)
    auc = calculate_aucroc(model, x_test, y_test)
    write_auc_results(experiment_dir, mimii_machine, strategy_name, model.name, fold, auc)

    # evaluate_ood(experiment_dir, model, dataset, strategy_name, model.name, fold, mimii_machine, machine_types)


def train_and_evaluate_model(experiment_dir, config, epochs, dataset, model_architecture, train_index, val_index, xtrain, ytrain, x_test, y_test, fold, mimii_machine, machine_types):
    strategy_name = config['strategy_name']
    data_augmentation_layers = config['data_augmentation_layers']

    model = init_model(
        model_architecture,
        use_MFCC=True,
        mono=True,
        use_array2mfcc=True,
        isForWaveData=True,
        learning_rate=1e-3,
        batch_size=512,
        verbose=1,
        epochs=epochs,
    )

    print_execution(fold, f'{mimii_machine}: {strategy_name}', model.name)

    x_train, y_train = dataset.get(xtrain[train_index], ytrain[train_index], augmentation_layer=data_augmentation_layers)
    x_val, y_val = dataset.get(xtrain[val_index], ytrain[val_index])

    model.fit(
        x_train,
        validation_data=(x_val, x_val),
        callbacks=[
            # callbacks.EarlyStopping(
            #     patience=10,
            #     monitor='val_loss',
            #     restore_best_weights=True,
            #     verbose=1
            # )
        ],
    )

    print_evaluation(fold, strategy_name, model.name, IN_DISTRIBUTION_LABEL)
    auc = calculate_aucroc(model, x_test, y_test)

    write_auc_results(experiment_dir, mimii_machine, strategy_name, model.name, fold, auc)

    # evaluate_ood(experiment_dir, model, dataset, strategy_name, model.name, fold, mimii_machine, machine_types)


def evaluate_ood(experiment_dir, model, dataset, strategy_name, model_name, fold, machine, machine_types):
    for machine_type in machine_types:
        print_evaluation(fold, strategy_name, model_name, f'in {machine}:{machine_type}')
        x_ood, y_ood = dataset.get_ood_dataset(machine, machine_type)

        auc = calculate_aucroc(model, x_ood, y_ood)

        write_auc_results(experiment_dir, machine, strategy_name, model.name, fold, auc, evaluation_set=machine_type)


def experiment(dataset, epochs, kfold_n_splits, configs, model_architectures, mimii_machines, machine_types):
    experiments_config = filter_active(configs)
    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_dir = os.path.join(os.getcwd(), f'output/experiment_{date_str}')

    for mimii_machine in mimii_machines:
        x_train, y_train, x_test, y_test, splits = dataset.get_kfold_splits(kfold_n_splits, mimii_machine)

        combinations = itertools.product(enumerate(experiments_config), model_architectures, splits)

        for (config_index, config), model_architecture, (fold, (train_index, val_index)) in combinations:
            fold_number = fold + 1
            if not config['curriculum_learning']:
                train_and_evaluate_model(
                    experiment_dir,
                    config,
                    epochs,
                    dataset,
                    model_architecture,
                    train_index,
                    val_index,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    fold_number,
                    mimii_machine,
                    machine_types,
                )
            else:
                curriculum_train_and_evaluate_model(
                    experiment_dir,
                    config,
                    epochs,
                    dataset,
                    model_architecture,
                    train_index,
                    val_index,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    fold_number,
                    mimii_machine,
                    machine_types,
                )
