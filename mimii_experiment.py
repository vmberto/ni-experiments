import itertools

from mtsa import calculate_aucroc
from tensorflow.python.ops.gen_data_flow_ops import stage

from lib.metrics import write_auc_results
from lib.consts import IN_DISTRIBUTION_LABEL
from lib.logger import print_execution, print_evaluation, print_training_stage
from lib.helpers import filter_active


def curriculum_train_and_evaluate_model(config, epochs, dataset, model_architecture, train_index, val_index, x_train, y_train, x_test, y_test, fold, different_machines):
    strategy_name = config['strategy_name']
    augmentation_stages = config['data_augmentation_layers']

    model = model_architecture(use_MFCC=True, mono=True, learning_rate=1e-3, batch_size=512, verbose=1, epochs=epochs)

    print_execution(fold, strategy_name, model.name)

    for stage_index, aug_layers in enumerate(augmentation_stages):
        print_training_stage(stage, aug_layers)
        x_train, y_train = x_train[train_index], y_train[train_index]
        # x_val, y_val = x_train[val_index], y_train[val_index]
        x_train, y_train = dataset.get(x_train, y_train, augmentation_layer=aug_layers)

        model.fit(x_train, y_train)

    print_evaluation(fold, strategy_name, model.name, IN_DISTRIBUTION_LABEL)
    auc = calculate_aucroc(model, x_test, y_test)
    write_auc_results(strategy_name, model.name, fold, auc)

    evaluate_ood(model, dataset, strategy_name, model.name, fold, different_machines)


def train_and_evaluate_model(config, epochs, dataset, model_architecture, train_index, val_index, x_train, y_train, x_test, y_test, fold, different_machines):
    strategy_name = config['strategy_name']
    data_augmentation_layers = config['data_augmentation_layers']

    model = model_architecture(use_MFCC=True, mono=True, learning_rate=1e-3, batch_size=512, verbose=1, epochs=epochs)

    print_execution(fold, strategy_name, model.name)

    x_train, y_train = x_train[train_index], y_train[train_index]
    # x_val, y_val = x_train[val_index], y_train[val_index]
    x_train, y_train = dataset.get(x_train, y_train, augmentation_layer=data_augmentation_layers)

    model.fit(x_train, y_train)

    print_evaluation(fold, strategy_name, model.name, IN_DISTRIBUTION_LABEL)
    auc = calculate_aucroc(model, x_test, y_test)

    write_auc_results(strategy_name, model.name, fold, auc)

    evaluate_ood(model, dataset, strategy_name, model.name, fold, different_machines)


def evaluate_ood(model, dataset, strategy_name, model_name, fold, different_machines):
    for machine in different_machines:
        print_evaluation(fold, strategy_name, model_name, f'in {machine}')
        x_ood, y_ood = dataset.get_ood_dataset(machine)

        auc = calculate_aucroc(model, x_ood, y_ood)

        write_auc_results(strategy_name, model.name, fold, auc, evaluation_set=machine)


def experiment(dataset, epochs, kfold_n_splits, configs, model_architectures, different_machines):
    experiments_config = filter_active(configs)

    x_train, y_train, x_test, y_test, splits = dataset.get_kfold_splits(kfold_n_splits)

    combinations = itertools.product(enumerate(experiments_config), model_architectures, splits)

    for (config_index, config), model_architecture, (fold, (train_index, val_index)) in combinations:
        fold_number = fold + 1
        if not config['curriculum_learning']:
            train_and_evaluate_model(
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
                different_machines,
            )
        else:
            curriculum_train_and_evaluate_model(
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
                different_machines,
            )
