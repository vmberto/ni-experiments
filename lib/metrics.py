from datetime import datetime
import pandas as pd
from lib.helpers import clean_string
from scipy.stats import entropy
import os
import shutil


def create_experiment_folder(experiment_dir, config_path: str) -> str:
    os.makedirs(experiment_dir, exist_ok=True)

    config_dst = os.path.join(experiment_dir, 'config')
    try:
        shutil.copyfile(config_path, config_dst)
    except FileNotFoundError:
        print(f"Warning: config file '{config_path}' not found. Skipping copy.")

    return experiment_dir


def convert_dict(data_dict):
    converted_dict = {}
    for key, value in data_dict.items():
        if key.isdigit():
            new_value = {}
            for sub_key, sub_value in value.items():
                new_value[f"{sub_key}({key})"] = sub_value
            converted_dict.update(new_value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                new_key = f"{sub_key}({key})"
                converted_dict[new_key] = sub_value
        else:
            converted_dict[key] = value
    return converted_dict


def write_fscore_result(
    evaluation_set, strategy_name, model_name,
    training_time, fold_number, loss, acc,
    report, epochs_run, experiment_dir,
    config_path='./cifar_experiments_config.py'
):
    experiment_dir = create_experiment_folder(experiment_dir, config_path)
    csv_file = os.path.join(experiment_dir, 'output.csv')

    base_fields = {
        'strategy', 'model', 'evaluation_set', 'date_finished',
        'training_time', 'fold', 'accuracy', 'loss', 'epochs_run'
    }

    dict_report_raw = convert_dict(report)
    dict_report = {k: v for k, v in dict_report_raw.items() if k not in base_fields}

    new_line = {
        'strategy': strategy_name,
        'model': model_name,
        'evaluation_set': clean_string(evaluation_set),
        'date_finished': datetime.now(),
        'training_time': training_time,
        'fold': fold_number,
        'accuracy': acc,
        'loss': loss,
        'epochs_run': epochs_run,
        **dict_report,
    }

    try:
        df = pd.read_csv(csv_file)
        # Ensure no duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
    except FileNotFoundError:
        # Use all current keys as initial columns
        df = pd.DataFrame(columns=list(new_line.keys()))

    # Also make sure the new_line matches df columns exactly
    for col in df.columns:
        if col not in new_line:
            new_line[col] = None  # fill missing keys

    df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)
    df.to_csv(csv_file, index=False)

    return experiment_dir


def write_auc_results(experiment_dir, machine, strategy_name, model_name, fold_number, auc_value, evaluation_set='In-Distribution'):
    config_path = './mimii_experiments_config.py'
    create_experiment_folder(experiment_dir, config_path)
    csv_file = os.path.join(experiment_dir, 'output.csv')

    new_line = {
        'machine': machine,
        'strategy': strategy_name,
        'model': model_name,
        'evaluation_set': evaluation_set,
        'fold': fold_number,
        'auc_roc': auc_value,
    }

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=['strategy', 'model', 'evaluation_set', 'fold', 'auc_roc']
        )

    df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)
    df.to_csv(csv_file, index=False)

    return experiment_dir


def calculate_kl_divergence(latent_clean, latent_corrupted):
    epsilon = 1e-10
    latent_clean += epsilon
    latent_corrupted += epsilon
    return entropy(latent_clean.flatten(), latent_corrupted.flatten())
