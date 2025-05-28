from datetime import datetime
import pandas as pd
from lib.functions import clean_string
from scipy.stats import entropy
import os

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
        evauation_set, strategy_name, model_name,
        training_time, fold_number, loss, acc,
        report, epochs_run
):
    csv_file = f'{os.getcwd()}/output/output.csv'

    dict_report = convert_dict(report)

    new_line = {
        'strategy': strategy_name,
        'model': model_name,
        'evaluation_set': clean_string(evauation_set),
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
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=[
                'strategy', 'model', 'evaluation_set', 'date_finished', 'training_time', 'fold', 'accuracy', 'loss',
            ])

    df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)

    df.to_csv(csv_file, index=False)


def calculate_kl_divergence(latent_clean, latent_corrupted):
    epsilon = 1e-10
    latent_clean += epsilon
    latent_corrupted += epsilon
    return entropy(latent_clean.flatten(), latent_corrupted.flatten())
