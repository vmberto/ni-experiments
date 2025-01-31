from datetime import datetime
import pandas as pd
from lib.functions import clean_string
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np


def plot_loss_convergence(results):
    """
    Plots Loss vs Val_Loss convergence for all folds of each model and strategy combination.

    Parameters:
        results (dict): Nested dictionary containing the training history for each model-strategy combination and fold.

    Saves:
        - A PDF and JPEG plot for Loss vs Val_Loss convergence for each model-strategy.
        - A JSON file containing the history data for each model-strategy.
    """
    sns.set(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],  # Specify a list of fallback options
        "text.usetex": False,  # Disable TeX if not required
    })

    output_dir = "./output"
    json_path = f"{output_dir}/loss_convergence.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    for model_strategy, history_data in results.items():
        num_folds = len(history_data)

        plt.figure(figsize=(12, 8))

        epochs = 0
        for i, (fold, data) in enumerate(history_data.items()):
            loss = data["loss"]
            val_loss = data["val_loss"]
            epochs = np.arange(1, len(loss) + 1)

            sns.lineplot(x=epochs, y=loss, linewidth=2.5)
            sns.lineplot(x=epochs, y=val_loss, linestyle="--", linewidth=2.5)

        plt.title(f"{model_strategy}", fontsize=28)
        plt.xlabel("Epochs", fontsize=36)
        plt.ylabel("Loss", fontsize=36)
        plt.xticks(epochs, fontsize=24)
        plt.yticks(fontsize=20)
        plt.legend([], [], frameon=False)
        plt.tight_layout()

        sanitized_name = model_strategy.replace(" ", "_").replace("/", "_")
        plot_pdf_path = f"{output_dir}/loss_convergence_{sanitized_name}_plot.pdf"
        plot_jpeg_path = f"{output_dir}/loss_convergence_{sanitized_name}_plot.jpeg"
        plt.savefig(plot_pdf_path, bbox_inches='tight')
        plt.savefig(plot_jpeg_path, bbox_inches='tight')

        print(f"Plot saved to {plot_pdf_path}")
        print(f"Plot saved to {plot_jpeg_path}")
        print(f"History saved to {json_path}")


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
