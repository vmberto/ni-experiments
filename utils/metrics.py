from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.functions import extract_preprocessing_layer_names
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy
import os

sns.set_theme(style="whitegrid")


def write_evaluation_result(corruption_type, aug_layers, execution_name, loss, acc):
    csv_file = f'{os.getcwd()}/output/output.csv'
    augmentation_layers = extract_preprocessing_layer_names(aug_layers)

    new_line = {
        'Execution Name': execution_name,
        'Corruption Type': corruption_type,
        'Date': datetime.now(),
        'Augment Layers': augmentation_layers,
        'Accuracy': acc,
        'Loss': loss
    }

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Execution Name', 'Corruption Type', 'Date', 'Augment Layers', 'Accuracy', 'Loss'])

    df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)

    df.to_csv(csv_file, index=False)


def write_fscore_result(corruption_type, aug_layers, execution_name, fscore):
    csv_file = f'{os.getcwd()}/output/output.csv'
    augmentation_layers = extract_preprocessing_layer_names(aug_layers)

    new_line = {
        'Execution Name': execution_name,
        'Corruption Type': corruption_type,
        'Date': datetime.now(),
        'Augment Layers': augmentation_layers,
        'Fscore': fscore,
    }

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=['Execution Name', 'Corruption Type', 'Date', 'Augment Layers', 'Fscore'])

    df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)

    df.to_csv(csv_file, index=False)


def save_confusion_matrix(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{os.getcwd()}/output/confusion_matrix.png')
    plt.clf()


def plot_history(execution_name, history, fold_number):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{os.getcwd()}/output/{execution_name}_fold_{str(fold_number)}_history.png')


def write_acc_each_dataset(execution_name):
    df = pd.read_csv(f"{os.getcwd()}/output/output.csv")

    for i, augment_layer in enumerate(['Baseline', 'S&P', 'DefaultAug', 'DefaultAug+S&P']):
        rows = df.loc[df['Augment Layers'] == augment_layer]

        sns.barplot(x=rows['Corruption Type'], y=rows['Accuracy'])
        plt.xlabel('Dataset')
        plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig(f'{os.getcwd()}/output/{execution_name}_barplot-{augment_layer}.png')
        plt.clf()


def write_acc_each_dataset_line(execution_name):
    df = pd.read_csv(f"{os.getcwd()}/output/output.csv")

    sns.lineplot(df, x=df['Corruption Type'], y=df['Accuracy'], hue=df['Augment Layers'])
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')

    plt.xticks(rotation=45)
    plt.savefig(f'{os.getcwd()}/output/{execution_name}_lineplot.png')
    plt.clf()


def plot_confidence_interval(execution_name):
    df = pd.read_csv(f"{os.getcwd()}/output/output.csv")

    def ci(data):
        ci = scipy.stats.bootstrap((data,), np.std, confidence_level=0.95).confidence_interval
        low, high = ci
        interval = high - low
        mean = np.mean(data)
        return mean - interval, mean + interval

    ax = sns.heatmap(data=df, x='Accuracy', y='Corruption Type', hue='Execution Name')

    plt.subplots_adjust(top=0.925,
                        bottom=0.20,
                        left=0.07,
                        right=0.90,
                        hspace=0.01,
                        wspace=0.01)
    ax.legend(ncol=2, loc="lower right", frameon=True)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(f'{os.getcwd()}/output/{execution_name}_ci.png')
