from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.functions import extract_preprocessing_layer_names
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import os
sns.set_theme(style="whitegrid")


def write_evaluation_result(evaluation_name, aug_layers, execution_name, loss, acc):
    csv_file = f'{os.getcwd()}/output/output.csv'
    augmentation_layers = extract_preprocessing_layer_names(aug_layers)
    new_line = {'Execution Name': execution_name, 'Evaluation Name': evaluation_name, 'Date': datetime.now(), 'Augment Layers': augmentation_layers, 'Accuracy': acc, 'Loss': loss}

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Execution Name', 'Evaluation Name', 'Date', 'Augment Layers', 'Accuracy', 'Loss'])

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


def save_accuracy_evolution(history):
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.ylim(0, 1)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{os.getcwd()}/output/history.png')
    plt.clf()


def write_acc_avg():
    df = pd.read_csv(f'{os.getcwd()}/output/output.csv')

    accuracy_average = df.groupby("Augment Layers")["Accuracy"].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.bar(accuracy_average["Augment Layers"], accuracy_average["Accuracy"])
    plt.title('Average Accuracy by Augment Layers')
    plt.xlabel('Augment Layers')
    plt.ylabel('Average Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f'{os.getcwd()}/output/avg.png')
    plt.clf()


def write_acc_each_dataset():
    df = pd.read_csv(f"{os.getcwd()}/output/output.csv")

    for i, augment_layer in enumerate(['Baseline', 'S&P', 'DefaultAug', 'DefaultAug+S&P']):
        rows = df.loc[df['Augment Layers'] == augment_layer]

        print(f"Augment Layers: {augment_layer} Acc Avg: {rows['Accuracy'].mean()}")

        sns.barplot(x=rows['Name'], y=rows['Accuracy'])
        plt.xlabel('Dataset')
        plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig(f'{os.getcwd()}/output/barplot-{augment_layer}.png')
        plt.clf()


def write_acc_each_dataset_line():
    df = pd.read_csv(f"{os.getcwd()}/output/output.csv")

    sns.lineplot(df, x=df['Name'], y=df['Accuracy'], hue=df['Augment Layers'])
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')

    plt.xticks(rotation=45)
    plt.savefig(f'{os.getcwd()}/output/lineplot.png')
    plt.clf()