from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.functions import extract_preprocessing_layer_names
import os


def write_evaluation_result(evaluation_name, aug_layers, loss, acc):
    csv_file = f'{os.getcwd()}/output/output.csv'
    augmentation_layers = extract_preprocessing_layer_names(aug_layers)
    new_line = {'Name': evaluation_name, 'Date': datetime.now(), 'Augment Layers': augmentation_layers, 'Accuracy': acc, 'Loss': loss}

    # Try to read existing CSV file, if it doesn't exist create new DataFrame
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Name', 'Date', 'Accuracy', 'Loss'])

    # Append the new line to the DataFrame
    df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)

    # Write the DataFrame back to the CSV file
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