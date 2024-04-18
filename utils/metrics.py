from datetime import datetime
import pandas as pd
import os


def write_acc_loss_result(corruption_type, execution_name, loss, acc):
    csv_file = f'{os.getcwd()}/output/output.csv'

    new_line = {
        'Execution Name': execution_name,
        'Corruption Type': corruption_type,
        'Date': datetime.now(),
        'Accuracy': acc,
        'Loss': loss
    }

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Execution Name', 'Corruption Type', 'Date', 'Augment Layers', 'Accuracy', 'Loss'])

    df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)

    df.to_csv(csv_file, index=False)


def write_fscore_result(corruption_type, approach_name, model_name, fscore, training_time):
    csv_file = f'{os.getcwd()}/output/output.csv'

    new_line = {
        'Approach': approach_name,
        'Model': model_name,
        'Corruption Type': corruption_type,
        'Date': datetime.now(),
        'Training Time': training_time,
        'Fscore': fscore,
    }

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=['Approach', 'Model', 'Corruption Type', 'Date', 'Training Time', 'Fscore'])

    df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)

    df.to_csv(csv_file, index=False)
