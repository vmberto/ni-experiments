from datetime import datetime
import pandas as pd
import os
import json
from lib.functions import clean_string


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


def confusion_matrix_format(y_actual, y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_pred)):
        if y_actual[i]==y_pred[i]==1:
           tp += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
           fp += 1
        if y_actual[i]==y_pred[i]==0:
           tn += 1
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
           fn += 1

    return tp, fp, tn, fn


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


def write_fscore_result(
        evauation_set, approach_name, model_name, report, conf_matrix, training_time, fold_number, loss, acc
):
    csv_file = f'{os.getcwd()}/output/output.csv'

    single_dict_report = convert_dict(report)
    tp, fp, tn, fn = conf_matrix

    new_line = {
        'approach': approach_name,
        'model': model_name,
        'evaluation_set': clean_string(evauation_set),
        'date_finished': datetime.now(),
        'training_time': training_time,
        'fold': fold_number,
        'accuracy': acc,
        'loss': loss,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        **single_dict_report,
    }

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=[
                'approach', 'model', 'evaluation_set', 'date_finished', 'training_time', 'fold', 'tn', 'fp', 'fn', 'tp'
            ])

    df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)

    df.to_csv(csv_file, index=False)


def write_fscore_result_json(corruption_type, approach_name, model_name, fscore, training_time, y_pred):
    json_file = f'{os.getcwd()}/output/output.json'

    new_line = {
        'Approach': approach_name,
        'Model': model_name,
        'Corruption Type': corruption_type,
        'Date': str(datetime.now()),
        'Training Time': training_time,
        'Fscore': fscore,
        'Predictions': y_pred.tolist(),
    }

    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []

    data.append(new_line)

    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)
