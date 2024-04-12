from models.xception import XceptionModel
from experiment import experiment
import multiprocessing
import os

INPUT_SHAPE = (72, 72, 3)


def run():
    execution_name = 'Baseline'

    data_augmentation_layers = []

    xception = XceptionModel(input_shape=INPUT_SHAPE, approach_name=execution_name)

    experiment(xception, data_augmentation_layers)


if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()
    print("finished")
