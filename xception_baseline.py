from models.xception import XceptionModel
from experiment import experiment
import multiprocessing
import os

INPUT_SHAPE = (72, 72, 3)


def run():
    execution_name = os.path.splitext(os.path.basename(__file__))[0]
    print(execution_name)
    data_augmentation_layers = []

    xception = XceptionModel(input_shape=INPUT_SHAPE, execution_name=execution_name)

    experiment(execution_name, xception, data_augmentation_layers)


if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()
    print("finished")
