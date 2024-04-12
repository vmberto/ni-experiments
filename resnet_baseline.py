from models.resnet import ResNet50Model
from experiment import experiment
import multiprocessing

INPUT_SHAPE = (72, 72, 3)


def run():
    approach_name = 'Baseline'

    data_augmentation_layers = []

    resnet = ResNet50Model(input_shape=INPUT_SHAPE, approach_name=approach_name)

    experiment(resnet, data_augmentation_layers)


if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()
    print("finished")
