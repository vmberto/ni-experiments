import multiprocessing
from resnet_baseline import run as baseline
from resnet_sp import run as sp
from resnet_defaultaug import run as defaultaug
from resnet_defaultaug_sp import run as defaultaugsp


def run():
    baseline()
    sp()
    defaultaug()
    defaultaugsp()


if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()
    print("finished")
