import multiprocessing
from resnet_baseline import run as resnet_baseline
from resnet_sp import run as resnet_sp
from resnet_gaussian import run as resnet_gaussian
from resnet_defaultaug import run as resnet_defaultaug
from resnet_defaultaug_sp import run as resnet_defaultaugsp
from xception_baseline import run as xception_baseline
from xception_sp import run as xception_sp
from xception_gaussian import run as xception_gaussian
from xception_defaultaug import run as xception_defaultaug
from xception_defaultaug_sp import run as xception_defaultaugsp


def run():
    resnet_baseline()
    resnet_sp()
    resnet_gaussian()
    resnet_defaultaug()
    resnet_defaultaugsp()
    xception_baseline()
    xception_sp()
    xception_gaussian()
    xception_defaultaug()
    xception_defaultaugsp()


if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()
    print("finished")
