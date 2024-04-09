import multiprocessing
from xception_baseline import run as baseline
from xception_sp import run as sp
from xception_defaultaug import run as defaultaug
from xception_defaultaug_sp import run as defaultaugsp


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
