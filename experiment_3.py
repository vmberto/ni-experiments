"""

RESNET50, Batch-size 128, 20 epochs
CIFAR-10: Training 50000 images (train 45000, val 5000), Testing 10000 images
Random Noise: 0, 0.2, 0.4, 0.6

"""
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import multiprocessing
from models.resnet import ResNet50Model
from dataset.cifar import get_cifar10, get_cifar10_corrupted
from utils.images import save_img_examples

BATCH_SIZE = 32


def run():
    train_ds, val_ds = get_cifar10(BATCH_SIZE)
    x, y = get_cifar10_corrupted(BATCH_SIZE)

    resnet = ResNet50Model()

    resnet.compile()

    resnet.fit(
        train_ds,
        val_dataset=val_ds,
        epochs=30,
        callbacks=[EarlyStopping(patience=4, monitor='val_loss', verbose=1)]
    )

    resnet.evaluate(x, y)
    # resnet.predict(eval_ds, eval_labels)


if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()
    print("finished")