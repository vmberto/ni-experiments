from keras.callbacks import EarlyStopping
from models.resnet import ResNet50Model
from dataset.cifar import get_cifar10, get_cifar10_corrupted
from layers.salt_and_pepper import RandomSaltAndPepper
import keras_cv.layers as layers
import tensorflow as tf
from utils.configs import set_memory_growth
from utils.metrics import write_acc_avg, write_acc_each_dataset
from utils.consts import CORRUPTIONS_TYPES
from keras_cv.core import NormalFactorSampler

BATCH_SIZE = 128
IMG_SIZE = 72
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
SEED = 42


def experiment(execution_name, model, data_augmentation_layers):
    set_memory_growth(tf)

    train_ds, val_ds = get_cifar10(BATCH_SIZE, data_augmentation_layers)

    model.fit(
        train_ds,
        val_dataset=val_ds,
        epochs=100,
        callbacks=[
            EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)
        ]
    )

    for corruption in CORRUPTIONS_TYPES:
        model.evaluate(
            get_cifar10_corrupted(BATCH_SIZE, corruption),
            f'cifar10/{corruption}',
            data_augmentation_layers,
            execution_name,
        )

    write_acc_avg()
    write_acc_each_dataset()
