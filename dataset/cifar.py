import tensorflow as tf
from tensorflow.data import Dataset
import numpy as np
import keras_cv as keras_cv
import tensorflow_datasets as tfds
from keras.datasets import cifar10
from sklearn.model_selection import KFold
from experiments_config import INPUT_SHAPE

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128


def prepare(ds, shuffle=False, data_augmentation=None):

    resize_and_rescale = tf.keras.Sequential([
        keras_cv.layers.Resizing(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        keras_cv.layers.Rescaling(1. / 255)
    ])

    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(BATCH_SIZE)

    if data_augmentation:
        data_augmentation_sequential = tf.keras.Sequential(data_augmentation)
        ds = ds.map(lambda x, y: (data_augmentation_sequential(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)


def get_cifar10_kfold_splits(n_splits):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    dataset_splits = list(enumerate(kf.split(x_train, y_train)))

    return x_train, y_train, x_test, y_test, dataset_splits


def get_cifar10_dataset(x, y, aug_layers=None):
    dataset = prepare(Dataset.from_tensor_slices((x, y)), data_augmentation=aug_layers)
    return dataset


def get_cifar10_corrupted(corruption_type):
    cifar_10_c = tfds.load(f"cifar10_corrupted/{corruption_type}", split="test", as_supervised=True)

    cifar_10_c = prepare(cifar_10_c)

    return cifar_10_c
