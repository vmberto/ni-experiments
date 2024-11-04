import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential
import keras_cv as keras_cv
import tensorflow_datasets as tfds
from keras.datasets import cifar10
from sklearn.model_selection import KFold
from experiments_config import INPUT_SHAPE, BATCH_SIZE
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE


def mixed_preprocess(image, label, augmentation_pipelines):
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
    # Generate a random integer index
    random_index = tf.random.uniform((), minval=0, maxval=len(augmentation_pipelines), dtype=tf.int32)

    # Define a function to handle the None case
    def apply_none():
        return tf.identity(image)

    # Define a function to apply regular augmentation pipeline
    def apply_augmentation(pipeline):
        return pipeline(image)

    # Use tf.case to handle different pipeline types
    augmented_image = tf.case([
        (tf.equal(random_index, i),
         lambda p=pipeline: apply_none() if p is None
         else apply_augmentation(p if isinstance(p, tf.keras.Sequential)
                                 else tf.keras.Sequential(p)))
        for i, pipeline in enumerate(augmentation_pipelines)
    ], exclusive=True)

    return augmented_image, label


def prepare(ds, shuffle=False, data_augmentation=None, mixed=False):
    resize_and_rescale = tf.keras.Sequential([
        keras_cv.layers.Resizing(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        keras_cv.layers.Rescaling(1. / 255)
    ])

    ds = ds.map(
        lambda x, y: (resize_and_rescale(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        ds = ds.shuffle(1000)


    if data_augmentation and not mixed:
        data_augmentation_sequential = tf.keras.Sequential(data_augmentation)
        ds = ds.map(
            lambda x, y: (data_augmentation_sequential(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    if data_augmentation and mixed:
        ds = ds.map(
            lambda x, y: (mixed_preprocess(x, y, data_augmentation)),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.batch(BATCH_SIZE)

    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def get_cifar10_kfold_splits(n_splits):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    dataset_splits = list(enumerate(kf.split(x_train, y_train)))

    return x_train, y_train, x_test, y_test, dataset_splits


def get_cifar10_dataset(x, y, data_augmentation=None, mixed=False):
    dataset = prepare(Dataset.from_tensor_slices((x, y)), data_augmentation=data_augmentation, mixed=mixed)
    return dataset


def get_cifar10_corrupted(corruption_type):
    cifar_10_c = tfds.load(f"cifar10_corrupted/{corruption_type}", split="test", as_supervised=True)

    cifar_10_c = prepare(cifar_10_c)

    return cifar_10_c
