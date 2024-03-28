import tensorflow as tf
import tensorflow_datasets as tfds
from keras.datasets import cifar10
import numpy as np


def preprocess_image(image, label):
    image = tf.image.resize(image, (32, 32))
    image = image / 255.0
    image = tf.cast(image, 'float32')
    return image, label


def preprocess(image, label):
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.1, dtype=tf.float32)
    image_noisy = tf.add(image, noise)

    return (image, label), (image_noisy, label)


def get_cifar10(batch_size):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=10)

    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
                .batch(batch_size)
                .map(preprocess_image)
                .prefetch(tf.data.experimental.AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test_one_hot))
              .batch(batch_size).map(preprocess_image))

    return train_ds, val_ds


def get_cifar10_corrupted(batch_size):
    cifar_10_c = tfds.load("cifar10", split="test", as_supervised=True, batch_size=-1)

    x, y = tfds.as_numpy(cifar_10_c)

    y = tf.keras.utils.to_categorical(y)

    return x, y
