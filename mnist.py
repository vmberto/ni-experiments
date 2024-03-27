import tensorflow

import tensorflow as tf
import tensorflow_datasets as tfds
from keras.datasets import mnist
import numpy as np


def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    y_train = y_train.astype(np.float64)

    x_test = x_test / 255.0
    y_test = y_test.astype(np.float64)

    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return (x_train, y_train_one_hot), (x_test, y_test_one_hot)


def get_mnist_corrupted():
    # Load the MNIST Corrupted dataset
    ds = tfds.load('mnist_corrupted/motion_blur', split='train', shuffle_files=True)

    # Initialize empty lists to store features and labels
    x_data = []
    y_data = []

    # Iterate through the dataset and extract features and labels
    for example in ds:
        image, label = example["image"], example["label"]
        # Normalize the image data to range [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float64)

        image = tf.expand_dims(image, axis=-1)
        x_data.append(image.numpy())  # Append the image data to x_data
        y_data.append(label.numpy())  # Append the label to y_data

    # Convert lists to numpy arrays
    x_data = np.array(x_data)
    y_data = tf.keras.utils.to_categorical(np.array(y_data), num_classes=10)

    return x_data, y_data
