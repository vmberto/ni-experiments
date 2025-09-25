import tensorflow as tf
from tensorflow.data import Dataset
import keras_cv as keras_cv
import tensorflow_datasets as tfds
from keras import datasets, models
from sklearn.model_selection import KFold
import numpy as np
from lib.metrics import calculate_kl_divergence


class Cifar10Dataset:
    AUTOTUNE = tf.data.AUTOTUNE

    def __init__(self, input_shape, batch_size):
        self.input_shape = input_shape
        self.batch_size = batch_size

        # Define once
        self.resize_and_rescale = models.Sequential([
            keras_cv.layers.Resizing(input_shape[0], input_shape[1]),
            keras_cv.layers.Rescaling(1. / 255)
        ])

    def prepare(self, ds, shuffle=False, augmentation_layer=None):
        ds = ds.map(
            lambda x, y: (self.resize_and_rescale(x), y),
            num_parallel_calls=self.AUTOTUNE
        )

        if shuffle:
            ds = ds.shuffle(1000)

        if augmentation_layer:
            data_augmentation_sequential = models.Sequential(augmentation_layer)
            ds = ds.map(
                lambda x, y: (data_augmentation_sequential(x, training=True), y),
                num_parallel_calls=self.AUTOTUNE
            )

        ds = ds.batch(self.batch_size)
        return ds.prefetch(buffer_size=self.AUTOTUNE)

    def get_kfold_splits(self, n_splits):
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        dataset_splits = list(enumerate(kf.split(x_train, y_train)))
        return x_train, y_train, x_test, y_test, dataset_splits

    def get(self, x, y, augmentation_layer=None):
        return self.prepare(Dataset.from_tensor_slices((x, y)), augmentation_layer=augmentation_layer)

    def get_corrupted(self, corruption_type):
        cifar_10_c = tfds.load(f"cifar10_corrupted/{corruption_type}", split="test", as_supervised=True)
        return self.prepare(cifar_10_c)

    def get_dataset_for_autoencoder(self, x_data, augmentation_layer=None):
        return self.prepare(Dataset.from_tensor_slices((x_data, x_data)), augmentation_layer=augmentation_layer)

    def prepare_cifar10_kfold_for_autoencoder(self, n_splits):
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        dataset_splits = list(enumerate(kf.split(x_train)))
        return x_train, x_test, dataset_splits

    def prepare_cifar10_c_with_distances(self, encoder, corruption_type, test_ds):
        dataset = tfds.load(f'cifar10_corrupted/{corruption_type}', split='test', as_supervised=True)
        x_corrupted = np.array([image for image, _ in tfds.as_numpy(dataset)])
        x_corrupted = x_corrupted.astype('float32') / 255.0
        corrupted_ds = self.get_dataset_for_autoencoder(x_corrupted)

        latent_clean = encoder.predict(test_ds)
        latent_corrupted = encoder.predict(corrupted_ds)

        print(f"latent_clean shape: {latent_clean.shape}")
        print(f"latent_corrupted shape: {latent_corrupted.shape}")

        return calculate_kl_divergence(latent_clean, latent_corrupted)
