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

    def __init__(self, input_shape=(72, 72, 3), batch_size=128):
        self.input_shape = input_shape
        self.batch_size = batch_size

    def _resize_and_rescale(self):
        """Resizing and rescaling layers for preprocessing."""
        return models.Sequential([
            keras_cv.layers.Resizing(self.input_shape[0], self.input_shape[1]),
            keras_cv.layers.Rescaling(1. / 255)
        ])

    def mixed_preprocess(self, image, label, augmentation_pipelines):
        image = tf.image.convert_image_dtype(image, tf.float32)
        random_index = tf.random.uniform((), minval=0, maxval=len(augmentation_pipelines), dtype=tf.int32)

        def apply_none():
            return tf.identity(image)

        def apply_augmentation(pipeline):
            return pipeline(image)

        augmented_image = tf.case([
            (tf.equal(random_index, i),
             lambda p=pipeline: apply_none() if p is None
             else apply_augmentation(p if isinstance(p, models.Sequential)
                                     else models.Sequential(p)))
            for i, pipeline in enumerate(augmentation_pipelines)
        ], exclusive=True)

        return augmented_image, label

    def prepare(self, ds, shuffle=False, data_augmentation=None, mixed=False):
        resize_and_rescale = self._resize_and_rescale()

        ds = ds.map(
            lambda x, y: (resize_and_rescale(x), y),
            num_parallel_calls=self.AUTOTUNE
        )

        if shuffle:
            ds = ds.shuffle(1000)

        if data_augmentation and not mixed:
            data_augmentation_sequential = models.Sequential(data_augmentation)
            ds = ds.map(
                lambda x, y: (data_augmentation_sequential(x, training=True), y),
                num_parallel_calls=self.AUTOTUNE
            )

        if data_augmentation and mixed:
            ds = ds.map(
                lambda x, y: self.mixed_preprocess(x, y, data_augmentation),
                num_parallel_calls=self.AUTOTUNE
            )

        ds = ds.batch(self.batch_size)

        return ds.prefetch(buffer_size=self.AUTOTUNE)

    def get_kfold_splits(self, n_splits):
        """Generate K-Fold splits for CIFAR-10."""
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        dataset_splits = list(enumerate(kf.split(x_train, y_train)))

        return x_train, y_train, x_test, y_test, dataset_splits

    def get(self, x, y, data_augmentation=None, mixed=False):
        """Get a processed CIFAR-10 dataset."""
        return self.prepare(Dataset.from_tensor_slices((x, y)), data_augmentation=data_augmentation, mixed=mixed)

    def get_corrupted(self, corruption_type):
        """Load and prepare a corrupted CIFAR-10 dataset."""
        cifar_10_c = tfds.load(f"cifar10_corrupted/{corruption_type}", split="test", as_supervised=True)
        return self.prepare(cifar_10_c)

    def get_dataset_for_autoencoder(self, x_data):
        """Prepare CIFAR-10 data for autoencoder training."""
        return self.prepare(tf.data.Dataset.from_tensor_slices((x_data, x_data)))

    def prepare_cifar10_kfold_for_autoencoder(self, n_splits):
        """Prepare K-Fold CIFAR-10 splits for autoencoder."""
        dataset_train = tfds.load('cifar10', split='train', as_supervised=True)
        dataset_test = tfds.load('cifar10', split='test', as_supervised=True)

        x_train = np.array([image for image, label in tfds.as_numpy(dataset_train)])
        x_test = np.array([image for image, label in tfds.as_numpy(dataset_test)])

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        dataset_splits = list(enumerate(kf.split(x_train)))

        return x_train, x_test, dataset_splits

    def prepare_cifar10_c_with_distances(self, encoder, corruption_type, test_ds):
        """Calculate KL divergence between clean and corrupted data."""
        dataset = tfds.load(f'cifar10_corrupted/{corruption_type}', split='test', as_supervised=True)
        x_corrupted = np.array([image for image, label in tfds.as_numpy(dataset)])
        x_corrupted = x_corrupted.astype('float32') / 255.0
        corrupted_ds = self.get_dataset_for_autoencoder(x_corrupted)

        latent_clean = encoder.predict(test_ds)
        latent_corrupted = encoder.predict(corrupted_ds)

        kl_div = calculate_kl_divergence(latent_clean, latent_corrupted)

        return kl_div