import tensorflow as tf
from tensorflow.data import Dataset
import keras_cv as keras_cv
import tensorflow_datasets as tfds
from keras import datasets, models
from sklearn.model_selection import KFold
import numpy as np
from dataset.ood_characterization import calculate_kl_divergence
from pathlib import Path
import os


class Cifar100Dataset:
    AUTOTUNE = tf.data.AUTOTUNE
    
    # CIFAR-100-C must be downloaded separately
    # Download from: https://zenodo.org/records/3555552
    # Use absolute path to avoid issues when running from different directories
    CIFAR100_C_PATH = Path(__file__).parent.parent / 'dataset' / 'CIFAR-100-C'

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
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        dataset_splits = list(enumerate(kf.split(x_train, y_train)))
        return x_train, y_train, x_test, y_test, dataset_splits

    def get(self, x, y, augmentation_layer=None):
        return self.prepare(Dataset.from_tensor_slices((x, y)), augmentation_layer=augmentation_layer)

    def get_corrupted(self, corruption_type):
        """
        Load CIFAR-100-C corrupted dataset from downloaded .npy files.
        
        CIFAR-100-C must be downloaded from:
        https://zenodo.org/records/3555552
        
        Extract to: ./dataset/CIFAR-100-C/
        
        Args:
            corruption_type: e.g., 'gaussian_noise_1', 'motion_blur_3', etc.
                Format: <corruption_name>_<severity_level>
        
        Returns:
            Prepared TensorFlow dataset
        """
        # Parse corruption type and severity
        parts = corruption_type.rsplit('_', 1)
        if len(parts) == 2:
            corruption_name = parts[0]
            severity = int(parts[1])
        else:
            raise ValueError(f"Invalid corruption type format: {corruption_type}. Expected format: 'corruption_name_severity'")
        
        # Check if CIFAR-100-C directory exists
        if not self.CIFAR100_C_PATH.exists():
            raise FileNotFoundError(
                f"\nCIFAR-100-C not found at {self.CIFAR100_C_PATH}\n"
                f"Please download CIFAR-100-C from: https://zenodo.org/records/3555552\n"
                f"Extract CIFAR-100-C.tar to: {self.CIFAR100_C_PATH.parent}\n"
            )
        
        # Load corruption .npy file
        corruption_file = self.CIFAR100_C_PATH / f'{corruption_name}.npy'
        labels_file = self.CIFAR100_C_PATH / 'labels.npy'
        
        if not corruption_file.exists():
            raise FileNotFoundError(
                f"Corruption file not found: {corruption_file}\n"
                f"Available corruptions should be in .npy format in {self.CIFAR100_C_PATH}"
            )
        
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        # Load data
        # CIFAR-100-C structure: 50,000 images (10,000 per severity level)
        # Severity levels 1-5 are stored sequentially
        images = np.load(corruption_file)
        labels = np.load(labels_file)
        
        # Extract images for the specific severity level
        # Each severity has 10,000 images
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        
        x_corrupted = images[start_idx:end_idx]
        y_corrupted = labels[start_idx:end_idx]
        
        return self.prepare(Dataset.from_tensor_slices((x_corrupted, y_corrupted)))

    def get_dataset_for_autoencoder(self, x_data, augmentation_layer=None):
        """Prepare dataset for autoencoder without rescaling (data already normalized)."""
        ds = Dataset.from_tensor_slices((x_data, x_data))
        
        if augmentation_layer:
            data_augmentation_sequential = models.Sequential(augmentation_layer)
            ds = ds.map(
                lambda x, y: (data_augmentation_sequential(x, training=True), y),
                num_parallel_calls=self.AUTOTUNE
            )
        
        ds = ds.batch(self.batch_size)
        return ds.prefetch(buffer_size=self.AUTOTUNE)

    def prepare_cifar100_kfold_for_autoencoder(self, n_splits):
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        dataset_splits = list(enumerate(kf.split(x_train)))
        return x_train, x_test, dataset_splits

    def prepare_cifar100_c_with_distances(self, encoder, corruption_type, test_ds):
        """
        Calculate KL divergence between clean and corrupted CIFAR-100 images.
        Uses feature-wise KL divergence for better sensitivity in high-dimensional latent space.
        """
        parts = corruption_type.rsplit('_', 1)
        if len(parts) == 2:
            corruption_name = parts[0]
            severity = int(parts[1])
        else:
            raise ValueError(f"Invalid corruption type: {corruption_type}")
        
        corruption_file = self.CIFAR100_C_PATH / f'{corruption_name}.npy'
        
        if not corruption_file.exists():
            raise FileNotFoundError(f"Corruption file not found: {corruption_file}")
        
        images = np.load(corruption_file)
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        x_corrupted = images[start_idx:end_idx].astype('float32') / 255.0
        
        corrupted_ds = self.get_dataset_for_autoencoder(x_corrupted)

        latent_clean = encoder.predict(test_ds)
        latent_corrupted = encoder.predict(corrupted_ds)

        return calculate_kl_divergence(latent_clean, latent_corrupted, method='feature_wise')



