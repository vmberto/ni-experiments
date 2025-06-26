import os
import tempfile
import numpy as np
import soundfile as sf
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.model_selection import KFold
from mtsa import files_train_test_split
from mtsa.features.mel import Array2Mfcc
from mtsa.utils import Wav2Array


class MIMIIDataset:
    def __init__(self, batch_size, sample_rate=16000):
        self.batch_size = batch_size
        self.sample_rate = sample_rate

    def load_audio_np(self, path):
        waveform, sr = sf.read(path, dtype='float32')
        if sr != self.sample_rate:
            raise ValueError(f"Sample rate mismatch: expected {self.sample_rate}, got {sr}")
        if waveform.ndim == 2:
            waveform = np.mean(waveform, axis=1)  # Convert stereo to mono
        return waveform

    def augment_waveform(self, waveform_np, augmentation_layers=None):
        if augmentation_layers is None:
            return waveform_np
        if callable(augmentation_layers):
            return augmentation_layers(samples=waveform_np, sample_rate=self.sample_rate)
        if isinstance(augmentation_layers, list):
            for layer in augmentation_layers:
                waveform_np = layer(samples=waveform_np, sample_rate=self.sample_rate)
            return waveform_np
        return waveform_np

    def augment_and_save_to_temp_wavs(self, file_paths, augmentation_layers=None, save_dir=None):
        if save_dir is None:
            save_dir = tempfile.mkdtemp()

        augmented_file_paths = []

        for path in tqdm(file_paths, desc="Augmenting WAVs"):
            try:
                waveform = self.load_audio_np(path)
                augmented_waveform = self.augment_waveform(waveform, augmentation_layers)

                new_filename = os.path.join(save_dir, f"aug_{os.path.basename(path)}")
                sf.write(new_filename, augmented_waveform, samplerate=self.sample_rate)
                augmented_file_paths.append(new_filename)

            except Exception as e:
                print(f"Failed to process {path}: {e}")

        return augmented_file_paths

    def prepare(self, x_paths, y_labels, shuffle=False, augmentation_layer=None):
        if shuffle:
            combined = list(zip(x_paths, y_labels))
            np.random.shuffle(combined)
            x_paths, y_labels = zip(*combined)

        if augmentation_layer is not None:
            x_paths = self.augment_and_save_to_temp_wavs(x_paths, augmentation_layers=augmentation_layer)

        return list(x_paths), list(y_labels)

    def get_kfold_splits(self, n_splits, machine):
        x_train, x_test, y_train, y_test = files_train_test_split(
            f'{os.getcwd()}/dataset/mimii_dataset/{machine}/id_00'
        )
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        dataset_splits = list(enumerate(kf.split(x_train, y_train)))
        return x_train, y_train, x_test, y_test, dataset_splits

    def get(self, x_paths, y_labels, shuffle=False, augmentation_layer=None):
        return self.prepare(x_paths, y_labels, shuffle=shuffle, augmentation_layer=augmentation_layer)

    def get_ood_dataset(self, machine, type):
        _, x_test, _, y_test = files_train_test_split(
            f'{os.getcwd()}/dataset/mimii_dataset/{machine}/{type}'
        )
        return x_test, y_test