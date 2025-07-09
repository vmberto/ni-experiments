import tensorflow as tf
from tensorflow.data import Dataset
from sklearn.model_selection import KFold
import numpy as np
import tensorflow_datasets as tfds
from keras import layers, models
import pandas as pd
from lib.metrics import calculate_kl_divergence
import os


class AGNewsDataset:
    AUTOTUNE = tf.data.AUTOTUNE

    def __init__(self, max_sequence_length=128, vocab_size=20000, batch_size=128):
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.text_vectorizer = layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.max_sequence_length,
        )

    def preprocess_text(self, text):
        return self.text_vectorizer(text)

    def prepare_text_vectorizer(self, ds, vocab_path="agnews_vocab.txt"):
        if os.path.exists(vocab_path):
            print("🔁 Loading cached vocabulary...")
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = f.read().splitlines()
            self.text_vectorizer.set_vocabulary(vocab)
        else:
            print("🧠 Adapting vectorizer...")
            text_ds = ds.map(lambda x, y: x).prefetch(self.AUTOTUNE)
            self.text_vectorizer.adapt(text_ds)
            # Save vocabulary
            vocab = self.text_vectorizer.get_vocabulary()
            with open(vocab_path, "w", encoding="utf-8") as f:
                f.write("\n".join(vocab))
            print("✅ Vocabulary saved.")

    def prepare(self, ds, shuffle=False, data_augmentation=None, autoencoder=False):
        if autoencoder:
            ds = ds.map(
                lambda x, y: (self.preprocess_text(x), self.preprocess_text(x)),
                num_parallel_calls=self.AUTOTUNE,
            ).cache()
        else:
            ds = ds.map(
                lambda x, y: (self.preprocess_text(x), self.preprocess_text(y)),
                num_parallel_calls=self.AUTOTUNE,
            )

        if shuffle:
            ds = ds.shuffle(1000)

        if data_augmentation:
            aug_seq = models.Sequential(data_augmentation)
            ds = ds.map(
                lambda x, y: (aug_seq(x, training=True), y),
                num_parallel_calls=self.AUTOTUNE
            )

        ds = ds.batch(self.batch_size)
        return ds.prefetch(buffer_size=self.AUTOTUNE)

    def load_and_preprocess(self):
        ds_train, ds_test = tfds.load("ag_news_subset", split=["train", "test"], as_supervised=True)
        self.prepare_text_vectorizer(ds_train)
        return ds_train, ds_test

    def get_kfold_splits(self, n_splits):
        ds_train, ds_test = self.load_and_preprocess()

        train_texts, train_labels = [], []
        for text, label in tfds.as_numpy(ds_train):
            train_texts.append(text.decode("utf-8"))
            train_labels.append(label)

        test_texts, test_labels = [], []
        for text, label in tfds.as_numpy(ds_test):
            test_texts.append(text.decode("utf-8"))
            test_labels.append(label)

        train_texts = np.array(train_texts)
        train_labels = np.array(train_labels)
        test_texts = np.array(test_texts)
        test_labels = np.array(test_labels)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        dataset_splits = list(enumerate(kf.split(train_texts, train_labels)))

        return train_texts, train_labels, test_texts, test_labels, dataset_splits

    def extract_nlpaug_augmentations(self, array):
        if not array:
            return []

        if isinstance(array[0], list):
            first_element = array[0]
            if first_element and "nlpaug" in str(type(first_element[0])):
                return array.pop(0)
        elif "nlpaug" in str(type(array[0])):
            return [array.pop(0)]

        return []

    def get(self, X, y, data_augmentation=None):
        da_layers = data_augmentation.copy() if data_augmentation else None
        nlpaug_pipeline = self.extract_nlpaug_augmentations(da_layers)
        augmented_X = []

        if nlpaug_pipeline:
            print('Applying NLP augmentations...')
            for text in X:
                try:
                    augmented_text = nlpaug_pipeline.augment(text)
                    augmented_X.append(augmented_text[0])
                except ValueError:
                    augmented_X.append(text)

        return self.prepare(
            Dataset.from_tensor_slices((augmented_X if augmented_X else X, y)),
            data_augmentation=da_layers
        )

    def get_corrupted(self, corruption_name):
        filepath = f'./dataset/agnewsdataset-c/ag_news_{corruption_name}.csv'
        df = pd.read_csv(filepath, encoding='utf-8')
        labels = df['label'].tolist()
        texts = [str(text).encode('utf-8').decode('utf-8') for text in df['text'].tolist()]
        return self.prepare(Dataset.from_tensor_slices((texts, labels)))

    def get_dataset_for_autoencoder(self, x_data):
        return self.prepare(Dataset.from_tensor_slices((x_data, x_data)), autoencoder=True)

    def prepare_agnews_c_with_distances(self, encoder, corruption_type, test_ds):
        filepath = f'../dataset/agnewsdataset-c/ag_news_{corruption_type}.csv'
        df = pd.read_csv(filepath, encoding='utf-8')
        texts = [str(text).encode('utf-8').decode('utf-8') for text in df['text'].tolist()]
        corrupted_ds = self.prepare(Dataset.from_tensor_slices((texts, texts)), autoencoder=True)

        latent_clean = encoder.predict(test_ds)
        latent_corrupted = encoder.predict(corrupted_ds)
        return calculate_kl_divergence(latent_clean, latent_corrupted)

    def get_testset_for_autoencoder(self):
        _, ds_test = self.load_and_preprocess()

        test_texts, test_labels = [], []
        for text, label in tfds.as_numpy(ds_test):
            test_texts.append(text.decode("utf-8"))
        test_texts = np.array(test_texts)

        return test_texts