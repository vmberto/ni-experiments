import tensorflow as tf
from tensorflow.data import Dataset
from sklearn.model_selection import KFold
import numpy as np
import tensorflow_datasets as tfds
from keras import layers
from keras import models
import pandas as pd


class AGNewsDataset:
    AUTOTUNE = tf.data.AUTOTUNE

    def __init__(self, max_sequence_length=128, vocab_size=20000, batch_size=32):
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.text_vectorizer = layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.max_sequence_length,
        )

    def preprocess_text(self, text):
        print('preprocessing')
        return self.text_vectorizer(text)

    def prepare_text_vectorizer(self, ds):
        print('adapting')
        text_ds = ds.map(lambda x, y: x).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.text_vectorizer.adapt(text_ds)

    def mixed_preprocess(self, text, label, augmentation_pipeline):
        random_index = tf.random.uniform((), minval=0, maxval=len(augmentation_pipeline), dtype=tf.int32)

        def apply_none():
            return tf.identity(text)

        def apply_augmentation(pipeline):
            return pipeline(text)

        augmented_text = tf.case([
            (tf.equal(random_index, i),
             lambda p=pipeline: apply_none() if p is None
             else apply_augmentation(p if isinstance(p, models.Sequential)
                                     else models.Sequential(p)))
            for i, pipeline in enumerate(augmentation_pipeline)
        ], exclusive=True)

        return augmented_text, label

    def prepare(self, ds, shuffle=False, data_augmentation=None, mixed=False):
        ds = ds.map(
            lambda x, y: (self.preprocess_text(x), y),
            num_parallel_calls=self.AUTOTUNE,
        )

        if shuffle:
            ds = ds.shuffle(1000)

        if data_augmentation and not mixed:
            data_augmentation_sequential = models.Sequential(data_augmentation)
            ds = ds.map(
                lambda x, y: (data_augmentation_sequential(x, training=True), y),
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
        if array and isinstance(array[0], list):
            first_element = array[0]
            if first_element and "nlpaug" in str(type(first_element[0])):
                nlpaug_augmentations = array.pop(0)
                return nlpaug_augmentations
        return []

    def get(self, X, y, data_augmentation=None, mixed=False):
        da_layers = data_augmentation.copy() if data_augmentation is not None else data_augmentation
        nlpaug_pipeline = self.extract_nlpaug_augmentations(da_layers)
        augmented_X = []
        if nlpaug_pipeline:
            print('augmenting')
            for text in X:
                try:
                    augmented_text = nlpaug_pipeline.augment(text)
                    augmented_X.append(*augmented_text)
                except ValueError as _e:
                    augmented_X.append(text)

        return self.prepare(Dataset.from_tensor_slices((augmented_X if len(augmented_X) > 0 else X, y)), data_augmentation=da_layers, mixed=mixed)

    def get_corrupted(self, corruption_name):
        filepath = f'./dataset/agnewsdataset-c/ag_news_{corruption_name}.csv'

        df = pd.read_csv(filepath, encoding='utf-8')

        labels = df['label'].tolist()
        texts = [str(text).encode('utf-8').decode('utf-8') for text in df['text'].tolist()]

        return self.prepare(tf.data.Dataset.from_tensor_slices((texts, labels)))
