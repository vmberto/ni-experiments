import gdown
import zipfile
import os
from tensorflow.data import Dataset
from sklearn.model_selection import KFold
import keras_cv as keras_cv
import tensorflow as tf
import pandas as pd
from experiments_config import INPUT_SHAPE, BATCH_SIZE
AUTOTUNE = tf.data.AUTOTUNE


DATASET_URL = "https://drive.google.com/uc?id=1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86"
TRAIN_LABELS_URL = "https://drive.google.com/uc?id=1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH"
VAL_LABELS_URL = "https://drive.google.com/uc?id=1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH"

DATASET_OUTPUT = "fairface-dataset.zip"
EXTRACT_DIR = "dataset/fairface"
TRAIN_LABELS_OUTPUT = "dataset/fairface/fairface_train_labels.csv"
VAL_LABELS_OUTPUT = "dataset/fairface/fairface_val_labels.csv"


def __load_and_preprocess_image(path, label, target_size=(72, 72)):
    """Loads and preprocesses an image."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1] range
    return img, label


def __create_dataset(image_paths, labels, data_augmentation=None):
    """Creates a tf.data.Dataset from image paths and labels."""
    ds = Dataset.from_tensor_slices((image_paths, labels))
    target_size = (72, 72)

    ds = ds.shuffle(buffer_size=len(image_paths))

    ds = ds.map(lambda path, label: __load_and_preprocess_image(path, label, target_size),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    if data_augmentation:
        data_augmentation_sequential = tf.keras.Sequential(data_augmentation)
        ds = ds.map(lambda x, y: (data_augmentation_sequential(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)


def __prepare(ds, shuffle=False, data_augmentation=None):

    resize_and_rescale = tf.keras.Sequential([
        keras_cv.layers.Resizing(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        keras_cv.layers.Rescaling(1. / 255)
    ])

    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

    ds = ds.map(lambda path, label: __load_and_preprocess_image(path, label),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(BATCH_SIZE)

    if data_augmentation:
        data_augmentation_sequential = tf.keras.Sequential(data_augmentation)
        ds = ds.map(lambda x, y: (data_augmentation_sequential(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)


def __download_file(url, output):
    """Download a file from Google Drive."""
    gdown.download(url, output, quiet=False)


def __check_and_download(url, output):
    """Check if a file exists, and if not, download it."""
    if not os.path.exists(output):
        print(f"{output} not found locally. Downloading now...")
        __download_file(url, output)
    else:
        print(f"{output} already exists. Skipping download.")


def __download_and_extract_dataset():
    """Download and extract the FairFace dataset if not already done."""
    if not os.path.exists(EXTRACT_DIR):
        __check_and_download(DATASET_URL, DATASET_OUTPUT)
        with zipfile.ZipFile(DATASET_OUTPUT, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        os.remove(DATASET_OUTPUT)
    else:
        print("FairFace dataset already exists. Skipping download.")


def __download_labels():
    """Download the train and validation labels."""
    __check_and_download(TRAIN_LABELS_URL, TRAIN_LABELS_OUTPUT)
    __check_and_download(VAL_LABELS_URL, VAL_LABELS_OUTPUT)


def __load_fairface_data():
    """Load and return the FairFace data as (x_train, y_train), (x_val, y_val)."""
    train_data = pd.read_csv(TRAIN_LABELS_OUTPUT, delimiter=',')
    val_data = pd.read_csv(VAL_LABELS_OUTPUT, delimiter=',')

    # Define race to integer mapping
    race_mapping = {
        'White': 0,
        'Black': 1,
        'Asian': 2,
        'Indian': 3,
        'Middle Eastern': 4,
        'Latino': 5
    }

    x_train = train_data['file'].apply(lambda x: os.path.join(EXTRACT_DIR, x)).values
    y_train = train_data['race'].map(race_mapping).values  # Convert to integers

    x_val = val_data['file'].apply(lambda x: os.path.join(EXTRACT_DIR, x)).values
    y_val = val_data['race'].map(race_mapping).values  # Convert to integers

    return (x_train, y_train), (x_val, y_val)


def __prepare_fairface_dataset():
    """Main function to download, extract, and load the FairFace dataset."""
    __download_and_extract_dataset()
    __download_labels()
    return __load_fairface_data()


def get_fairface_kfold_splits(n_splits):
    (x_train, y_train), (x_test, y_test) = __prepare_fairface_dataset()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    dataset_splits = list(enumerate(kf.split(x_train, y_train)))

    return x_train, y_train, x_test, y_test, dataset_splits


def get_fairface_dataset(x, y, data_augmentation=None):
    dataset = __create_dataset(x, y, data_augmentation=data_augmentation)
    return dataset
