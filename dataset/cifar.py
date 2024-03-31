import tensorflow as tf
import tensorflow_datasets as tfds
from keras.datasets import cifar10
from utils.images import save_img_examples

AUTO = tf.data.AUTOTUNE
IMG_SIZE = 72


def get_cifar10(batch_size, aug_layers):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    simple_aug = tf.keras.Sequential(aug_layers)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(batch_size * 100)
        .batch(batch_size)
        .map(lambda x, y: (simple_aug(x), y), num_parallel_calls=AUTO)
        .prefetch(AUTO)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .map(
            lambda x, y: (tf.image.resize(x, (IMG_SIZE, IMG_SIZE)), y),
            num_parallel_calls=AUTO,
        )
        .prefetch(AUTO)
    )

    save_img_examples(train_ds)

    return train_ds, val_ds


def get_cifar10_corrupted(batch_size, corruption_type):
    cifar_10_c = tfds.load(f"cifar10_corrupted/{corruption_type}", split="test", as_supervised=True)

    cifar_10_c = (
        cifar_10_c
        .shuffle(batch_size * 100)
        .batch(batch_size)
        .map(
            lambda x, y: (tf.image.resize(x, (IMG_SIZE, IMG_SIZE)), y),
            num_parallel_calls=AUTO,
        )
        .prefetch(AUTO)
    )

    return cifar_10_c
