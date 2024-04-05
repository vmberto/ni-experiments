import tensorflow as tf
import tensorflow_datasets as tfds
from keras.datasets import cifar10

AUTOTUNE = tf.data.AUTOTUNE


def prepare(ds, shuffle=False, batch_size=128, data_augmentation=None, img_size=72):

    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.Resizing(img_size, img_size),
        tf.keras.layers.Rescaling(1. / 255)
    ])

    data_augmentation_sequential = tf.keras.Sequential(data_augmentation)

    ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(batch_size)

    if data_augmentation:
        ds = ds.map(lambda x, y: (data_augmentation_sequential(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)


def get_cifar10(batch_size, aug_layers):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_ds = prepare(train_ds, shuffle=True, batch_size=batch_size, data_augmentation=aug_layers)
    val_ds = prepare(val_ds, batch_size=batch_size)

    return train_ds, val_ds


def get_cifar10_corrupted(batch_size, corruption_type):
    cifar_10_c = tfds.load(f"cifar10_corrupted/{corruption_type}", split="test", as_supervised=True)

    cifar_10_c = prepare(cifar_10_c, batch_size=batch_size)

    return cifar_10_c
