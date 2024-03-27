"""

RESNET50, Batch-size 128, 20 epochs
CIFAR-10: Training 50000 images (train 45000, val 5000), Testing 10000 images
Random Noise: 0, 0.2, 0.4, 0.6

"""
from keras.callbacks import EarlyStopping
import numpy as np
import keras as k
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from models.resnet import ResNet50Model


def inject_noise(image, noise_factor=0.1):
    noise = np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    noise *= noise_factor
    noisy_image = np.clip(image + noise, 0.0, 1.0)
    return noisy_image


# DATASET
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_rows, img_cols, channels = 32, 32, 3

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

num_classes = 10

y_train = k.utils.to_categorical(y_train, num_classes)
y_test = k.utils.to_categorical(y_test, num_classes)

for noise_level in [0, .2, .4, .6]:
    datagen = ImageDataGenerator(
        preprocessing_function=lambda x: inject_noise(x, noise_factor=noise_level),
        validation_split=.1
    )
    datagen.fit(x_train)

    train_gen = datagen.flow(
        x_train,
        y_train,
        batch_size=128,
        subset='training'
    )
    val_gen = datagen.flow(
        x_train,
        y_train,
        batch_size=128,
        subset='validation'
    )

    resnet = ResNet50Model()

    resnet.compile()

    resnet.fit(
        train_gen,
        val_generator=val_gen,
        epochs=20,
        callbacks=[EarlyStopping(patience=4, monitor='val_loss', verbose=1)]
    )

    resnet.evaluate(x_test, y_test)

