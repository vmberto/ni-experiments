from keras.callbacks import EarlyStopping
import multiprocessing
from models.resnet import ResNet50Model
from dataset.cifar import get_cifar10, get_cifar10_corrupted
import tensorflow as tf
from layers.salt_and_pepper import SaltAndPepperNoise

BATCH_SIZE = 128
IMG_SIZE = 72
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


def run():
    aug_layers = [
        tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
    ]

    train_ds, val_ds = get_cifar10(BATCH_SIZE, aug_layers)

    resnet = ResNet50Model(input_shape=INPUT_SHAPE)

    resnet.compile()

    resnet.fit(
        train_ds,
        val_dataset=val_ds,
        epochs=100,
        callbacks=[EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)]
    )

    for corruption in [
        'brightness_3',
        'spatter_3',
        'contrast_3',
        'jpeg_compression_3',
        'saturate_3',
        'pixelate_3',
        'defocus_blur_3',
        'gaussian_blur_3',
        'speckle_noise_3',
        'gaussian_noise_3'
    ]:
        resnet.evaluate(
            get_cifar10_corrupted(BATCH_SIZE, corruption),
            f'cifar10/{corruption}',
            aug_layers
        )


if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()
    print("finished")
