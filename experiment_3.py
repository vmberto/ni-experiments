from keras.callbacks import EarlyStopping
import multiprocessing
from models.resnet import ResNet50Model
from dataset.cifar import get_cifar10, get_cifar10_corrupted
from layers.salt_and_pepper import RandomSaltAndPepper
import keras_cv.layers as layers
import keras_cv as keras_cv
from utils.images import save_img_examples
import tensorflow as tf
from utils.metrics import write_acc_avg, write_acc_each_dataset

BATCH_SIZE = 128
IMG_SIZE = 72
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


def run():
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)
    #
    # factor = keras_cv.core.NormalFactorSampler(
    #     mean=0.3, stddev=0.1, min_value=.0, max_value=.9
    # )
    #
    # data_augmentation_layers = [
    #     RandomSaltAndPepper(factor),
    #     # layers.RandomFlip("horizontal"),
    #     # layers.RandomRotation(factor=0.02),
    #     # layers.RandomZoom(
    #     #     height_factor=0.2, width_factor=0.2
    #     # ),
    # ]
    # data_augmentation_params = [
    #     # 'horizontal',
    #     # 'factor0.02',
    #     # 'heightwidthfactor0.2',
    #     ''
    # ]
    #
    # train_ds, val_ds = get_cifar10(BATCH_SIZE, data_augmentation_layers)
    #
    # resnet = ResNet50Model(input_shape=INPUT_SHAPE)
    #
    # resnet.compile()
    #
    # resnet.fit(
    #     train_ds,
    #     val_dataset=val_ds,
    #     epochs=100,
    #     callbacks=[EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)]
    # )
    #
    # for corruption in [
    #     'brightness_3',
    #     'contrast_3',
    #     'defocus_blur_3',
    #     'elastic_3',
    #     'fog_3',
    #     'frost_3',
    #     'frosted_glass_blur_3',
    #     'gaussian_blur_3',
    #     'gaussian_noise_3',
    #     'impulse_noise_3',
    #     'jpeg_compression_3',
    #     'motion_blur_3',
    #     'pixelate_3',
    #     'saturate_3',
    #     'shot_noise_3',
    #     'snow_3',
    #     'spatter_3',
    #     'speckle_noise_3',
    #     'zoom_blur_3',
    # ]:
    #     resnet.evaluate(
    #         get_cifar10_corrupted(BATCH_SIZE, corruption),
    #         f'cifar10/{corruption}',
    #         data_augmentation_layers,
    #         data_augmentation_params,
    #     )

    write_acc_avg()
    write_acc_each_dataset()


if __name__ == "__main__":
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()
    print("finished")
