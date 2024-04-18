import tensorflow as tf
import random
from keras_cv.layers import BaseImageAugmentationLayer
import keras_cv as keras_cv
import numpy as np

def parse_factor(param, min_value=0.0, max_value=1.0, seed=None):
    if isinstance(param, keras_cv.core.FactorSampler):
        return param
    if isinstance(param, float) or isinstance(param, int):
        param = (min_value, param)
    if param[0] == param[1]:
        return keras_cv.core.ConstantFactorSampler(param[0])
    return keras_cv.core.UniformFactorSampler(param[0], param[1], seed=seed)


class RandomSaltAndPepper(BaseImageAugmentationLayer):

    def __init__(self, factor, seed=None, **kwargs):
        super().__init__(**kwargs, seed=seed)
        self.seed = seed
        self.factor = parse_factor(factor)

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_image(self, image, transformation=None, **kwargs):
        mask = tf.random.uniform(shape=tf.shape(image), minval=0, maxval=1)
        noisy_outputs = tf.where(mask < random.random() * np.random.beta(1, 10) / 2, 0.0, image)
        noisy_outputs = tf.where(mask > 1 - random.random() * np.random.beta(1, 10) / 2, 1.0, noisy_outputs)
        return noisy_outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'noise_level': self.noise_level,
            'seed': self.seed,
        }
        base_config = super(RandomSaltAndPepper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
