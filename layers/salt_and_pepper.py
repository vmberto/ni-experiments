import tensorflow as tf
from tensorflow.keras.layers import Layer


class SaltAndPepperNoise(Layer):
    def __init__(self, noise_level=0.05):
        super(SaltAndPepperNoise, self).__init__()
        self.noise_level = noise_level

    def call(self, inputs, training=None):
        salt_pepper = tf.random.uniform(tf.shape(inputs), minval=0, maxval=1)
        salt = salt_pepper < self.noise_level / 2
        pepper = salt_pepper > 1 - self.noise_level / 2
        noisy_inputs = tf.where(salt, 255.0, inputs)
        noisy_inputs = tf.where(pepper, 0.0, noisy_inputs)
        return noisy_inputs
