import tensorflow as tf
from keras import layers


class RandomSaltAndPepper(layers.Layer):
    """
    A custom Keras layer that adds salt-and-pepper noise to inputs.

    Args:
        factor (float): The probability of noise corruption. Should be in the range [0.0, 1.0].
        seed (int, optional): Random seed for reproducibility.
    """

    def __init__(self, factor=0.1, seed=None, **kwargs):
        super(RandomSaltAndPepper, self).__init__(**kwargs)
        self.factor = factor
        self.seed = seed
        self.random_generator = tf.random.Generator.from_seed(seed) if seed else tf.random.Generator.from_non_deterministic_state()

    def call(self, inputs, training=None):
        if not training:
            return inputs

        shape = tf.shape(inputs)
        random_values = self.random_generator.uniform(shape, minval=0.0, maxval=1.0)

        salt_mask = random_values > (1 - self.factor / 2)
        pepper_mask = random_values < (self.factor / 2)

        outputs = tf.where(salt_mask, 1.0, inputs)
        outputs = tf.where(pepper_mask, 0.0, outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(RandomSaltAndPepper, self).get_config()
        config.update({"factor": self.factor, "seed": self.seed})
        return config
