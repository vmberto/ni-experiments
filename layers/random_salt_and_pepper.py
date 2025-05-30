import tensorflow as tf
from keras import layers


class RandomSaltAndPepper(layers.Layer):
    def __init__(self, factor=None, min_factor=None, max_factor=None, seed=None, **kwargs):
        super(RandomSaltAndPepper, self).__init__(**kwargs)
        if factor is not None:
            if min_factor is not None or max_factor is not None:
                raise ValueError("If 'factor' is defined, 'min_factor' and 'max_factor' cannot be defined.")
            self.factor = factor
            self.min_factor = None
            self.max_factor = None
        elif max_factor is not None:
            self.min_factor = 0.0 if min_factor is None else min_factor
            self.max_factor = max_factor
            if not 0.0 <= self.min_factor <= 1.0 or not 0.0 <= self.max_factor <= 1.0:
                raise ValueError("'min_factor' and 'max_factor' must be in the range [0.0, 1.0].")
            if self.min_factor > self.max_factor:
                raise ValueError("'min_factor' must be less than or equal to 'max_factor'.")
            self.factor = None
        else:
            raise ValueError("Either 'factor' must be defined, or 'max_factor' must be.")

        self.seed = seed
        self.random_generator = tf.random.Generator.from_seed(seed) if seed else tf.random.Generator.from_non_deterministic_state()

    def call(self, inputs, training=None):
        if not training:
            return inputs
        return self._add_salt_and_pepper(inputs)

    @tf.function
    def _add_salt_and_pepper(self, inputs):
        factor = (
            self.factor
            if self.factor is not None
            else self.random_generator.uniform(shape=[], minval=self.min_factor, maxval=self.max_factor)
        )

        shape = tf.shape(inputs)
        random_values = self.random_generator.uniform(shape, minval=0.0, maxval=1.0)

        salt_mask = random_values > (1 - factor / 2)
        pepper_mask = random_values < (factor / 2)

        outputs = tf.where(salt_mask, 1.0, inputs)
        outputs = tf.where(pepper_mask, 0.0, outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "factor": self.factor,
            "min_factor": self.min_factor,
            "max_factor": self.max_factor,
            "seed": self.seed,
        })
        return config