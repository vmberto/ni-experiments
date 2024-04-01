import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import random_uniform


class SaltAndPepperNoise(Layer):
    """A preprocessing layer which adds Salt and Pepper noise to the input during training.

    This layer will add Salt and Pepper noise to the input during training. At inference time,
    the output will be identical to the input. Call the layer with `training=True`
    to add noise to the input.

    Args:
        noise_level: Float between 0 and 1. The ratio of input pixels affected by salt and pepper noise.
        salt_ratio: Float between 0 and 1. The ratio of salt noise relative to the total noise.
        seed: Optional integer. Used to seed the random generator for reproducibility.

    Inputs: A tensor of any shape.

    Output: A tensor with Salt and Pepper noise added to the input tensor during training.

    Example:

    ```python
    salt_and_pepper_noise = SaltAndPepperNoise(noise_level=0.1, salt_ratio=0.5)

    # A tensor input with shape [2, 2]
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])

    # Adding salt and pepper noise during training
    output = salt_and_pepper_noise(x, training=True)

    # output will be a tensor of the same shape as input with salt and pepper noise added
    ```
    """

    def __init__(self, noise_level, salt_ratio=0.5, seed=None, **kwargs):
        super(SaltAndPepperNoise, self).__init__(**kwargs)
        self.noise_level = noise_level
        self.salt_ratio = salt_ratio
        self.seed = seed

    def call(self, inputs, training=None):
        if training:
            # Generate random values for salt and pepper noise
            salt_pepper = random_uniform(shape=tf.shape(inputs), dtype=inputs.dtype, seed=self.seed)
            salt = tf.cast(salt_pepper < self.noise_level * self.salt_ratio, inputs.dtype)
            pepper = tf.cast(salt_pepper > (1 - self.noise_level * (1 - self.salt_ratio)), inputs.dtype)

            # Apply salt and pepper noise to the input
            noisy_inputs = inputs * (1 - salt) * (1 - pepper) + salt + pepper
            return noisy_inputs
        return inputs

    def get_config(self):
        config = {'noise_level': self.noise_level, 'salt_ratio': self.salt_ratio, 'seed': self.seed}
        base_config = super(SaltAndPepperNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
