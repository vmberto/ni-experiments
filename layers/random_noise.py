import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import random_uniform


class RandomNoise(Layer):
    """A preprocessing layer which adds random noise to the input during training.

    This layer will add random noise to the input during training. At inference time,
    the output will be identical to the input. Call the layer with `training=True`
    to add noise to the input.

    Args:
        stddev: Float. The standard deviation of the Gaussian noise to be added.
        seed: Optional integer. Used to seed the random generator for reproducibility.

    Inputs: A tensor of any shape.

    Output: A tensor with random noise added to the input tensor during training.

    Example:

    ```python
    random_noise = keras.layers.RandomNoise(stddev=0.1)

    # A tensor input with shape [2, 2]
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])

    # Adding random noise during training
    output = random_noise(x, training=True)

    # output will be a tensor of the same shape as input with random noise added
    ```
    """

    def __init__(self, stddev, seed=None, **kwargs):
        super(RandomNoise, self).__init__(**kwargs)
        self.stddev = stddev
        self.seed = seed

    def call(self, inputs, training=None):
        if training:
            noise = random_uniform(shape=tf.shape(inputs), minval=-self.stddev, maxval=self.stddev, dtype=inputs.dtype,
                                   seed=self.seed)
            return inputs + noise
        return inputs

    def get_config(self):
        config = {'stddev': self.stddev, 'seed': self.seed}
        base_config = super(RandomNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
