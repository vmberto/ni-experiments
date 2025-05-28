from keras.src import layers
import tensorflow as tf


class CustomGaussianNoise(layers.Layer):
    def __init__(self, stddev=None, min_stddev=None, max_stddev=None, **kwargs):
        super().__init__(**kwargs)

        if stddev is not None:
            if min_stddev is not None or max_stddev is not None:
                raise ValueError("If `stddev` is defined, `min_stddev` and `max_stddev` cannot be defined.")
            if not 0 <= stddev <= 1:
                raise ValueError(f"Invalid value for `stddev`. Expected a float between 0 and 1. Received: {stddev}")
            self.stddev = stddev
            self.min_stddev = None
            self.max_stddev = None
        elif max_stddev is not None:
            self.min_stddev = 0.0 if min_stddev is None else min_stddev
            self.max_stddev = max_stddev
            if not 0.0 <= self.min_stddev <= 1.0 or not 0.0 <= self.max_stddev <= 1.0:
                raise ValueError("`min_stddev` and `max_stddev` must be in [0.0, 1.0]")
            if self.min_stddev > self.max_stddev:
                raise ValueError("`min_stddev` must be less than or equal to `max_stddev`.")
            self.stddev = None
        else:
            raise ValueError("You must provide either `stddev` or `max_stddev` (and optionally `min_stddev`).")

        self.random_generator = tf.random.Generator.from_non_deterministic_state()
        self.supports_masking = True
        self.built = True
        self.last_stddev = None

    def call(self, inputs, training=False):
        if not training:
            return inputs
        return self._apply_gaussian_noise(inputs)

    @tf.function
    def _apply_gaussian_noise(self, inputs):
        stddev = (
            self.stddev
            if self.stddev is not None
            else self.random_generator.uniform([], self.min_stddev, self.max_stddev)
        )

        self.last_stddev = stddev

        noise = tf.random.normal(
            shape=tf.shape(inputs),
            mean=0.0,
            stddev=stddev,
            dtype=inputs.dtype,
        )
        return inputs + noise

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        config = {
            "stddev": self.stddev,
            "min_stddev": getattr(self, "min_stddev", None),
            "max_stddev": getattr(self, "max_stddev", None),
        }
        return {**base_config, **config}
