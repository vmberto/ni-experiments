from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src.api_export import keras_export
import tensorflow as tf


@keras_export("keras.layers.CustomGaussianNoise")
class CustomGaussianNoise(layers.Layer):
    """
    Apply additive zero-centered Gaussian noise with a fixed or variable standard deviation.

    This is useful for regularization and can help mitigate overfitting.

    Args:
        stddev (float, optional): Fixed standard deviation of the noise. Must be in the range [0, 1].
        min_stddev (float, optional): Minimum standard deviation for a range of noise. Must be in [0, 1].
        max_stddev (float, optional): Maximum standard deviation for a range of noise. Must be in [0, 1].
        seed (int, optional): Optional random seed for reproducibility.

    Notes:
        - If `stddev` is defined, `min_stddev` and `max_stddev` cannot be defined.
        - If `min_stddev` is defined, `max_stddev` must also be defined.
        - If only `max_stddev` is defined, `min_stddev` defaults to 0.0.
    """

    def __init__(self, stddev=None, min_stddev=None, max_stddev=None, **kwargs):
        super().__init__(**kwargs)

        if stddev is not None:
            if min_stddev is not None or max_stddev is not None:
                raise ValueError(
                    "If `stddev` is defined, `min_stddev` and `max_stddev` cannot be defined."
                )
            if not 0 <= stddev <= 1:
                raise ValueError(
                    f"Invalid value for `stddev`. Expected a float between 0 and 1. Received: {stddev}"
                )
            self.stddev = stddev
            self.min_stddev = None
            self.max_stddev = None
        elif max_stddev is not None:
            self.min_stddev = 0.0 if min_stddev is None else min_stddev
            self.max_stddev = max_stddev
            if not 0.0 <= self.min_stddev <= 1.0 or not 0.0 <= self.max_stddev <= 1.0:
                raise ValueError(
                    "`min_stddev` and `max_stddev` must be in the range [0.0, 1.0]."
                )
            if self.min_stddev > self.max_stddev:
                raise ValueError("`min_stddev` must be less than or equal to `max_stddev`.")
            self.stddev = None
        else:
            raise ValueError(
                "Either `stddev` must be defined, or `max_stddev` (with optional `min_stddev`) must be defined."
            )

        self.random_generator = tf.random.Generator.from_non_deterministic_state()
        self.supports_masking = True
        self.built = True

    def call(self, inputs, training=False):
        if not training:
            return inputs

        stddev = (
            self.stddev
            if self.stddev is not None
            else self.random_generator.uniform(
                shape=[],
                minval=self.min_stddev,
                maxval=self.max_stddev,
            )
        )

        noise = backend.random.normal(
            shape=ops.shape(inputs),
            mean=0.0,
            stddev=stddev,
            dtype=self.compute_dtype,
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
