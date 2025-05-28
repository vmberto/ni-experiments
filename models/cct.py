from keras import layers, models
from models.experimental_model import ExperimentalModel
import tensorflow as tf


class SequencePooling(layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.attention = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
            name="attn_pool_weights",
        )

    def call(self, inputs):
        # inputs: (batch_size, sequence_length, dim)
        weights = tf.matmul(inputs, self.attention)  # (batch, seq, 1)
        weights = tf.nn.softmax(weights, axis=1)
        pooled = tf.reduce_sum(inputs * weights, axis=1)
        return pooled


def mlp(x, hidden_units):
    for units in hidden_units:
        x = layers.Dense(units, activation="gelu")(x)
    return x


class CCTCIFAR10Model(ExperimentalModel):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10, strategy_name=''):
        super().__init__(input_shape, num_classes, strategy_name)
        self.name = 'CCT'

    def _build_model(self):
        conv_layers_config = [
            (64, 3, 1),
            (128, 3, 2),
            (256, 3, 2)
        ]

        projection_dim = 128
        num_heads = 4
        transformer_layers = 4
        transformer_units = [projection_dim * 2, projection_dim]

        inputs = layers.Input(shape=self.input_shape)
        x = inputs

        # Convolutional Tokenizer
        for filters, kernel_size, stride in conv_layers_config:
            x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("gelu")(x)

        # Reshape to (batch_size, sequence_length, embedding_dim)
        h, w = x.shape[1], x.shape[2]
        x = layers.Reshape((h * w, x.shape[-1]))(x)

        # Project to transformer dimension
        x = layers.Dense(projection_dim)(x)

        # Transformer Encoder
        for _ in range(transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(x)
            attn_out = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=projection_dim // num_heads
            )(x1, x1)
            x2 = layers.Add()([attn_out, x])

            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            mlp_out = mlp(x3, transformer_units)
            x = layers.Add()([mlp_out, x2])

        # Sequence pooling instead of CLS token
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = SequencePooling()(x)

        # Final classification head
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        return models.Model(inputs=inputs, outputs=outputs)
