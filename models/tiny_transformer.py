import tensorflow as tf
from keras import layers, models
from models.experimental_model import ExperimentalModel


class TinyTransformerModel(ExperimentalModel):
    def __init__(self, input_shape=(128,), num_classes=4, vocab_size=20000, embedding_dim=128, strategy_name=''):
        """
        Initialize the TinyTransformer model.

        Args:
            input_shape (tuple): Shape of input sequences.
            num_classes (int): Number of output classes.
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Dimension of embeddings.
            strategy_name (str): Optional label for the regularization strategy used.
        """
        super().__init__(input_shape, num_classes, strategy_name=strategy_name,
                         vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.name = 'TinyTransformer'

    def _build_model(self):
        """
        Build the Tiny Transformer model.

        Returns:
            keras.Model: A compiled Transformer-based model.
        """
        inputs = layers.Input(shape=self.input_shape, dtype=tf.int32)

        # Embedding + positional encoding
        x = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        x = self._add_positional_encoding(x)

        # Transformer block
        attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=self.embedding_dim)(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)

        # Feed-forward block
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        return models.Model(inputs=inputs, outputs=outputs)

    def _add_positional_encoding(self, x):
        """
        Add sinusoidal positional encoding to the input tensor.
        """
        seq_len = x.shape[1]
        d_model = x.shape[2]

        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]

        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = position * angle_rates

        # apply sin to even indices and cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)

        pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # shape (1, seq_len, d_model)
        return x + tf.cast(pos_encoding, dtype=tf.float32)
