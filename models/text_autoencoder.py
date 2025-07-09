from keras import layers, models

class TextAutoencoder(models.Model):
    def __init__(self, vocab_size, embedding_dim, max_len, latent_dim):
        super(TextAutoencoder, self).__init__()
        self.encoder = models.Sequential([
            layers.Input(shape=(max_len,), dtype='int32'),
            layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
            layers.Bidirectional(layers.LSTM(64, return_sequences=False)),  # Optional: LSTM for more expressiveness
            layers.Dense(latent_dim, activation='relu', name='latent_space')
        ], name='encoder')

    def call(self, inputs):
        return self.encoder(inputs)