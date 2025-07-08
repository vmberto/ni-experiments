from keras import layers, models

class TextAutoencoder(models.Model):
    def __init__(self, vocab_size, embedding_dim, max_len):
        super(TextAutoencoder, self).__init__()
        self.encoder = self.build_encoder(vocab_size, embedding_dim, max_len)
        self.decoder = self.build_decoder(self.encoder.output_shape[1], max_len, vocab_size)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def build_encoder(self, vocab_size, embedding_dim, max_len):
        encoder_input = layers.Input(shape=(max_len,))
        x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(encoder_input)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
        x = layers.Dense(128, activation='relu')(x)
        return models.Model(encoder_input, x, name='encoder')

    def build_decoder(self, latent_dim, max_len, vocab_size):
        decoder_input = layers.Input(shape=(latent_dim,))
        x = layers.RepeatVector(max_len)(decoder_input)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(x)
        return models.Model(decoder_input, x, name='decoder')
