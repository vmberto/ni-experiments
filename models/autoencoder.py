from tensorflow.keras import models, layers


class Autoencoder(models.Model):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        self.encoder = self.build_encoder(input_shape)
        self.decoder = self.build_decoder(self.encoder.output_shape[1:])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def residual_block(self, x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)

        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.add([shortcut, x])
        x = layers.Activation('relu')(x)
        return x

    def build_encoder(self, input_shape):
        encoder_input = layers.Input(shape=input_shape)
        x = layers.Conv2D(64, (3, 3), strides=2, padding='same')(encoder_input)  # 16x16x64
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = self.residual_block(x, 64)
        x = self.residual_block(x, 64)
        x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)  # 8x8x128
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = self.residual_block(x, 128)
        x = self.residual_block(x, 128)
        x = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)  # 4x4x256
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = self.residual_block(x, 256)
        x = self.residual_block(x, 256)
        encoder_output = x
        return models.Model(encoder_input, encoder_output, name='encoder')

    def build_decoder(self, input_shape):
        decoder_input = layers.Input(shape=input_shape)
        x = self.residual_block(decoder_input, 256)
        x = self.residual_block(x, 256)
        x = layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same')(x)  # 8x8x256
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = self.residual_block(x, 128)
        x = self.residual_block(x, 128)
        x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)  # 16x16x128
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = self.residual_block(x, 64)
        x = self.residual_block(x, 64)
        x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)  # 32x32x64
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Output layer with sigmoid activation
        return models.Model(decoder_input, x, name='decoder')
