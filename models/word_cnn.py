from keras import models, layers
from models.experimental_model import ExperimentalModel


class CNNTextModel(ExperimentalModel):
    def __init__(self, input_shape=(128,), num_classes=4, vocab_size=20000, embedding_dim=128, strategy_name=''):
        """
        Initialize the CNN text model.

        Args:
            input_shape (tuple): Shape of the input sequences (sequence length,).
            num_classes (int): Number of output classes.
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding layer.
            strategy_name (str): Name of the strategy used (if any).
        """
        super().__init__(input_shape, num_classes, strategy_name=strategy_name,
                         vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.name = 'CNNText'

    def _build_model(self):
        """
        Build the CNN model for text classification.

        Returns:
            keras.Model: Compiled CNN text classification model.
        """
        input_length = self.input_shape[0]
        inputs = layers.Input(shape=self.input_shape, dtype="int32")

        x = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=input_length
        )(inputs)

        x = layers.Reshape((input_length, self.embedding_dim, 1))(x)

        convs = []
        for filter_size in [3, 4, 5]:
            conv = layers.Conv2D(
                filters=100,
                kernel_size=(filter_size, self.embedding_dim),
                activation='relu'
            )(x)
            pool = layers.MaxPool2D(pool_size=(input_length - filter_size + 1, 1))(conv)
            convs.append(pool)

        x = layers.Concatenate(axis=-1)(convs)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        return models.Model(inputs=inputs, outputs=outputs)
