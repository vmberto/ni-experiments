from keras import models, layers
from models.experimental_model import ExperimentalModel


class BiLSTMModel(ExperimentalModel):
    def __init__(self, input_shape=(128,), num_classes=4, vocab_size=20000, embedding_dim=128, strategy_name=''):
        """
        Initialize the BiLSTM model.

        Args:
            input_shape (tuple): Shape of the input sequences.
            num_classes (int): Number of output classes.
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding layer.
            strategy_name (str): Name of the strategy used (if any).
        """
        super().__init__(input_shape, num_classes, strategy_name=strategy_name,
                         vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.name = 'BiLSTM'

    def _build_model(self):
        """
        Build the BiLSTM model.

        Returns:
            keras.Model: Compiled BiLSTM model.
        """
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim),
            layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax'),
        ])
        return model
