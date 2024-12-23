from keras import models, layers
from models.experimental_model import ExperimentalModel


class VanillaRNNModel(ExperimentalModel):
    def __init__(
        self,
        input_shape=(128,),
        num_classes=4,
        vocab_size=20000,
        embedding_dim=128,
        strategy_name=""
    ):
        """
        Initialize the vanilla RNN model.

        Args:
            input_shape (tuple): Shape of the input sequences (sequence length,).
            num_classes (int): Number of output classes.
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding layer.
            strategy_name (str): Name of the strategy used (if any).
        """
        super().__init__(
            input_shape,
            num_classes,
            strategy_name=strategy_name,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )
        self.name = "VanillaRNN"

    def _build_model(self):
        """
        Build a Vanilla RNN (SimpleRNN) model.

        Returns:
            keras.Model: Compiled Vanilla RNN model.
        """
        model = models.Sequential(
            [
                layers.Input(self.input_shape),
                layers.Embedding(
                    input_dim=self.vocab_size,
                    output_dim=self.embedding_dim,
                ),
                # Replaced LSTM with SimpleRNN
                layers.SimpleRNN(128, return_sequences=False),
                layers.Dense(64, activation="relu"),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        return model
