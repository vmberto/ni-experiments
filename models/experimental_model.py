from sklearn.metrics import classification_report
import numpy as np
import time


class ExperimentalModel:
    def __init__(self, input_shape=(32, 32, 3), num_classes=10, vocab_size=20000, embedding_dim=128, strategy_name=''):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.strategy_name = strategy_name
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
        self.compile()

    def _build_model(self):
        raise NotImplementedError()

    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy'):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    def fit(self, train_dataset, val_dataset=None, callbacks=None, epochs=10):
        start_time = time.time()
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
        )
        end_time = time.time()
        training_time = end_time - start_time

        return history, training_time

    def predict(self, dataset):
        y_true = np.concatenate([y for x, y in dataset], axis=0)

        y_pred = self.model.predict(dataset, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)
        report = classification_report(y_true, y_pred, output_dict=True)
        return report

    def evaluate(self, dataset):
        loss, acc = self.model.evaluate(dataset, verbose=0)
        return loss, acc
