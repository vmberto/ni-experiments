import numpy as np
from utils.metrics import save_accuracy_evolution, write_evaluation_result, save_confusion_matrix


class ExperimentalModel:
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        return

    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy'):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
        self.model.summary()

    def fit(self, train_dataset, val_dataset=None, callbacks=None, batch_size=32, epochs=10):
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )

        save_accuracy_evolution(history.history)

        return history

    def evaluate(self, eval_ds, evaluation_name, aug_layers):
        loss, acc = self.model.evaluate(eval_ds)

        write_evaluation_result(evaluation_name, aug_layers, loss, acc)

        return loss, acc

    def predict(self, dataset, y_true):
        prediction = self.model.predict(dataset)
        y_pred = np.argmax(prediction, axis=1)

        save_confusion_matrix(y_pred, y_true)

        return prediction
