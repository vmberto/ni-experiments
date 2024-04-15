from utils.metrics import write_acc_loss_result
import time


class ExperimentalModel:
    def __init__(self, input_shape=(32, 32, 3), num_classes=10, approach_name=''):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.approach_name = approach_name
        self.model = self._build_model()
        self.compile()

    def _build_model(self):
        raise NotImplementedError()

    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy'):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    def fit(self, train_dataset, val_dataset=None, callbacks=None, batch_size=32, epochs=10):
        start_time = time.time()
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
        )
        end_time = time.time()
        training_time = end_time - start_time

        return history, training_time

    def evaluate(self, eval_ds, corruption_type, execution_name):
        loss, acc = self.model.evaluate(eval_ds)

        write_acc_loss_result(corruption_type, execution_name, loss, acc)

        return loss, acc

    def predict(self, dataset):
        return self.model.predict(dataset, verbose=0)
