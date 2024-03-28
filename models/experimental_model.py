import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


class ExperimentalModel:
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        return

    def compile(self, optimizer='sgd', loss='categorical_crossentropy', metrics='accuracy'):
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
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.ylim(0, 1)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'./output/history.png')
        plt.clf()
        return history

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def predict(self, dataset, y_true):
        prediction = self.model.predict(dataset)
        y_prediction = np.argmax(prediction, axis=1)
        cm = confusion_matrix(y_true, y_prediction)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f'./output/confusion_matrix.png')
        plt.clf()

        return prediction
