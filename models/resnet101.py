import tensorflow as tf
from models.experimental_model import ExperimentalModel
from keras import applications, models, layers
from models.experimental_model import ExperimentalModel


class ResNet101Model(ExperimentalModel):

    def __init__(self, input_shape=(32, 32, 3), num_classes=10, strategy_name=''):
        super().__init__(input_shape, num_classes, strategy_name)
        self.name = 'ResNet101'

    def _build_model(self):
        resnet101 = applications.ResNet101(
            weights=None,
            include_top=True,
            input_shape=self.input_shape,
            classes=10,
        )
        model = models.Sequential(
            [
                layers.Input(self.input_shape),
                resnet101,
            ]
        )
        return model
