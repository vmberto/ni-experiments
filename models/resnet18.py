import tensorflow as tf
from models.experimental_model import ExperimentalModel
from keras import applications, models, layers
from models.experimental_model import ExperimentalModel
from classification_models.tfkeras import Classifiers


class ResNet18Model(ExperimentalModel):

    def __init__(self, input_shape=(72, 72, 3), num_classes=10, strategy_name=''):
        super().__init__(input_shape, num_classes, strategy_name)
        self.name = 'ResNet18'

    def _build_model(self):
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        resnet18 = ResNet18(
            weights=None,
            include_top=True,
            input_shape=self.input_shape,
            classes=10,
        )
        model = models.Sequential(
            [
                layers.Input(self.input_shape),
                resnet18,
            ]
        )
        return model
