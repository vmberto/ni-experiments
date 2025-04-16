from keras import applications, models, layers
from models.experimental_model import ExperimentalModel


class EfficientNetB0Model(ExperimentalModel):

    def __init__(self, input_shape=(72, 72, 3), num_classes=10, strategy_name=''):
        super().__init__(input_shape, num_classes, strategy_name)
        self.name = 'EfficientNetB0'

    def _build_model(self):
        efficientnetb0 = applications.EfficientNetB0(
            weights=None,
            include_top=True,
            input_shape=self.input_shape,
            classes=10,
        )
        model = models.Sequential(
            [
                layers.Input(self.input_shape),
                efficientnetb0,
            ]
        )
        return model
