import tensorflow as tf
import random


class RandomPreprocessApplier(tf.keras.layers.Layer):

    def __init__(self, approaches):
        super(RandomPreprocessApplier, self).__init__()
        self.approaches = approaches

    def call(self, inputs):
        print(inputs)
        random_index = random.randint(0, len(self.approaches) - 1)
        aug_approaches = self.approaches[random_index]
        if aug_approaches is not None:
            return aug_approaches(inputs, training=True)

        return inputs
