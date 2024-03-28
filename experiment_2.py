"""

RESNET50, Batch-size 128, 20 epochs
CIFAR-10: Training 50000 images (train 45000, val 5000), Testing 10000 images
Random Noise: 0, 0.2, 0.4, 0.6

"""
from keras.callbacks import EarlyStopping
import tensorflow as tf
from models.resnet import ResNet50Model
from utils.augment import apply_noise
from dataset.mnist import get_mnist, get_mnist_corrupted

BATCH_SIZE = 128

(x_train, y_train), (x_test, y_test) = get_mnist()
x_data, y_data = get_mnist_corrupted()

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(apply_noise)
train_ds = train_ds.shuffle(5000).batch(BATCH_SIZE)

val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = val_ds.shuffle(5000).batch(BATCH_SIZE)

resnet = ResNet50Model()

resnet.compile()

resnet.fit(
    train_ds,
    val_generator=val_ds,
    epochs=20,
    callbacks=[EarlyStopping(patience=4, monitor='val_loss', verbose=1)]
)


loss, acc = resnet.evaluate(x_data, y_data)

resnet.predict(x_data, y_data)


