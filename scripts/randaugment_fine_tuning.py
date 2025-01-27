import tensorflow as tf
import keras_cv
from models.resnet50 import ResNet50Model
from keras import layers, datasets, applications, callbacks
import numpy as np
from layers.random_salt_and_pepper import RandomSaltAndPepper
import keras

##############################################################################
# 1. Load CIFAR-10 at native size (32x32)
##############################################################################
def load_cifar10():
    """
    Loads CIFAR-10, resizes images to 224x224 for ResNet50.
    Returns: (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()  # (50000,)
    y_test = y_test.squeeze()    # (10000,)

    # ResNet50 expects ~224x224 input, so let's resize:
    x_train = tf.image.resize(x_train, (72, 72))
    x_test = tf.image.resize(x_test, (72, 72))

    # Convert to float & scale to [0,1]
    x_train = tf.cast(x_train, tf.float32) / 255.0
    x_test = tf.cast(x_test, tf.float32) / 255.0

    return (x_train, y_train), (x_test, y_test)

##############################################################################
# 2. Build a ResNet50 model from scratch for 32×32 input
##############################################################################
def build_resnet50_cifar(input_shape=(72, 72, 3), num_classes=10):
    """
    Creates a ResNet50 with random initialization (weights=None),
    for a 32×32 input and 10 classes (CIFAR-10).
    """
    # 1) Base ResNet50, no pretrained weights, no top
    model = ResNet50Model()
    return model

##############################################################################
# 3. Prepare a tf.data.Dataset pipeline with optional RandAugment
##############################################################################
def prepare_dataset(x, y, randaugment_layer=None, batch_size=32, shuffle=False, salt_pepper_layer=None):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x))

    if randaugment_layer:
        # Apply RandAugment in the map step
        def augment_map(image, label):
            # KerasCV RandAugment expects images in [0,1] if value_range=(0,1)
            image = randaugment_layer(image)
            return image, label
        ds = ds.map(augment_map, num_parallel_calls=tf.data.AUTOTUNE)

    if salt_pepper_layer:
        def augment_map(image, label):
            image = salt_pepper_layer(image)
            return image, label
        ds = ds.map(augment_map, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

##############################################################################
# 4. Train & Evaluate a single set of RandAugment params
##############################################################################
def train_and_evaluate(
    randaug_params,
    x_train, y_train,
    x_test, y_test,
    epochs=5,
    batch_size=128
):
    """
    Builds a RandAugment layer with the given params,
    trains a ResNet50-from-scratch on CIFAR-10,
    returns the final validation accuracy.
    """
    # 1) RandAugment layer with the given hyperparams
    randaugment_augmenter = keras_cv.layers.RandAugment(
        value_range=(0, 1),
        augmentations_per_image=randaug_params['augmentations_per_image'],
        magnitude=randaug_params['magnitude'],
        rate=randaug_params['rate'],
    )

    random_salt_pepper = RandomSaltAndPepper(max_factor=randaug_params['s&p_factor'])

    # 2) Build the ResNet50 model (no pretrained weights)
    model = build_resnet50_cifar()

    # 3) Prepare dataset: apply augmentation to the training set only
    train_ds = prepare_dataset(
        x_train, y_train,
        randaugment_layer=randaugment_augmenter,
        salt_pepper_layer=random_salt_pepper,
        batch_size=batch_size,
        shuffle=True
    )
    test_ds = prepare_dataset(
        x_test, y_test,
        randaugment_layer=None,  # No augmentation at test time
        batch_size=batch_size,
        shuffle=False
    )

    # 4) Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics='accuracy'
    )

    # 5) EarlyStopping callback
    early_stopper = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # 6) Train
    model.fit(
        train_ds,
        val_dataset=test_ds,
        epochs=epochs,
        callbacks=[early_stopper]
    )

    # Evaluate final test accuracy
    _, test_acc = model.evaluate(test_ds)
    return test_acc

##############################################################################
# 5. Main: a hyperparam search loop for RandAugment
##############################################################################
def main():
    # A) Load CIFAR-10 at 32x32
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # B) Define some RandAugment parameter combos
    param_candidates = [
        # {'augmentations_per_image': 2, 'magnitude': 0.2},
        # {'augmentations_per_image': 3, 'magnitude': 0.3}, # .88 BEST
        # {'augmentations_per_image': 4, 'magnitude': 0.4},

        # {'augmentations_per_image': 3, 'magnitude': 0.2, 'rate': .5}, # .8493
        # {'augmentations_per_image': 3, 'magnitude': 0.2, 'rate': .75}, # .8684
        # {'augmentations_per_image': 3, 'magnitude': 0.2, 'rate': 1}, # .8717
        #
        # {'augmentations_per_image': 3, 'magnitude': 0.3, 'rate': .5}, # .8438
        # {'augmentations_per_image': 3, 'magnitude': 0.3, 'rate': .75}, # .8512
        # {'augmentations_per_image': 3, 'magnitude': 0.3, 'rate': 1}, # .8714

        # {'augmentations_per_image': 3, 'magnitude': 0.4, 'rate': .5}, # .8668
        # {'augmentations_per_image': 3, 'magnitude': 0.4, 'rate': .75}, # .8406
        # {'augmentations_per_image': 3, 'magnitude': 0.4, 'rate': 1},  # .8703

        # {'augmentations_per_image': 3, 'magnitude': 0.2, 'rate': 1}, # .8864
        # {'augmentations_per_image': 3, 'magnitude': 0.3, 'rate': 1}, # .8786
        # {'augmentations_per_image': 3, 'magnitude': 0.4, 'rate': 1}, # .8881

        # AS A FIXED FACTOR
        # {'augmentations_per_image': 3, 'magnitude': 0.3, 'rate': 1, 's&p_factor': .1}, # .8546
        # {'augmentations_per_image': 3, 'magnitude': 0.3, 'rate': 1, 's&p_factor': .2}, # .8910
        # {'augmentations_per_image': 3, 'magnitude': 0.3, 'rate': 1, 's&p_factor': .3}, # .8667

        # AS A UNIFORM DISTRIBUTION
        {'augmentations_per_image': 3, 'magnitude': 0.3, 'rate': 1, 's&p_factor': .1}, # .8635
        {'augmentations_per_image': 3, 'magnitude': 0.3, 'rate': 1, 's&p_factor': .2}, # .8631
        {'augmentations_per_image': 3, 'magnitude': 0.3, 'rate': 1, 's&p_factor': .3}, # .8835
        # {'augmentations_per_image': 3, 'magnitude': 0.3, 'rate': 1, 's&p_factor': .4}, # .8859
        # {'augmentations_per_image': 3, 'magnitude': 0.3, 'rate': 1, 's&p_factor': .5},
    ]

    best_acc = 0.0
    best_params = None

    for params in param_candidates:
        print(f"\n=== Testing RandAugment params: {params} ===")
        acc = train_and_evaluate(
            randaug_params=params,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            epochs=100,
            batch_size=128
        )
        print(f"--> Test Accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_params = params

    print("\n========== SEARCH FINISHED ==========")
    print(f"Best Accuracy: {best_acc:.4f} with params: {best_params}")

if __name__ == "__main__":
    main()