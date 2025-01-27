import tensorflow as tf
from keras import layers, datasets, callbacks, models
from layers.custom_gaussian_noise import CustomGaussianNoise
import numpy as np


##############################################################################
# 1. Load CIFAR-10 at native size (32x32)
##############################################################################
def load_cifar10():
    """
    Loads CIFAR-10, resizes images to 72x72 for ResNet50.
    Returns: (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    y_train = y_train.squeeze()  # Shape: (50000,)
    y_test = y_test.squeeze()    # Shape: (10000,)

    # Resize for compatibility with ResNet50 input
    x_train = tf.image.resize(x_train, (72, 72))
    x_test = tf.image.resize(x_test, (72, 72))

    # Normalize to range [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return (x_train, y_train), (x_test, y_test)


##############################################################################
# 2. Build a ResNet50 model from scratch for 72×72 input
##############################################################################
def build_resnet50_cifar(input_shape=(72, 72, 3), num_classes=10):
    """
    Creates a ResNet50 with random initialization, for CIFAR-10.
    """
    base_model = tf.keras.applications.ResNet50(
        weights=None,  # No pretrained weights
        include_top=False,
        input_shape=input_shape
    )
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=base_model.input, outputs=x)


##############################################################################
# 3. Prepare a tf.data.Dataset pipeline with Gaussian Noise
##############################################################################
def prepare_dataset(x, y, gaussian_noise_layer=None, batch_size=32, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x))

    if gaussian_noise_layer:
        def augment_map(image, label):
            image = gaussian_noise_layer(image)
            return image, label

        ds = ds.map(augment_map, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


##############################################################################
# 4. Train & Evaluate with Gaussian Noise Layer
##############################################################################
def train_and_evaluate(
    noise_stddev,
    x_train, y_train,
    x_test, y_test,
    epochs=10,
    batch_size=128
):
    """
    Builds a Gaussian noise layer, trains a ResNet50-from-scratch on CIFAR-10,
    and returns the final validation accuracy.
    """
    # 1) Create the Gaussian noise layer
    gaussian_noise_layer = layers.GaussianNoise(noise_stddev)
    # gaussian_noise_layer = CustomGaussianNoise(max_stddev=noise_stddev)

    # 2) Build the ResNet50 model
    model = build_resnet50_cifar()

    # 3) Prepare the datasets
    train_ds = prepare_dataset(
        x_train, y_train,
        gaussian_noise_layer=gaussian_noise_layer,
        batch_size=batch_size,
        shuffle=True
    )
    test_ds = prepare_dataset(
        x_test, y_test,
        gaussian_noise_layer=None,  # No noise at test time
        batch_size=batch_size,
        shuffle=False
    )

    # 4) Compile the model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 5) EarlyStopping callback
    early_stopper = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # 6) Train the model
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=[early_stopper]
    )

    # Evaluate final test accuracy
    _, test_acc = model.evaluate(test_ds)
    return test_acc


##############################################################################
# 5. Main: Perform hyperparameter search for Gaussian Noise
##############################################################################
def main():
    # A) Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # B) FIXED STDDEV
    noise_stddev_candidates = [
        0.1, # .7758 / .7457
        0.2, # .7254 / .7305
        0.3, # .7383 / .7357
        0.4, # .7107 / .7621
        0.5  # .7535 / .7382
    ]

    # B) UNIFORM STDDEV
    # noise_stddev_candidates = [
    #     0.1, # .7330 / .8079
    #     0.2, # .7658 / .7341
    #     0.3, # .7381 / .7225
    #     0.4, # .7221 / .7257
    #     0.5  # .7585 / .7404
    # ]

    best_acc = 0.0
    best_stddev = None

    for stddev in noise_stddev_candidates:
        print(f"\n=== Testing Gaussian Noise stddev: {stddev} ===")
        acc = train_and_evaluate(
            noise_stddev=stddev,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            epochs=50,
            batch_size=128
        )
        print(f"--> Test Accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_stddev = stddev

    print("\n========== SEARCH FINISHED ==========")
    print(f"Best Accuracy: {best_acc:.4f} with Gaussian Noise stddev: {best_stddev}")


if __name__ == "__main__":
    main()