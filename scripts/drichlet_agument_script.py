import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # for Dirichlet, Categorical

tfd = tfp.distributions
from keras import layers, models, optimizers, losses


###############################################################################
# 1. A custom DirichletAugment Keras Layer (same as before)
###############################################################################
class DirichletAugment(layers.Layer):
    """
    Holds a list of k augmentation callables (e.g. small Keras layers)
    and a trainable Dirichlet alpha. On forward pass (call), it:
      1. Samples p ~ Dirichlet(alpha).
      2. For each image in the batch, samples an augmentation index ~ Categorical(p).
      3. Applies that augmentation transform.
    """

    def __init__(self, transform_list, alpha_init=1.0, **kwargs):
        super().__init__(**kwargs)
        self.transform_list = transform_list
        self.k = len(transform_list)

        # We'll store log(alpha) as a trainable Variable of shape (k,).
        # Then map it to (0, ∞) via tf.nn.softplus.
        log_alpha_init = tf.math.log(tf.ones((self.k,)) * alpha_init)
        self.log_alpha = tf.Variable(
            initial_value=log_alpha_init,
            trainable=True,
            name="log_alpha"
        )

    def call(self, x, training=None):
        """
        x: a batch of images, shape (B, H, W, C).
        Returns an augmented batch of the same shape.
        """
        # If not training, skip augmentation (pass-through).
        if not training:
            return x

        # Convert log_alpha -> alpha > 0
        alpha_pos = tf.nn.softplus(self.log_alpha)  # shape (k,)

        # Sample p ~ Dirichlet(alpha_pos)
        dirichlet_dist = tfd.Dirichlet(alpha_pos)
        p = dirichlet_dist.sample()  # shape (k,)

        # For each image, sample an augmentation index from Categorical(p).
        # Then apply that augmentation.
        batch_size = tf.shape(x)[0]

        def augment_one_image(image):
            # Sample an index idx in [0, k-1]
            cat_dist = tfd.Categorical(probs=p)
            idx = cat_dist.sample()
            idx_int = tf.cast(idx, tf.int32)

            # We'll do a 'switch_case' to pick the correct transform.
            out = tf.switch_case(
                branch_index=idx_int,
                branch_fns=[
                    lambda i=i: self.transform_list[i](
                        tf.expand_dims(image, axis=0), training=True
                    )[0]
                    for i in range(self.k)
                ]
            )
            return out

        # Map over the batch dimension
        augmented_x = tf.map_fn(augment_one_image, x, fn_output_signature=tf.float32)
        return augmented_x


###############################################################################
# 2. Define transformations (akin to your PyTorch transform_list)
###############################################################################
# We'll define 3 "always apply" transformations:
#  1) Random horizontal flip
#  2) "Convert to grayscale" (approx)
#  3) Random vertical flip

transform_1 = models.Sequential([
    layers.RandomFlip(mode="horizontal")
])


class ForceGrayscale(layers.Layer):
    def call(self, x, training=None):
        # x shape: (batch, H, W, 3)
        gray = tf.image.rgb_to_grayscale(x)  # -> (batch, H, W, 1)
        gray_3c = tf.tile(gray, [1, 1, 1, 3])  # replicate to shape (batch, H, W, 3)
        return gray_3c


transform_2 = models.Sequential([
    ForceGrayscale()
])

transform_3 = models.Sequential([
    layers.RandomFlip(mode="vertical")
])

transform_list = [transform_1, transform_2, transform_3]


###############################################################################
# 3. A small CNN model (you could replace with ResNet-like if desired)
###############################################################################
def build_cnn(num_classes=10, input_shape=(32, 32, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


###############################################################################
# 4. Wrap DirichletAugment + CNN in a single Keras Model
###############################################################################
# This is crucial so that 'model.fit' can see the entire forward pass,
# and gradients can flow to BOTH the CNN and the DirichletAugment's log_alpha.

class AugmentAndModel(tf.keras.Model):
    def __init__(self, augment_layer, base_model):
        super().__init__()
        self.augment_layer = augment_layer
        self.base_model = base_model

    def call(self, inputs, training=None):
        # 1) Augment
        x = self.augment_layer(inputs, training=training)
        # 2) Forward pass of the main model
        return self.base_model(x, training=training)


###############################################################################
# 5. Fake dataset (similar to Torch FakeData)
###############################################################################
def create_fake_dataset(num_samples=256, num_classes=10, image_size=(32, 32, 3), batch_size=32):
    H, W, C = image_size
    x = np.random.rand(num_samples, H, W, C).astype('float32')
    y = np.random.randint(0, num_classes, size=(num_samples,))

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(1000).batch(batch_size)
    return ds


###############################################################################
# 6. main() - Put it all together and use model.fit() with tf.data.Dataset
###############################################################################
def main():
    # A) Create the Dirichlet augmentation module
    augment_module = DirichletAugment(transform_list, alpha_init=1.0)

    # B) Build a small CNN
    base_model = build_cnn(num_classes=10, input_shape=(32, 32, 3))

    # C) Combine them in a single Keras Model
    full_model = AugmentAndModel(augment_module, base_model)

    # D) Create a synthetic dataset
    train_ds = create_fake_dataset(
        num_samples=256,
        num_classes=10,
        image_size=(32, 32, 3),
        batch_size=32
    )

    # E) Compile the model
    #    This ensures we optimize BOTH the CNN's weights and augment_module.log_alpha
    full_model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # F) Train with model.fit, providing the tf.data.Dataset
    epochs = 5
    history = full_model.fit(
        train_ds,
        epochs=epochs
    )

    # G) Inspect the learned alpha after training
    alpha_pos = tf.nn.softplus(augment_module.log_alpha).numpy()
    print("Learned Dirichlet alpha:", alpha_pos)


if __name__ == "__main__":
    main()