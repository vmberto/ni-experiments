import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets, models
import keras_cv
from layers.custom_gaussian_noise import CustomGaussianNoise
import numpy as np
from layers.random_salt_and_pepper import RandomSaltAndPepper


output_dir = f'../../output'
os.makedirs(output_dir, exist_ok=True)


def normalize_image(image):
    return (image - image.min()) / (image.max() - image.min())


(x_train, y_train), _ = datasets.cifar10.load_data()

input_batch = tf.convert_to_tensor(x_train[:5] / 255.0)

rand_gauss = models.Sequential([
    keras_cv.layers.RandAugment(value_range=(0, 1), augmentations_per_image=3, magnitude=0.3, rate=1),
    CustomGaussianNoise(max_stddev=0.2)
])
rand_sp = models.Sequential([
    keras_cv.layers.RandAugment(value_range=(0, 1), augmentations_per_image=3, magnitude=0.3, rate=1),
    RandomSaltAndPepper(max_factor=.3)
])

original_images = input_batch.numpy()

rand_gauss_images = []
rand_sp_images = []
for img in input_batch:
    img = tf.expand_dims(img, axis=0)
    processed_img = rand_gauss(img, training=True)
    rand_gauss_images.append(processed_img[0].numpy())
    processed_img = rand_sp(img, training=True)
    rand_sp_images.append(processed_img[0].numpy())

fig, axes = plt.subplots(3, 5, figsize=(15, 12))

plt.subplots_adjust(
    top=0.85,
    bottom=0.05,
    left=0.05,
    right=0.95,
    hspace=0.4,
    wspace=0.05
)

font_properties = {
    'family': 'Times New Roman',
    'weight': 'normal'
}

for i in range(5):
    # Row 1: Original images
    axes[0, i].imshow(normalize_image(original_images[i]))
    axes[0, i].axis("off")

    axes[1, i].imshow(normalize_image(rand_sp_images[i]))
    axes[1, i].axis("off")

    axes[2, i].imshow(normalize_image(rand_gauss_images[i]))
    axes[2, i].set_title(f"")
    axes[2, i].axis("off")

fig.text(0.5, 0.86, "ORIGINAL", ha="center", fontsize=28, **font_properties)
fig.text(0.5, 0.57, "RandAugment+S&P", ha="center", fontsize=28, **font_properties)
fig.text(0.5, 0.28, "RandAugment+Gaussian", ha="center", fontsize=28, **font_properties)

grid_path = os.path.join(output_dir, "cifar10_gaussian_noise_grid_with_dynamic_stddev.png")
plt.savefig(grid_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Grid saved at: {grid_path}")