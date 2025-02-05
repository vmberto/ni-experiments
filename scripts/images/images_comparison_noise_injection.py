import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets, layers
from layers.custom_gaussian_noise import CustomGaussianNoise
import numpy as np

output_dir = f'../../output'
os.makedirs(output_dir, exist_ok=True)

def normalize_image(image):
    return (image - image.min()) / (image.max() - image.min())

(x_train, y_train), _ = datasets.cifar10.load_data()

input_batch = tf.convert_to_tensor(x_train[:5] / 255.0)

gaussian_layer_fixed = layers.GaussianNoise(stddev=0.2)
gaussian_layer_dynamic = CustomGaussianNoise(max_stddev=0.2)

original_images = input_batch.numpy()
gaussian_fixed_images = gaussian_layer_fixed(input_batch, training=True).numpy()

gaussian_dynamic_images = []
dynamic_stddevs = []
for img in input_batch:
    img = tf.expand_dims(img, axis=0)
    processed_img = gaussian_layer_dynamic(img, training=True)
    gaussian_dynamic_images.append(processed_img[0].numpy())
    dynamic_stddevs.append(gaussian_layer_dynamic.last_stddev)

gaussian_dynamic_images = np.array(gaussian_dynamic_images)

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

    # Row 2: Gaussian (Fixed)
    axes[1, i].imshow(normalize_image(gaussian_fixed_images[i]))
    axes[1, i].axis("off")

    # Row 3: Gaussian (Dynamic)
    axes[2, i].imshow(normalize_image(gaussian_dynamic_images[i]))
    axes[2, i].set_title(f"σ={dynamic_stddevs[i]:.2f}",
                        pad=8,
                        fontdict=font_properties,
                        fontsize=20)
    axes[2, i].axis("off")

fig.suptitle("Gaussian Noise Visualization",
             fontsize=34,
             **font_properties,
             y=0.95)

fig.text(0.5, 0.86, "ORIGINAL", ha="center", fontsize=28, **font_properties)
fig.text(0.5, 0.57, "FIXED (σ=0.2)", ha="center", fontsize=28, **font_properties)
fig.text(0.5, 0.30, "DYNAMIC [0,0.2]", ha="center", fontsize=28, **font_properties)

grid_path = os.path.join(output_dir, "cifar10_gaussian_noise_grid_with_dynamic_stddev.png")
plt.savefig(grid_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Grid saved at: {grid_path}")