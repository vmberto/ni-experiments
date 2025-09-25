import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets
import keras_cv
from layers.custom_gaussian_noise import CustomGaussianNoise
from layers.random_salt_and_pepper import RandomSaltAndPepper
import numpy as np
import random
from PIL import Image
import io

# Parameters
output_dir = f'../../output'
os.makedirs(output_dir, exist_ok=True)
gif_path = os.path.join(output_dir, "cifar10_randaug_noise.pdf")
num_frames = 15

def normalize_image(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    return image

# Load a single CIFAR-10 image
(x_train, y_train), _ = datasets.cifar10.load_data()
input_image = tf.convert_to_tensor(x_train[0:1] / 255.0)
original_image = input_image[0].numpy()

# Shared RandAugment layer
randaugment_layer = keras_cv.layers.RandAugment(
    value_range=(0, 1), augmentations_per_image=3, magnitude=0.3, rate=1
)

# Store frames
frames = []

for i in range(num_frames):
    randaugmented = randaugment_layer(input_image, training=True)

    noise_layer = random.choice([
        CustomGaussianNoise(max_stddev=0.2),
        RandomSaltAndPepper(max_factor=0.3)
    ])

    randaugmented_plus_noise = noise_layer(randaugmented, training=True)

    randaug_img = randaugmented[0].numpy()
    randaug_noise_img = randaugmented_plus_noise[0].numpy()

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plt.subplots_adjust(top=0.75, bottom=0.05, left=0.05, right=0.95, wspace=0.4)

    images = [original_image, randaug_img, randaug_noise_img]
    titles = ["Original", "RandAugment", "RandAugment + Noise"]

    for j in range(3):
        axes[j].imshow(normalize_image(images[j]))
        axes[j].set_title(titles[j], fontsize=18, family='Times New Roman')
        axes[j].axis("off")

    # Add arrows between the images
    for idx in [0, 1]:
        x_start = axes[idx].get_position().x1
        x_end = axes[idx + 1].get_position().x0
        y_middle = axes[idx].get_position().y0 + axes[idx].get_position().height / 2
        fig.text(
            (x_start + x_end) / 2,
            y_middle,
            '→',
            ha='center',
            va='center',
            fontsize=24,
            fontweight='bold'
        )

    # Save to memory and add to frames
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    frame = Image.open(buf)
    frames.append(frame)

# Save GIF
frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=1000,
    loop=0
)

print(f"GIF saved at: {gif_path}")
