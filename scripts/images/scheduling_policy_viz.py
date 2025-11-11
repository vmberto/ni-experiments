import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from keras import datasets
import keras_cv
import numpy as np
from layers.custom_gaussian_noise import CustomGaussianNoise


# === Output Path ===
output_dir = "../../output"
os.makedirs(output_dir, exist_ok=True)
pdf_path = os.path.join(output_dir, "cifar10_curriculum_progression_fixed.pdf")


# === Utility Function ===
def normalize_image(image):
    image = (image - image.min()) / (image.max() - image.min())
    return (image * 255).astype(np.uint8)


# === Load Data ===
(x_train, _), _ = datasets.cifar10.load_data()
input_image = tf.convert_to_tensor(x_train[0:1] / 255.0)
original_image = input_image[0].numpy()


# === RandAugment and Noise Layers ===
randaugment_layer = keras_cv.layers.RandAugment(
    value_range=(0, 1), augmentations_per_image=3, magnitude=0.3, rate=1
)

randaugmented = randaugment_layer(input_image, training=True)

noise_half = CustomGaussianNoise(max_stddev=0.1)
noise_full = CustomGaussianNoise(max_stddev=0.2)

randaug_half = noise_half(randaugmented, training=True)
randaug_full = noise_full(randaugmented, training=True)


# === Image List ===
images = [
    original_image,
    randaugmented[0].numpy(),
    randaug_half[0].numpy(),
    randaug_full[0].numpy(),
]
titles = [
    "Original",
    "RandAugment",
    "RA + Gauss/2",
    "RA + Gauss/Full",
]
epochs = [
    "",  # Original
    "Stage 1\nES = 3 epochs",
    "Stage 2\nES = 5 epochs",
    "Stage 3\nES = 8 epochs",
]


# === Create Figure ===
fig, axes = plt.subplots(1, 4, figsize=(11, 3))
plt.subplots_adjust(top=0.75, bottom=0.20, left=0.04, right=0.96, wspace=0.5)

# Plot each image
for i, ax in enumerate(axes):
    ax.imshow(normalize_image(images[i]))
    ax.axis("off")
    ax.set_title(titles[i], fontsize=24, family="Times New Roman", pad=8)
    if epochs[i]:
        ax.text(
            0.55, -0.07, epochs[i],
            fontweight="bold",
            ha="center", va="top",
            fontsize=18, family="Times New Roman", transform=ax.transAxes,
        )

# === Add arrows ===
for i in range(3):
    x_start = axes[i].get_position().x1
    x_end = axes[i + 1].get_position().x0
    y_middle = axes[i].get_position().y0 + axes[i].get_position().height / 2
    if i == 0:
        fig.text(
            ((x_start + x_end) / 2) - 0.01, y_middle,
            "→",
            ha="center", va="center",
            fontsize=42, fontweight="bold",
        )
    else:
        fig.text(
            ((x_start + x_end) / 2), y_middle,
            "→",
            ha="center", va="center",
            fontsize=42, fontweight="bold",
        )
# === Dashed Box (Model Training) ===
box_x0 = axes[1].get_position().x0 - 0.03
box_y0 = axes[1].get_position().y0 - 0.27
box_x1 = axes[-1].get_position().x1 + 0.04
box_y1 = axes[1].get_position().y1 + 0.18

rect = patches.Rectangle(
    (box_x0, box_y0),
    box_x1 - box_x0,
    box_y1 - box_y0,
    linewidth=1.5,
    edgecolor="black",
    facecolor="none",
    linestyle="--",
    transform=fig.transFigure,
)
fig.add_artist(rect)

# === Add "Model Training" label ===
fig.text(
    (box_x0 + box_x1) / 2,
    box_y1 + 0.05,
    "Model Training Pipeline",
    ha="center", va="bottom",
    fontsize=26, fontweight="bold", family="Times New Roman",
)

# === Add Global "ES" Label ===
fig.text(
    0.03, 0.035,
    "ES = Early Stopping",
    fontweight="bold",
    ha="left", va="bottom",
    fontsize=18, color="black", family="Times New Roman",
)

# === Save PDF ===
plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
plt.show()
plt.close(fig)

print(f"PDF saved at: {pdf_path}")