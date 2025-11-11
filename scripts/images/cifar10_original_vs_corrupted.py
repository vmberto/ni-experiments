import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras import datasets

# === OUTPUT CONFIGURATION ===
output_dir = "../../output"
os.makedirs(output_dir, exist_ok=True)
pdf_path = os.path.join(output_dir, "cifar10_original_vs_corrupted_matched_labeled.pdf")

def normalize_image(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return np.clip(image, 0, 1)

# CIFAR-10 label names
cifar10_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# === LOAD CIFAR-10 TEST DATA ===
corrupted_ds = tfds.load("cifar10", split="test", as_supervised=True)
x_test = np.array([img for img, _ in tfds.as_numpy(corrupted_ds)]).astype("float32") / 255.0
y_test = np.array([label for _, label in tfds.as_numpy(corrupted_ds)])

# Select 5 random fixed indices (same for clean and corrupted)
num_images = 5
indices = random.sample(range(len(x_test)), num_images)
original_images = x_test[indices]
original_labels = [cifar10_labels[int(y_test[i])] for i in indices]

# === VALID CIFAR-10-C CORRUPTION TYPES ===
corruption_types = [
    "fog", "elastic", "spatter", "impulse_noise", "zoom_blur", "contrast"
]

selected_corruptions = random.sample(corruption_types, num_images)
selected_severities = [5 for _ in range(num_images)]

corrupted_images = []
corruption_labels = []

# === LOAD ONE IMAGE PER CORRUPTION USING SAME INDEX AS ORIGINAL ===
for idx, (corr, sev) in enumerate(zip(selected_corruptions, selected_severities)):
    corruption_name = f"{corr}_{sev}"
    corrupted_ds = tfds.load(f"cifar10_corrupted/{corruption_name}", split="test", as_supervised=True)
    corrupted_images_full = np.array([img for img, _ in tfds.as_numpy(corrupted_ds)]).astype("float32") / 255.0

    corrupted_images.append(corrupted_images_full[indices[idx]])
    corruption_labels.append(f"{corr.replace('_', ' ').title()}")

# === PLOT GRID ===
fig, axes = plt.subplots(2, num_images, figsize=(12, 7))
plt.subplots_adjust(
    left=0.01,   # less margin on the left
    right=0.99,  # less margin on the right
    top=0.85,    # reduce top whitespace
    bottom=0.05, # reduce bottom whitespace
    wspace=0.05, # less horizontal space between images
    hspace=0.45  # less vertical space
)
font = {"family": "Times New Roman"}

# Top row: Original CIFAR-10 images with labels
for i in range(num_images):
    axes[0, i].imshow(normalize_image(original_images[i]))
    axes[0, i].axis("off")
    axes[0, i].set_title(original_labels[i].title(), fontsize=24, pad=8, **font)

# Bottom row: Same images but corrupted
for i in range(num_images):
    axes[1, i].imshow(normalize_image(corrupted_images[i]))
    axes[1, i].axis("off")
    axes[1, i].set_title(corruption_labels[i], fontsize=24, pad=8, **font)

# Row labels
fig.text(0.5, 0.95, "(I) Original CIFAR-10 Test Samples", fontweight="bold", ha="center", fontsize=24, **font)
fig.text(0.5, 0.46, "(II) CIFAR-10-C Samples", ha="center", fontweight="bold", fontsize=24, **font)

# Save figure
plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
plt.show()

print(f"✅ Figure saved at: {pdf_path}")
