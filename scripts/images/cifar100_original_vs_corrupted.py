import os
import random
import matplotlib.pyplot as plt
import numpy as np
from keras import datasets

# === OUTPUT CONFIGURATION ===
output_dir = "../../output"
os.makedirs(output_dir, exist_ok=True)
pdf_path = os.path.join(output_dir, "cifar100_original_vs_corrupted_matched_labeled.pdf")

# === PATH TO CIFAR-100-C ===
CIFAR100_C_PATH = "../../dataset/CIFAR-100-C"  # change if needed

def normalize_image(image):
    """Normalize image to [0, 1] for visualization."""
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return np.clip(image, 0, 1)

# === LOAD CIFAR-100 TEST DATA ===
(_, _), (x_test, y_test) = datasets.cifar100.load_data(label_mode="fine")
x_test = x_test.astype("float32") / 255.0

# === CIFAR-100 LABEL NAMES (fine labels) ===
label_names = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
    "bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle",
    "chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard",
    "lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain",
    "mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree",
    "plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket","rose","sea",
    "seal","shark","shrew","skunk","skyscraper","snail","snake","spider","squirrel","streetcar","sunflower",
    "sweet_pepper","table","tank","telephone","television","tiger","tractor","train","trout","tulip","turtle",
    "wardrobe","whale","willow_tree","wolf","woman","worm"
]

# === SELECT RANDOM INDICES ===
num_images = 5
indices = random.sample(range(len(x_test)), num_images)
original_images = x_test[indices]
original_labels = [label_names[int(y_test[i])] for i in indices]

# === AVAILABLE CORRUPTIONS IN CIFAR-100-C ===
corruption_types = [
    "jpeg_compression", "brightness", "saturate", "gaussian_blur", "shot_noise", "pixelate"
]
selected_corruptions = random.sample(corruption_types, num_images)
selected_severities = [5 for _ in range(num_images)]

# === LOAD MATCHED CORRUPTED IMAGES ===
corrupted_images = []
corruption_labels = []

for idx, (corr, sev) in enumerate(zip(selected_corruptions, selected_severities)):
    corruption_file = os.path.join(CIFAR100_C_PATH, f"{corr}.npy")
    labels_file = os.path.join(CIFAR100_C_PATH, "labels.npy")

    if not os.path.exists(corruption_file):
        raise FileNotFoundError(f"Corruption file not found: {corruption_file}")

    images = np.load(corruption_file)
    labels = np.load(labels_file)

    # CIFAR-100-C has 50k images (10k per severity)
    start, end = (sev - 1) * 10000, sev * 10000
    x_corr = images[start:end].astype("float32") / 255.0

    corrupted_images.append(x_corr[indices[idx]])
    corruption_labels.append(corr.replace("_", " ").title())

# === PLOT ===
fig, axes = plt.subplots(2, num_images, figsize=(12, 7))
plt.subplots_adjust(
    left=0.01, right=0.99, top=0.85, bottom=0.05,
    wspace=0.05, hspace=0.45
)
font = {"family": "Times New Roman"}

# Top row: Original CIFAR-100 images with labels
for i in range(num_images):
    axes[0, i].imshow(normalize_image(original_images[i]))
    axes[0, i].axis("off")
    axes[0, i].set_title(original_labels[i].title(), fontsize=20, pad=8, **font)

# Bottom row: Corrupted counterparts
for i in range(num_images):
    axes[1, i].imshow(normalize_image(corrupted_images[i]))
    axes[1, i].axis("off")
    axes[1, i].set_title(corruption_labels[i], fontsize=20, pad=8, **font)

# Row labels
fig.text(0.5, 0.95, "(I) Original CIFAR-100 Test Samples", ha="center", fontweight="bold", fontsize=24, **font)
fig.text(0.5, 0.46, "(II) CIFAR-100-C Samples", ha="center", fontweight="bold", fontsize=24, **font)

# Save figure
plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
plt.show()

print(f"✅ Figure saved at: {pdf_path}")
