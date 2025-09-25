import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets, models
import keras_cv
import numpy as np
from layers.custom_gaussian_noise import CustomGaussianNoise
from layers.random_salt_and_pepper import RandomSaltAndPepper

# Diretório de saída
output_dir = f'../../output'
os.makedirs(output_dir, exist_ok=True)

# Normalização simples
def normalize_image(image):
    return (image - image.min()) / (image.max() - image.min())

# Carregar dados
(x_train, y_train), _ = datasets.cifar10.load_data()
input_batch = tf.convert_to_tensor(x_train[:5] / 255.0, dtype=tf.float32)

# Camadas de aumento
RandAugment = keras_cv.layers.RandAugment(value_range=(0, 1), augmentations_per_image=3, magnitude=0.3, rate=1)

randaugment = models.Sequential([RandAugment])
rand_gauss = models.Sequential([
    RandAugment,
    CustomGaussianNoise(max_stddev=0.2)
])
rand_sp = models.Sequential([
    RandAugment,
    RandomSaltAndPepper(max_factor=0.3)
])

# Listas de imagens e parâmetros
original_images = input_batch.numpy()
randaugment_images = []
rand_gauss_images = []
rand_sp_images = []
stddev_values = []
sp_factors = []

# Processar cada imagem individualmente com valores únicos
for img in input_batch:
    img = tf.expand_dims(img, axis=0)

    # RandAugment simples
    processed_img = randaugment(img, training=True)
    randaugment_images.append(processed_img[0].numpy())

    # RandAugment + Gaussian Noise com stddev aleatório
    stddev = np.round(np.random.uniform(0.05, 0.2), 3)
    noisy_img = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=stddev, dtype=tf.float32)
    gauss_augmented = randaugment(img, training=True) + noisy_img
    rand_gauss_images.append(tf.clip_by_value(gauss_augmented[0], 0, 1).numpy())
    stddev_values.append(stddev)

    # RandAugment + Salt & Pepper com fator aleatório
    factor = np.round(np.random.uniform(0.1, 0.3), 3)
    sp_augmented = randaugment(img, training=True)
    binary_mask = tf.cast(tf.random.uniform(tf.shape(img), dtype=tf.float32) < factor, tf.float32)
    s_and_p_noise = tf.where(
        tf.random.uniform(tf.shape(img), dtype=tf.float32) < 0.5,
        tf.zeros_like(img, dtype=tf.float32),
        tf.ones_like(img, dtype=tf.float32)
    )
    sp_noisy = (1 - binary_mask) * sp_augmented + binary_mask * s_and_p_noise
    rand_sp_images.append(tf.clip_by_value(sp_noisy[0], 0, 1).numpy())
    sp_factors.append(factor)

# Criar figura
fig, axes = plt.subplots(4, 5, figsize=(15, 18))
plt.subplots_adjust(top=0.94, bottom=0.0, left=0.01, right=0.99, hspace=0.25, wspace=0.05)

font_properties = {'family': 'Times New Roman', 'weight': 'normal'}

# Plotar
for i in range(5):
    # Original
    axes[0, i].imshow(normalize_image(original_images[i]))
    axes[0, i].axis("off")

    # RandAugment
    axes[1, i].imshow(normalize_image(randaugment_images[i]))
    axes[1, i].axis("off")

    # RandAugment + S&P
    axes[2, i].imshow(normalize_image(rand_sp_images[i]))
    axes[2, i].axis("off")
    axes[2, i].set_title(f"α={sp_factors[i]:.2f}", fontsize=32)

    # RandAugment + Gaussian
    axes[3, i].imshow(normalize_image(rand_gauss_images[i]))
    axes[3, i].axis("off")
    axes[3, i].set_title(f"σ={stddev_values[i]:.2f}", fontsize=28)

# Títulos das linhas
fig.text(0.5, 0.93, "(a) Original", ha="center", fontsize=52, **font_properties)
fig.text(0.5, 0.69, "(b) RandAugment", ha="center", fontsize=52, **font_properties)
fig.text(0.5, 0.47, "(c) RandAugment+S&P", ha="center", fontsize=52, **font_properties)
fig.text(0.5, 0.22, "(d) RandAugment+Gaussian", ha="center", fontsize=52, **font_properties)

# Salvar
grid_path = os.path.join(output_dir, "cifar10_noise_grid_labeled.pdf")
plt.savefig(grid_path, bbox_inches='tight', dpi=300)
plt.show()
print(f"✅ Grid saved at: {grid_path}")
