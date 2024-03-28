from matplotlib import pyplot as plt
import tensorflow as tf
import os

output_folder = './output'
os.makedirs(output_folder, exist_ok=True)


def save_img_examples(dataset):
    count = 0
    for image, label in dataset.take(5):
        image_path = os.path.join(output_folder, f'image_{count}.png')
        plt.imsave(image_path, image.numpy(), cmap='brg')
        count += 1
