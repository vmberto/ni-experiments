from matplotlib import pyplot as plt
import os

output_folder = f'{os.getcwd()}/output'
os.makedirs(output_folder, exist_ok=True)


def save_img_examples(dataset, img_num=3):
    sample_images, _ = next(iter(dataset))
    plt.figure(figsize=(10, 10))
    for i, image in enumerate(sample_images[:img_num]):
        image_path = os.path.join(f'{os.getcwd()}/output', f'image_{i}.png')

        image = (image.numpy() - image.numpy().min()) / (image.numpy().max() - image.numpy().min())

        plt.imsave(image_path, image.astype("float32"))
        plt.axis("off")
        plt.clf()
