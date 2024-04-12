import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
sns.set_context("paper")
sns.set(font='serif')
sns.set_style("white", {
  "font.family": "serif",
  "font.serif": ["Times", "Palatino", "serif"],
})

df = pd.read_csv('output/resnet_output.csv')

grouped = df.groupby('Execution Name')

averages_total = grouped[['Fscore']].mean().reset_index()

averages_total.to_csv('./output/averages_total.csv', index=False)

grouped = df.groupby(['Execution Name', 'Corruption Type'])

averages_per_corruption = grouped[['Fscore']].mean().reset_index()

averages_per_corruption.to_csv('./output/averages_per_corruption.csv', index=False)

plt.figure(figsize=(12, 10))
ax = sns.pointplot(
    data=df,
    x='Fscore',
    y='Corruption Type',
    hue='Execution Name',
    linestyles='none',
    errorbar='ci',
    err_kws={'linewidth': 3},
    dodge=.6
)
ax.set_xlabel("Fscore", fontsize=24)
ax.set_ylabel("Corruption Type", fontsize=24)
ax.tick_params(labelsize=24)
sns.despine(left=True, bottom=True)
ax.legend(ncol=1, loc="lower left", fontsize=17, frameon=True)
plt.tight_layout()
plt.savefig(f'{os.getcwd()}/output/ci.png')
plt.clf()


# def save_images(dataset, num_images=5, output_dir='./output'):
#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Get the first 'num_images' samples from the dataset
#     image_count = 0
#     for image, label in dataset:
#         # Save only 'num_images' images
#         if image_count >= num_images:
#             break
#
#         # Save the image
#         plt.imshow(image.numpy())
#         plt.axis('off')
#         plt.savefig(os.path.join(output_dir, f'image_{image_count}.png'))
#         plt.close()
#
#         image_count += 1
#
#
# x, y, splits = get_cifar10_kfold_splits()
#
# for fold_number, (train_index, val_index) in splits:
#     x_train_fold, y_train_fold = x[train_index], y[train_index]
#     x_val_fold, y_val_fold = x[val_index], y[val_index]
#
#     train_ds = get_cifar10_dataset(x_train_fold, y_train_fold, [])
#     val_ds = get_cifar10_dataset(x_val_fold, y_val_fold)
#
#     save_images(train_ds, num_images=5, output_dir='./output')
