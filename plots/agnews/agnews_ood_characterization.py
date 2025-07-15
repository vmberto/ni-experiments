import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from lib.consts import AGNEWS_CORRUPTIONS
from lib.helpers import seaborn_styles
from models.text_autoencoder import TextAutoencoder
from dataset.agnewsdataset import AGNewsDataset



# Configuration
MAX_SEQUENCE_LENGTH = 128
VOCAB_SIZE = 20000
BATCH_SIZE = 256
EMBEDDING_DIM = 128
LATENT_DIM = 128
CATEGORY_PATH = "../../results/agnews/agnews_encoder_kldiv_categories.csv"
PLOT_PATH = "../../results/agnews/agnews_c_kldiv_characterization_plot.pdf"


def categorize_by_percentiles(pct_value):
    if pct_value <= 25:
        return 'Lowest'
    elif 25 < pct_value <= 75:
        return 'Mid-Range'
    else:
        return 'Highest'


def compute_kl_divergences(dataset, encoder, test_ds):
    results = []
    print("Calculating KL divergences...")
    for corruption_type in AGNEWS_CORRUPTIONS:
        print(f"  Processing corruption: {corruption_type}")
        kld = dataset.prepare_agnews_c_with_distances(encoder, corruption_type, test_ds)
        results.append({"corruption_type": corruption_type, "divergence": kld})
        print(f"Saved interim results after corruption: {corruption_type}")
    return pd.DataFrame(results)


def analyze_and_categorize(data, category_path):
    grouped = data.groupby('corruption_type')['divergence'].mean().to_frame(name='mean')
    percentiles = grouped['mean'].rank(pct=True) * 100
    grouped['category'] = percentiles.apply(categorize_by_percentiles)
    grouped.to_csv(category_path)
    print("Categorized divergences saved to:", category_path)
    return grouped, percentiles


def plot_results(grouped_data, percentiles, save_path):
    seaborn_styles(sns)
    rcParams['font.size'] = 24  # General font size
    rcParams['axes.titlesize'] = 32  # Title font size
    rcParams['axes.labelsize'] = 28  # X and Y label font size
    rcParams['xtick.labelsize'] = 26  # X tick font size
    rcParams['ytick.labelsize'] = 26  # Y tick font size
    rcParams['legend.fontsize'] = 18  # Legend font size

    grouped_data = grouped_data.copy()
    grouped_data['percentile'] = percentiles
    grouped_data = grouped_data.sort_values('percentile')

    counts = grouped_data['category'].value_counts()
    means_by_cat = grouped_data.groupby('category')['mean'].mean().to_dict()

    # Add AG-NEWS reference as first point (percentile = 0, mean = 0)
    ag_news_point = pd.DataFrame({'percentile': [0], 'mean': [0]})
    combined_data = pd.concat([ag_news_point, grouped_data], ignore_index=True)
    plt.figure(figsize=(14, 10))

    plt.scatter(0, 0, color='blue', s=160, zorder=5)
    plt.text(0, -0.6, 'AG-NEWS', ha='center', color='blue', fontsize=22, fontweight='bold')

    # Line connecting AG-NEWS to all points
    sns.lineplot(
        x='percentile', y='mean',
        data=combined_data,
        color='purple',
        label='AG-NEWS-C Trend',
        linewidth=2
    )

    # Scatterplot for the small points (all red)
    sns.scatterplot(
        x='percentile', y='mean',
        data=grouped_data,
        color='red',
        alpha=0.8,
        s=80
    )

    # Annotate each category on correct percentile
    y_max = grouped_data['mean'].max()
    y_offset = y_max * 0.05
    for cat, pos in zip(['Lowest', 'Mid-Range', 'Highest'], [12.5, 50, 87.5]):
        count = counts.get(cat, 0)
        mean = means_by_cat.get(cat, 0)
        plt.text(pos, y_max + y_offset * 2.5, f'{cat}', ha='center', fontsize=22, fontweight='bold')
        plt.text(pos, y_max + y_offset * 0.5, f'{count} corruptions\n{mean:.2f}', ha='center', fontsize=22)

    # Vertical lines for reference
    plt.axvline(x=25, color='grey', linestyle='--', linewidth=1)
    plt.axvline(x=75, color='grey', linestyle='--', linewidth=1)

    # Adjust axes
    plt.xlim(0, 101)
    plt.ylim(0, 9)  # increased Y-axis
    plt.xlabel('Percentiles')
    plt.ylabel('KL Divergence (AG-NEWS-C)')
    plt.xticks(ticks=[25, 75])
    plt.title('KL Divergence Distribution Across Percentiles')

    # Legend in bottom right
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print("Plot saved to:", save_path)


def main():
    # dataset = AGNewsDataset(
    #     max_sequence_length=MAX_SEQUENCE_LENGTH,
    #     vocab_size=VOCAB_SIZE,
    #     batch_size=BATCH_SIZE
    # )
    #
    # x_test = dataset.get_testset_for_autoencoder()
    # test_ds = dataset.get_dataset_for_autoencoder(x_test)
    #
    # encoder = TextAutoencoder(
    #     vocab_size=VOCAB_SIZE,
    #     embedding_dim=EMBEDDING_DIM,
    #     max_len=MAX_SEQUENCE_LENGTH,
    #     latent_dim=LATENT_DIM
    # )

    # results_df = compute_kl_divergences(dataset, encoder, test_ds)
    results_df = pd.read_csv('../../results/agnews/agnews_encoder_kldiv.csv')
    grouped_data, percentiles = analyze_and_categorize(results_df, CATEGORY_PATH)
    plot_results(grouped_data, percentiles, PLOT_PATH)


if __name__ == "__main__":
    main()
