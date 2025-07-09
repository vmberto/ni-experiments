import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from lib.consts import AGNEWS_CORRUPTIONS
from models.text_autoencoder import TextAutoencoder
from dataset.agnewsdataset import AGNewsDataset

# Configuration
MAX_SEQUENCE_LENGTH = 128
VOCAB_SIZE = 20000
BATCH_SIZE = 256
EMBEDDING_DIM = 128
LATENT_DIM = 128
RESULT_PATH = "../../output/agnews_encoder_kldiv.csv"
CATEGORY_PATH = "../../output/agnews_encoder_kldiv_categories.csv"
PLOT_PATH = "../../output/agnews_encoder_kldiv_plot.png"

# Plot styling
rcParams['font.size'] = 24
rcParams['axes.titlesize'] = 32
rcParams['axes.labelsize'] = 28
rcParams['xtick.labelsize'] = 26
rcParams['ytick.labelsize'] = 26
rcParams['legend.fontsize'] = 18


def bootstrap_ci(data, n_bootstrap=1000, ci_level=0.95):
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(boot_sample))
    lower_bound = np.percentile(boot_means, (1 - ci_level) / 2 * 100)
    upper_bound = np.percentile(boot_means, (1 + ci_level) / 2 * 100)
    return np.mean(boot_means), lower_bound, upper_bound


def categorize_by_percentiles(pct_value):
    if pct_value <= 25:
        return 'Lowest'
    elif 25 < pct_value <= 75:
        return 'Mid-Range'
    else:
        return 'Highest'


def compute_kl_divergences(dataset, encoder, test_ds, save_path):
    results = []
    print("Calculating KL divergences...")
    for corruption_type in AGNEWS_CORRUPTIONS:
        print(f"  Processing corruption: {corruption_type}")
        kld = dataset.prepare_agnews_c_with_distances(encoder, corruption_type, test_ds)
        results.append({"corruption_type": corruption_type, "divergence": kld})
        pd.DataFrame(results).to_csv(save_path, index=False)
        print(f"Saved interim results after corruption: {corruption_type}")
    return pd.DataFrame(results)


def analyze_bootstrap_and_categorize(data, category_path):
    bootstrap_results = data.groupby('corruption_type')['divergence'].apply(
        lambda x: bootstrap_ci(x.values))
    grouped = pd.DataFrame(bootstrap_results.tolist(), index=bootstrap_results.index,
                           columns=['mean', 'lower', 'upper'])
    percentiles = grouped['mean'].rank(pct=True) * 100
    grouped['category'] = percentiles.apply(categorize_by_percentiles)
    grouped.to_csv(category_path)
    print("Categorized divergences saved to:", category_path)
    return grouped, percentiles


def get_category_summary(grouped):
    category_bootstrap = grouped.groupby('category')['mean'].apply(lambda x: bootstrap_ci(x.values))
    return pd.DataFrame(category_bootstrap.tolist(), index=category_bootstrap.index,
                        columns=['mean', 'lower', 'upper'])


def plot_results(grouped_data, percentiles, category_stats, save_path):
    low_count = (percentiles <= 25).sum()
    mid_range_count = ((percentiles > 25) & (percentiles <= 75)).sum()
    high_count = (percentiles > 75).sum()

    plt.figure(figsize=(14, 10))

    plt.scatter(0, 0, color='blue', s=140, label='AG-NEWS')
    plt.text(0, -80, 'AG-NEWS', ha='center', color='blue', fontsize=22, fontweight='bold')

    sorted_percentiles = percentiles.sort_values()
    sorted_means = grouped_data.loc[sorted_percentiles.index, 'mean']
    sorted_lower = grouped_data.loc[sorted_percentiles.index, 'lower']
    sorted_upper = grouped_data.loc[sorted_percentiles.index, 'upper']

    colors = plt.cm.Reds(np.linspace(0.5, 1, len(sorted_percentiles)))
    for i, (x, y, lower, upper) in enumerate(zip(sorted_percentiles, sorted_means, sorted_lower, sorted_upper)):
        plt.errorbar(x, y, fmt='o', color=colors[i], ecolor='gray', capsize=5, alpha=0.7)

    plt.plot(sorted_percentiles, sorted_means, linestyle='-', color='purple', label='AG-NEWS-C Trend')

    text_y_position = grouped_data['upper'].max() - 100
    for category, pos in zip(['Lowest', 'Mid-Range', 'Highest'], [12.5, 50, 87.5]):
        count = {'Lowest': low_count, 'Mid-Range': mid_range_count, 'Highest': high_count}[category]
        mean = category_stats.loc[category, 'mean']
        lower = category_stats.loc[category, 'lower']
        upper = category_stats.loc[category, 'upper']
        plt.text(pos, text_y_position - 60, f'{category}', ha='center', fontsize=22, fontweight='bold')
        plt.text(pos, text_y_position - 0.25,
                 f'{count} corruptions\n{int(mean)} ({int(lower)}, {int(upper)})',
                 ha='center', fontsize=22)

    plt.axvline(x=25, color='grey', linestyle='--', linewidth=1)
    plt.axvline(x=75, color='grey', linestyle='--', linewidth=1)

    plt.xlim(0, 101)
    plt.ylim(0)
    plt.xlabel('Percentiles')
    plt.ylabel('KL Divergence (AG-NEWS-C)')
    plt.xticks(ticks=[25, 75])
    plt.title('Distribution Across Percentiles')
    plt.legend(loc='center left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print("Plot saved to:", save_path)


def main():
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    dataset = AGNewsDataset(
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
        batch_size=BATCH_SIZE
    )

    x_test = dataset.get_testset_for_autoencoder()
    test_ds = dataset.get_dataset_for_autoencoder(x_test)

    # We observed that the untrained encoder — despite its lack of semantic understanding — retained structural
    # differences introduced by corruption, enabling more effective characterization of distribution shift than a
    # trained AE encoder.
    # ------
    # Untrained encoders fail to capture the effects of image corruptions in CIFAR-10,
    # unlike in AG-NEWS where surface-level structure (like token order or insertion) is preserved.
    encoder = TextAutoencoder(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        max_len=MAX_SEQUENCE_LENGTH,
        latent_dim=LATENT_DIM
    )

    results_df = compute_kl_divergences(dataset, encoder, test_ds, RESULT_PATH)
    grouped_data, percentiles = analyze_bootstrap_and_categorize(results_df, CATEGORY_PATH)
    category_stats = get_category_summary(grouped_data)
    plot_results(grouped_data, percentiles, category_stats, PLOT_PATH)


if __name__ == "__main__":
    main()