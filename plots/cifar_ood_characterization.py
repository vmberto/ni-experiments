import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter

from lib.helpers import seaborn_styles

rcParams['font.size'] = 22
rcParams['axes.labelsize'] = 28
rcParams['xtick.labelsize'] = 24
rcParams['ytick.labelsize'] = 24
rcParams['legend.fontsize'] = 18


def bootstrap_ci(data, n_bootstrap=1000, ci_level=0.95):
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(boot_sample))
    return np.mean(boot_means), np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)


def categorize_by_percentiles(pct_value):
    if pct_value <= 33:
        return 'Lowest'
    elif pct_value <= 66:
        return 'Mid-Range'
    else:
        return 'Highest'


datasets = ['cifar10', 'cifar100']
fig, axes = plt.subplots(1, 2, figsize=(28, 10), sharey=True)  # sharey ensures scales are identical

for i, dataset in enumerate(datasets):
    ax = axes[i]
    file_path = f'../results/{dataset}/{dataset}_{dataset}c_full_characterization.csv'
    data = pd.read_csv(file_path)

    # --- Processing (Same as your script) ---
    bootstrap_results = data.groupby('corruption_type')['wasserstein_per_feature'].apply(
        lambda x: bootstrap_ci(x.values))
    grouped_data = pd.DataFrame(bootstrap_results.tolist(), index=bootstrap_results.index,
                                columns=['mean', 'lower', 'upper'])
    grouped_data['percentile'] = grouped_data['mean'].rank(pct=True) * 100
    grouped_data['category'] = grouped_data['percentile'].apply(categorize_by_percentiles)

    category_bootstrap = grouped_data.groupby('category')['mean'].apply(lambda x: bootstrap_ci(x.values))
    category_stats = pd.DataFrame(category_bootstrap.tolist(), index=category_bootstrap.index,
                                  columns=['mean', 'lower', 'upper'])

    cifar_point = pd.DataFrame({'percentile': [0], 'mean': [0], 'lower': [0], 'upper': [0]})
    combined_data = pd.concat([cifar_point, grouped_data.sort_values('percentile')], ignore_index=True)

    # --- Plotting ---
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    sns.lineplot(x='percentile', y='mean', data=combined_data, color='purple', linewidth=2.5, alpha=0.6,
                 label=f'{dataset.upper()} Trend', ax=ax)

    for _, row in grouped_data.iterrows():
        ax.errorbar(row['percentile'], row['mean'],
                    yerr=[[max(0, row['mean'] - row['lower'])], [row['upper'] - row['mean']]], fmt='none',
                    ecolor='black', capsize=3, alpha=0.15)

    sns.scatterplot(x='percentile', y='mean', data=grouped_data, color='red', s=110, edgecolor='white', ax=ax)

    # Reference Point (CIFAR10 or CIFAR100)
    ax.scatter(0, 0, color='blue', s=200, zorder=5)
    ax.text(0, -0.05, f'{dataset.upper()}', ha='center', va='top', color='blue', fontweight='bold', fontsize=32,
            transform=ax.get_xaxis_transform())

    # Annotations & Polish
    for cat, pos in zip(['Lowest', 'Mid-Range', 'Highest'], [16.5, 49.5, 83]):
        count = int((grouped_data['category'] == cat).sum())
        m, l, u = category_stats.loc[cat]
        ax.text(pos, 0.92, f'{cat}', ha='center', va='bottom', fontsize=32, fontweight='bold',
                transform=ax.get_xaxis_transform())
        ax.text(pos, 0.90, f'{count} types\n{m:.3f} ({l:.3f}, {u:.3f})', ha='center', va='top', fontsize=20,
                transform=ax.get_xaxis_transform())

    for x_val in [33, 66]:
        ax.axvline(x=x_val, color='black', linestyle='-', linewidth=1.2, alpha=0.3)

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 0.14)
    ax.set_xlabel('Percentiles of Corruption Severity', fontsize=28)

    if i == 0: ax.set_ylabel('Wasserstein Distance', fontsize=32)
    ax.set_xticks([0, 33, 66, 100])

    handles, labels = ax.get_legend_handles_labels()
    trend_handle = [h for h, l in zip(handles, labels) if l == f'{dataset.upper()} Trend']

    if trend_handle:
        ax.legend(
            handles=trend_handle,
            labels=[f'{dataset.upper()}-C Trend'],
            loc='lower right',  # Sets position to bottom right
            bbox_to_anchor=(0.98, 0.02),  # Fine-tunes the offset from the axes edges
            frameon=True,  # Adds a box around the legend
            facecolor='white',  # Ensures the background is solid
            edgecolor='black',  # Legend border color
            framealpha=0.8  # Slight transparency
        )

plt.tight_layout()
plt.savefig('../results/combined_wasserstein_characterization.pdf', bbox_inches='tight')
