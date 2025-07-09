import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font family and increase font sizes
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 24  # General font size
rcParams['axes.titlesize'] = 32  # Title font size
rcParams['axes.labelsize'] = 28  # X and Y label font size
rcParams['xtick.labelsize'] = 26  # X tick font size
rcParams['ytick.labelsize'] = 26  # Y tick font size
rcParams['legend.fontsize'] = 18  # Legend font size

# file_path = '../../results/agnews/agnews_c_divergences.csv'  # Replace with your actual file path
file_path = '../../output/agnews_encoder_kldiv.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)


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


bootstrap_results = data.groupby('corruption_type')['divergence'].apply(
    lambda x: bootstrap_ci(x.values))

grouped_data = pd.DataFrame(bootstrap_results.tolist(), index=bootstrap_results.index, columns=['mean', 'lower', 'upper'])
percentiles = grouped_data['mean'].rank(pct=True) * 100

grouped_data['category'] = percentiles.apply(categorize_by_percentiles)

grouped_data.to_csv('../../results/agnews/agnews_c_divergences_categories.csv')

low_count = (percentiles <= 25).sum()
mid_range_count = ((percentiles > 25) & (percentiles <= 75)).sum()
high_count = (percentiles > 75).sum()

category_bootstrap = grouped_data.groupby('category')['mean'].apply(lambda x: bootstrap_ci(x.values))
category_stats = pd.DataFrame(category_bootstrap.tolist(), index=category_bootstrap.index, columns=['mean', 'lower', 'upper'])

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
    plt.text(pos, text_y_position - 60,
             f'{category}', ha='center', fontsize=22, fontweight='bold')
    plt.text(pos, text_y_position - 0.25,
             f'{count} corruptions\n{int(mean)} ({int(lower)}, {int(upper)})',
             ha='center', fontsize=22)

plt.axvline(x=25, color='grey', linestyle='--', linewidth=1)
plt.axvline(x=75, color='grey', linestyle='--', linewidth=1)

plt.xlim(0, 101)
plt.ylim(0)
plt.xlabel('Percentiles')
plt.ylabel('Levenshtein Distance (AG-NEWS-C)')
plt.xticks(ticks=[25, 75])
plt.title('Distribution Across Percentiles')

plt.legend(loc='center left')

plt.tight_layout()
plt.show()
