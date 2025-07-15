import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

from lib.helpers import seaborn_styles

seaborn_styles(sns)
rcParams['font.size'] = 24  # General font size
rcParams['axes.titlesize'] = 32  # Title font size
rcParams['axes.labelsize'] = 28  # X and Y label font size
rcParams['xtick.labelsize'] = 26  # X tick font size
rcParams['ytick.labelsize'] = 26  # Y tick font size
rcParams['legend.fontsize'] = 18  # Legend font size

# Load data
file_path = '../../results/cifar10/cifar10_c_divergences.csv'
data = pd.read_csv(file_path)

# Bootstrap function
def bootstrap_ci(data, n_bootstrap=1000, ci_level=0.95):
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(boot_sample))
    lower_bound = np.percentile(boot_means, (1 - ci_level) / 2 * 100)
    upper_bound = np.percentile(boot_means, (1 + ci_level) / 2 * 100)
    return np.mean(boot_means), lower_bound, upper_bound

# Categorization by percentile
def categorize_by_percentiles(pct_value):
    if pct_value <= 25:
        return 'Lowest'
    elif 25 < pct_value <= 75:
        return 'Mid-Range'
    else:
        return 'Highest'

# Compute bootstrap results
bootstrap_results = data.groupby('corruption_type')['kl_divergences'].apply(
    lambda x: bootstrap_ci(x.values))

grouped_data = pd.DataFrame(bootstrap_results.tolist(), index=bootstrap_results.index, columns=['mean', 'lower', 'upper'])
percentiles = grouped_data['mean'].rank(pct=True) * 100
grouped_data['percentile'] = percentiles
grouped_data['category'] = percentiles.apply(categorize_by_percentiles)

# Save categorized data
grouped_data.to_csv('../../results/cifar10/cifar_10_c_divergences_categories.csv')

# Category stats
category_bootstrap = grouped_data.groupby('category')['mean'].apply(lambda x: bootstrap_ci(x.values))
category_stats = pd.DataFrame(category_bootstrap.tolist(), index=category_bootstrap.index, columns=['mean', 'lower', 'upper'])

# Add CIFAR-10 reference point (0,0)
cifar_point = pd.DataFrame({'percentile': [0], 'mean': [0], 'lower': [0], 'upper': [0]})
combined_data = pd.concat([cifar_point, grouped_data.sort_values('percentile')], ignore_index=True)

# Seaborn style
plt.figure(figsize=(14, 10))

# CIFAR-10 reference point
plt.scatter(0, 0, color='blue', s=160, zorder=5)
plt.text(0, -0.15, 'CIFAR-10', ha='center', color='blue', fontsize=22, fontweight='bold')

# Connect CIFAR-10 to CIFAR-10-C with line
sns.lineplot(
    x='percentile', y='mean',
    data=combined_data,
    color='purple',
    label='CIFAR-10-C Trend',
    linewidth=2
)

# Red scatter points for corruptions
sns.scatterplot(
    x='percentile', y='mean',
    data=grouped_data,
    color='red',
    alpha=0.85,
    s=80
)

# Error bars for CIFAR-10-C corruptions
for _, row in grouped_data.iterrows():
    plt.errorbar(row['percentile'], row['mean'],
                 yerr=[[row['mean'] - row['lower']], [row['upper'] - row['mean']]],
                 fmt='none', ecolor='gray', capsize=5, alpha=0.6)

# Dynamic Y-axis and annotations
y_max = grouped_data['upper'].max()
y_offset = y_max * 0.004
for cat, pos in zip(['Lowest', 'Mid-Range', 'Highest'], [12.5, 50, 87.5]):
    count = (grouped_data['category'] == cat).sum()
    mean, lower, upper = category_stats.loc[cat]
    plt.text(pos, y_max + y_offset, f'{cat}', ha='center', fontsize=22, fontweight='bold')
    plt.text(pos, y_max + y_offset - 0.2, f'{count} corruptions\n{mean:.2f} ({lower:.2f}, {upper:.2f})', ha='center', fontsize=20)

# Vertical reference lines
plt.axvline(x=25, color='grey', linestyle='--', linewidth=1)
plt.axvline(x=75, color='grey', linestyle='--', linewidth=1)

# Axes and labels
plt.xlim(0, 101)
plt.ylim(0, 2.5)
plt.xlabel('Percentiles')
plt.ylabel('KL Divergence (CIFAR-10-C)')
plt.xticks([25, 75])
plt.title('KL Divergence Distribution Across Percentiles')

# Legend at bottom right
plt.legend(loc='lower right')
plt.tight_layout()

# Save and show
save_path = '../../results/cifar10/cifar_c_kldiv_characterization_plot.pdf'
plt.savefig(save_path)
plt.show()
print(f"Plot saved to: {save_path}")
