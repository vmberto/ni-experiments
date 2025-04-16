import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font family and increase font sizes
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 24  # General font size
rcParams['axes.titlesize'] = 32  # Title font size
rcParams['axes.labelsize'] = 28  # X and Y label font size
rcParams['xtick.labelsize'] = 26  # X tick font size
rcParams['ytick.labelsize'] = 26  # Y tick font size
rcParams['legend.fontsize'] = 18  # Legend font size

file_path = '../output/autoencoder_results_kldiv_train_test_w_augmentation.csv'  # Replace with your actual file path
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


bootstrap_results = data.groupby('corruption_type')['kl_divergences'].apply(
    lambda x: bootstrap_ci(x.values))

grouped_data = pd.DataFrame(bootstrap_results.tolist(), index=bootstrap_results.index, columns=['mean', 'lower', 'upper'])
percentiles = grouped_data['mean'].rank(pct=True) * 100

grouped_data['category'] = percentiles.apply(categorize_by_percentiles)

grouped_data.to_csv('../results/cifar10/cifar_10_c_divergences_categories.csv')

low_count = (percentiles <= 25).sum()
mid_range_count = ((percentiles > 25) & (percentiles <= 75)).sum()
high_count = (percentiles > 75).sum()

# Calculate bootstrap CI for each category
category_bootstrap = grouped_data.groupby('category')['mean'].apply(lambda x: bootstrap_ci(x.values))
category_stats = pd.DataFrame(category_bootstrap.tolist(), index=category_bootstrap.index, columns=['mean', 'lower', 'upper'])

plt.figure(figsize=(14, 10))

# Plot CIFAR-10 reference point at (0%, 0)
plt.scatter(0, -0.8, color='blue', s=120, label='CIFAR-10')  # Increase point size
plt.text(0, -0.8, 'CIFAR-10', ha='center', color='blue', fontsize=22, fontweight='bold')  # Increase text size

# Sort percentiles and mean values to plot a single, smooth line
sorted_percentiles = percentiles.sort_values()
sorted_means = grouped_data.loc[sorted_percentiles.index, 'mean']
sorted_lower = grouped_data.loc[sorted_percentiles.index, 'lower']
sorted_upper = grouped_data.loc[sorted_percentiles.index, 'upper']

# Create a color gradient from light red to strong red
colors = plt.cm.Reds(np.linspace(0.5, 1, len(sorted_percentiles)))  # Adjust the start to 0.3 for a lighter red start

# Plot KL Divergence with error bars and gradient colors
for i, (x, y, lower, upper) in enumerate(zip(sorted_percentiles, sorted_means, sorted_lower, sorted_upper)):
    plt.errorbar(x, y, yerr=[[y - lower], [upper - y]], fmt='o', color=colors[i], ecolor='gray', capsize=5, alpha=0.7)

# Draw a single smooth line through CIFAR-10-C points without the gradient
plt.plot(sorted_percentiles, sorted_means, linestyle='-', color='purple', label='CIFAR-10-C Trend')

# Add range labels and corruption counts with confidence intervals
text_y_position = grouped_data['upper'].max() + 3

for category, pos in zip(['Lowest', 'Mid-Range', 'Highest'], [12.5, 50, 87.5]):
    count = {'Lowest': low_count, 'Mid-Range': mid_range_count, 'Highest': high_count}[category]
    mean = category_stats.loc[category, 'mean']
    lower = category_stats.loc[category, 'lower']
    upper = category_stats.loc[category, 'upper']
    plt.text(pos, text_y_position,
             f'{category}', ha='center', fontsize=22, fontweight='bold')  # Increase text size for categories
    plt.text(pos, text_y_position - 1.25,  # Adjust position slightly below
             f'{count} corruptions\n{mean:.2f} ({lower:.2f}, {upper:.2f})',
             ha='center', fontsize=22)  # Increase text size for details

# Vertical lines for percentile categories
plt.axvline(x=25, color='grey', linestyle='--', linewidth=1)
plt.axvline(x=75, color='grey', linestyle='--', linewidth=1)

# Set x-axis and y-axis limits and labels
plt.xlim(0, 101)  # Slightly negative to accommodate CIFAR-10 point
plt.ylim(0, 20)
plt.xlabel('Percentiles')
plt.ylabel('KL Divergence (CIFAR-10-C)')
plt.xticks(ticks=[25, 75])
# plt.yticks(ticks=[0.5, 1.0, 1.5, 2.0])
plt.title('KL Divergence Distribution Across Percentiles')

# Add a legend
plt.legend(loc='lower right')

# Save plot and show
plt.tight_layout()
plt.savefig('../results/cifar10/cifar_10_c_divergences_plot.pdf')
plt.show()
