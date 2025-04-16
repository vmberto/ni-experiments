import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Load the CSV file with KL divergences
df = pd.read_csv('../output/autoencoder_results_kldiv.csv')

# Set font and sizes
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 24
rcParams['axes.titlesize'] = 32
rcParams['axes.labelsize'] = 28
rcParams['xtick.labelsize'] = 26
rcParams['ytick.labelsize'] = 26
rcParams['legend.fontsize'] = 18

# Utility: bootstrap confidence interval
def bootstrap_ci(data, n_bootstrap=1000, ci_level=0.95):
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(boot_sample))
    lower = np.percentile(boot_means, (1 - ci_level) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci_level) / 2 * 100)
    return np.mean(boot_means), lower, upper

# Step 1: Compute adjusted KL = corruption - augmented (per fold)
augmented = df[df['corruption_type'] == 'augmented'].groupby('fold')['kl_divergences'].mean()
corruptions = df[df['corruption_type'] != 'augmented']

adjusted_klds = []
for fold in corruptions['fold'].unique():
    aug_kl = augmented.loc[fold]
    fold_data = corruptions[corruptions['fold'] == fold].copy()
    fold_data['adjusted_kl'] = fold_data['kl_divergences'] - aug_kl
    adjusted_klds.append(fold_data)

adjusted_df = pd.concat(adjusted_klds)

# Step 2: Bootstrap stats for each corruption type (relative to augmented)
bootstrap_results = adjusted_df.groupby('corruption_type')['adjusted_kl'].apply(
    lambda x: bootstrap_ci(x.values))

grouped_data = pd.DataFrame(bootstrap_results.tolist(),
                            index=bootstrap_results.index,
                            columns=['mean', 'lower', 'upper']).sort_values(by='mean').reset_index()

# Step 3: Compute percentiles and categories
percentiles = grouped_data['mean'].rank(pct=True) * 100

def categorize_by_percentiles(pct_value):
    if pct_value <= 25:
        return 'Lowest'
    elif 25 < pct_value <= 75:
        return 'Mid-Range'
    else:
        return 'Highest'

grouped_data['category'] = percentiles.apply(categorize_by_percentiles)

# Count corruption types by category
low_count = (percentiles <= 25).sum()
mid_count = ((percentiles > 25) & (percentiles <= 75)).sum()
high_count = (percentiles > 75).sum()

# Bootstrap CIs per category
category_stats = grouped_data.groupby('category')['mean'].apply(
    lambda x: bootstrap_ci(x.values)).apply(pd.Series)
category_stats.columns = ['mean', 'lower', 'upper']

# Step 4: Augmented KL mean and CI (vs. clean)
augmented_kl_values = df[df['corruption_type'] == 'augmented']['kl_divergences'].values
augmented_mean, augmented_lower, augmented_upper = bootstrap_ci(augmented_kl_values)

# Step 5: Plot
plt.figure(figsize=(14, 10))

# Plot Augmented point with CI
plt.errorbar(0, augmented_mean,
             yerr=[[augmented_mean - augmented_lower], [augmented_upper - augmented_mean]],
             fmt='o', color='green', ecolor='darkgreen', elinewidth=2, capsize=6, label='Augmented CIFAR-10')

# Sort for consistent plotting
sorted_percentiles = percentiles.sort_values()
sorted_means = grouped_data.loc[sorted_percentiles.index, 'mean']
sorted_lower = grouped_data.loc[sorted_percentiles.index, 'lower']
sorted_upper = grouped_data.loc[sorted_percentiles.index, 'upper']

colors = plt.cm.Reds(np.linspace(0.5, 1, len(sorted_percentiles)))

# Plot corruption points
for i, (x, y, lower, upper) in enumerate(zip(sorted_percentiles, sorted_means, sorted_lower, sorted_upper)):
    plt.errorbar(x, y,
                 yerr=[[y - lower], [upper - y]],
                 fmt='o', color=colors[i], ecolor='gray', capsize=5, alpha=0.7)

# Connect points with a trend line
plt.plot(sorted_percentiles, sorted_means, linestyle='-', color='purple', label='CIFAR-10-C Trend')

# Annotate categories
text_y = grouped_data['upper'].max() + 2
for category, pos in zip(['Lowest', 'Mid-Range', 'Highest'], [11, 50, 89]):
    count = {'Lowest': low_count, 'Mid-Range': mid_count, 'Highest': high_count}[category]
    mean = category_stats.loc[category, 'mean']
    lower = category_stats.loc[category, 'lower']
    upper = category_stats.loc[category, 'upper']
    plt.text(pos, text_y, f'{category}', ha='center', fontsize=22, fontweight='bold')
    plt.text(pos, text_y - 1.2, f'{count} corruptions\n{mean:.2f} ({lower:.2f}, {upper:.2f})',
             ha='center', fontsize=22)

# Vertical separators
plt.axvline(x=25, color='grey', linestyle='--', linewidth=1)
plt.axvline(x=75, color='grey', linestyle='--', linewidth=1)

# Axes settings
plt.ylim(0, grouped_data['upper'].max() + 3)
plt.xlabel('Percentiles')
plt.ylabel('KL Divergence (Relative to Augmented)')
plt.xticks([25, 75])
plt.title('KL Divergence to Corruptions (Relative to Augmented CIFAR-10)')
plt.legend(loc='lower right')
plt.tight_layout()

plt.show()
