import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


dataset = 'cifar10'
file_path = f'../results/{dataset}/{dataset}_{dataset}c_full_characterization.csv'

sns.set(font='serif')
sns.set_style("white", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"],
})
sns.set_theme(style="white")
rcParams['font.size'] = 22
rcParams['axes.labelsize'] = 28
rcParams['xtick.labelsize'] = 24
rcParams['ytick.labelsize'] = 24
rcParams['legend.fontsize'] = 18

# --- 2. Data Processing ---
data = pd.read_csv(file_path)

def bootstrap_ci(data, n_bootstrap=1000, ci_level=0.95):
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(boot_sample))
    return np.mean(boot_means), np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)

def categorize_by_percentiles(pct_value):
    if pct_value <= 33: return 'Lowest'
    elif pct_value <= 66: return 'Mid-Range'
    else: return 'Highest'

bootstrap_results = data.groupby('corruption_type')['wasserstein_per_feature'].apply(lambda x: bootstrap_ci(x.values))
grouped_data = pd.DataFrame(bootstrap_results.tolist(), index=bootstrap_results.index, columns=['mean', 'lower', 'upper'])
grouped_data['percentile'] = grouped_data['mean'].rank(pct=True) * 100
grouped_data['category'] = grouped_data['percentile'].apply(categorize_by_percentiles)

category_bootstrap = grouped_data.groupby('category')['mean'].apply(lambda x: bootstrap_ci(x.values))
category_stats = pd.DataFrame(category_bootstrap.tolist(), index=category_bootstrap.index, columns=['mean', 'lower', 'upper'])

cifar_point = pd.DataFrame({'percentile': [0], 'mean': [0], 'lower': [0], 'upper': [0]})
combined_data = pd.concat([cifar_point, grouped_data.sort_values('percentile')], ignore_index=True)

# --- 3. Plotting ---
fig, ax = plt.subplots(figsize=(14, 10))

# Main Trend Line
sns.lineplot(x='percentile', y='mean', data=combined_data, color='purple', linewidth=2.5, alpha=0.6, label=f'{dataset.upper()} Trend', zorder=3)

# Error Bars
for _, row in grouped_data.iterrows():
    ax.errorbar(row['percentile'], row['mean'], yerr=[[max(0, row['mean'] - row['lower'])], [row['upper'] - row['mean']]],
                 fmt='none', ecolor='black', capsize=3, alpha=0.15, zorder=2)

# Data Points
sns.scatterplot(x='percentile', y='mean', data=grouped_data, color='red', s=110, edgecolor='white', zorder=4)

# CIFAR Reference
ax.scatter(0, 0, color='blue', s=200, zorder=5)
ax.text(0, -0.05, f'{dataset.upper()}', ha='center', va='top', color='blue', fontweight='bold', fontsize=24, transform=ax.get_xaxis_transform())

# --- 4. FIXED ANNOTATIONS (Axes Coordinates) ---
# We use ax.get_xaxis_transform() for X (data-scale) and Y (axes-scale)
# This means Y=1.0 is the top of the plot area, Y=0.0 is the bottom.
for cat, pos in zip(['Lowest', 'Mid-Range', 'Highest'], [16.5, 49.5, 83]):
    if cat in category_stats.index:
        count = int((grouped_data['category'] == cat).sum())
        m, l, u = category_stats.loc[cat]

        # Header - Fixed at 92% height of the plot
        ax.text(pos, 0.92, f'{cat}', ha='center', va='bottom', fontsize=26, fontweight='bold',
                transform=ax.get_xaxis_transform())

        # Stats - Fixed at 90% height of the plot
        stats_text = f'{count} types\n{m:.3f} ({l:.3f}, {u:.3f})'
        ax.text(pos, 0.90, stats_text, ha='center', va='top', fontsize=20, linespacing=1.2,
                transform=ax.get_xaxis_transform())

# Vertical Separators
for x_val in [33, 66]:
    ax.axvline(x=x_val, color='black', linestyle='-', linewidth=1.2, alpha=0.3, zorder=1)

# --- 5. Final Polish ---
ax.set_xlim(0, 105)

# Use dynamic Y-limit with a fixed 40% headroom for text
y_limit = grouped_data['upper'].max() * 1.4
ax.set_ylim(0, y_limit)

ax.set_xlabel('Percentiles of Corruption Severity', labelpad=15)
ax.set_ylabel('Wasserstein Distance', labelpad=15)
ax.set_xticks([0, 33, 66, 100])

# Safe Legend
handles, labels = ax.get_legend_handles_labels()
trend_handle = [h for h, l in zip(handles, labels) if l == f'{dataset.upper()} Trend']
if trend_handle:
    ax.legend(handles=trend_handle, labels=[f'{dataset.upper()}-C Trend'], loc='lower right', frameon=True)

plt.tight_layout()
plt.show()
