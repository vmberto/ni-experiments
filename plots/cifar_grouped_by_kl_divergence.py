import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lib.helpers import seaborn_styles, prepare_df, bootstrap_confidence_interval, markers

# Paths
RAW_RESULTS_PATH = '../results/cifar10/resnet20_wrn2810_cct.csv'
CATEGORIES_DF_PATH = '../results/cifar10/cifar_10_c_divergences_categories.csv'

# Constants
estimator = 'f1-score(weighted avg)'
noise_like = {"gaussian_noise", "shot_noise", "speckle_noise", "impulse_noise", "contrast", "brightness"}

seaborn_styles(sns)

# Load raw results and category metadata
results = pd.read_csv(RAW_RESULTS_PATH)
results = prepare_df(results, CATEGORIES_DF_PATH)

in_dist = results[results['evaluation_set'] == 'in_distribution']
corruptions = results[results['evaluation_set'] != 'in_distribution']
corruptions_wo_noise = corruptions[~corruptions['corruption_group'].isin(noise_like)]

# Utility to format CI output
def format_no_leading_zero(x):
    return "{:.2f}".format(x * 100)

def bootstrap_ci_format(series, n_bootstrap=1000, ci=95):
    values = series.dropna().values
    if len(values) == 0:
        return "–", 0.0

    lower, upper = bootstrap_confidence_interval(values, num_samples=n_bootstrap, ci=ci / 100)
    mean = np.mean(values)
    return f"{format_no_leading_zero(mean)} ({format_no_leading_zero(lower)}, {format_no_leading_zero(upper)})", mean

# Bootstrap summary
def summarize_with_ci(df_subset, severity_label):
    grouped = df_subset.groupby(['model', 'strategy'])[estimator]
    rows = []
    for (model, strategy), series in grouped:
        formatted, mean_val = bootstrap_ci_format(series)
        rows.append({
            'model': model,
            'strategy': strategy,
            'Severity': severity_label,
            estimator: mean_val,
            'ci': formatted
        })
    return pd.DataFrame(rows)

# Compute summaries
summary_all = summarize_with_ci(corruptions, 'All Corruptions')
summary_wo_noise = summarize_with_ci(corruptions_wo_noise, 'All Corruptions w/o Noise')

# Add summaries to full dataset
results = pd.concat([results, summary_all, summary_wo_noise], ignore_index=True)

# Plot function
def plot_results(df):
    df = df.copy()
    severity_order = [
        "In-Distribution", "All Corruptions", "All Corruptions w/o Noise", "Lowest", "Mid-Range", "Highest",

    ]
    strategy_order = [
        "Baseline",
        "RandAugment",
        "RandAugment+S&P",
        "RandAugment+Gaussian",
        "Curriculum Learning"
    ]

    df["Severity"] = pd.Categorical(df["Severity"], categories=severity_order, ordered=True)
    df["strategy"] = pd.Categorical(df["strategy"], categories=strategy_order, ordered=True)

    unique_models = df['model'].unique()
    num_models = len(unique_models)

    fig, axes = plt.subplots(1, num_models, figsize=(11 * num_models, 10))

    if num_models == 1:
        axes = [axes]

    for i, model_name in enumerate(unique_models):
        model_results = df[df['model'] == model_name]
        num_strategies = model_results['strategy'].nunique()

        ax = sns.pointplot(
            data=model_results,
            linestyles='none',
            x=estimator,
            y='Severity',
            hue='strategy',
            hue_order=strategy_order,
            order=severity_order,
            markers=markers,
            dodge=0.7 if num_strategies > 1 else False,
            err_kws={'linewidth': 3},
            errorbar=('ci', 95),
            ax=axes[i]
        )

        ax.set_xlabel("F1-Score", fontsize=42)
        ax.set_xlim(0.4, 0.95)

        for y in range(1, len(severity_order)):
            ax.axhline(y=y - 0.5, color='grey', linestyle='--', linewidth=1)

        if i == 0:
            ax.set_ylabel("Severity", fontsize=42)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        ax.set_title(model_name, fontsize=42)

    for ax in axes:
        ax.legend_.remove()

    handles, labels = ax.get_legend_handles_labels()

    for handle in handles:
        handle.set_markersize(30)

    fig.legend(
        handles, labels, title="Strategy", loc='lower center',
        bbox_to_anchor=(0.5, -0.42), fontsize=42, title_fontsize=42, ncol=3
    )
    fig.suptitle('Distributions Domain Range', fontsize=50, y=1)
    plt.tight_layout()
    plt.savefig('../output/cifar_results_by_kl_divergence.png', bbox_inches='tight')
    plt.show()

# Plot
plot_results(results)

# 📌 Compute summaries for each severity level
in_dist_summary = summarize_with_ci(in_dist, 'In-Distribution')
corruptions_all = summarize_with_ci(corruptions, 'All Corruptions')
corruptions_wo_noise_summary = summarize_with_ci(corruptions_wo_noise, 'All Corruptions w/o Noise')

lowest = results[results['Severity'] == 'Lowest']
midrange = results[results['Severity'] == 'Mid-Range']
highest = results[results['Severity'] == 'Highest']

lowest_summary = summarize_with_ci(lowest, 'Lowest')
midrange_summary = summarize_with_ci(midrange, 'Mid-Range')
highest_summary = summarize_with_ci(highest, 'Highest')

# Concatenate all summaries
all_summaries = pd.concat([
    in_dist_summary,
    corruptions_all,
    corruptions_wo_noise_summary,
    lowest_summary,
    midrange_summary,
    highest_summary
])

# Pivot the table: rows = (model, strategy), columns = Severity
summary_df = all_summaries.pivot(index=['model', 'strategy'], columns='Severity', values='ci').reset_index()

# Optional: reorder columns
desired_order = [
    'In-Distribution',
    'All Corruptions',
    'All Corruptions w/o Noise',
    'Lowest',
    'Mid-Range',
    'Highest'
]
summary_df = summary_df[['model', 'strategy'] + [col for col in desired_order if col in summary_df.columns]]

# Print the final table
print(summary_df.to_string(index=False))