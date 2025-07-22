import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

from lib.helpers import seaborn_styles, prepare_df, bootstrap_confidence_interval, markers

# Paths
results = pd.concat([
    pd.read_csv('../../results/agnews/output.csv'),
], ignore_index=True)
CATEGORIES_DF_PATH = '../../results/agnews/agnews_encoder_kldiv_categories.csv'

# Constants
estimator = 'f1-score(weighted avg)'

seaborn_styles(sns)

def normalize_strategy_name(name: str) -> str:
    # Remove trailing numbers (with optional space before)
    name = re.sub(r'\s*\d+$', '', name.strip())
    # Replace spaces with underscores and lowercase
    return name.lower().replace(' ', '_')

results['corruption_group'] = results['evaluation_set'].apply(normalize_strategy_name)

# Prepare dataframe
results = prepare_df(results, CATEGORIES_DF_PATH)

# Subsets
in_dist = results[results['evaluation_set'] == 'in_distribution']
corruptions = results[results['evaluation_set'] != 'in_distribution']

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

# Add summaries to full dataset
results = pd.concat([results, summary_all], ignore_index=True)

# ✅ Plot function
def plot_results(df):
    df = df.copy()
    severity_order = [
        "In-Distribution", "All Corruptions", "Lowest", "Mid-Range", "Highest"
    ]
    strategy_order = [
        "Baseline",
        "RandAugment",
        "RandAugment+S&P",
        "RandAugment+Gaussian",
        "Curriculum Learning",
        "Salt&Pepper",
        "Gaussian",
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
            x=estimator,
            y='Severity',
            hue='strategy',
            hue_order=strategy_order,
            order=severity_order,
            markers=markers,
            dodge=0.7 if num_strategies > 1 else False,
            linewidth=5,
            errorbar=('ci', 95),
            ax=axes[i]
        )

        ax.set_xlabel("F1-Score", fontsize=42)
        ax.set_xlim(0.4, .95)

        for y in range(1, len(severity_order)):
            ax.axhline(y=y - 0.5, color='grey', linestyle='-', linewidth=1)

        if i != 0:
            ax.set_yticklabels([])

        ax.set_ylabel("")
        ax.tick_params(axis='x', labelsize=28)
        ax.tick_params(axis='y', labelsize=32)
        ax.set_title(model_name, fontsize=42)

    for ax in axes:
        ax.legend_.remove()

    handles, labels = ax.get_legend_handles_labels()

    for handle in handles:
        handle.set_markersize(30)

    fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.32), fontsize=42, title_fontsize=42, ncol=3
    )
    plt.tight_layout()
    plt.savefig('../../results/agnews/agnews_results_by_domain.png', bbox_inches='tight')
    plt.savefig('../../results/agnews/agnews_results_by_domain.pdf', bbox_inches='tight')
    plt.show()

# ✅ Plot
plot_results(results)

# ✅ Compute summaries for all severity levels
in_dist_summary = summarize_with_ci(in_dist, 'In-Distribution')
corruptions_all = summarize_with_ci(corruptions, 'All Corruptions')

lowest = results[results['Severity'] == 'Lowest']
midrange = results[results['Severity'] == 'Mid-Range']
highest = results[results['Severity'] == 'Highest']

lowest_summary = summarize_with_ci(lowest, 'Lowest')
midrange_summary = summarize_with_ci(midrange, 'Mid-Range')
highest_summary = summarize_with_ci(highest, 'Highest')

# ✅ Concatenate all summaries
all_summaries = pd.concat([
    in_dist_summary,
    corruptions_all,
    lowest_summary,
    midrange_summary,
    highest_summary
])

# ✅ Pivot the table: rows = (model, strategy), columns = Severity
summary_df = all_summaries.pivot(index=['model', 'strategy'], columns='Severity', values='ci').reset_index()

# ✅ Compute average epochs_run per (model, strategy)
epochs_summary = results.groupby(['model', 'strategy'])['epochs_run'].mean().reset_index()
epochs_summary['epochs_run'] = epochs_summary['epochs_run'].round(2)

# ✅ Merge into summary_df
summary_df = summary_df.merge(epochs_summary, on=['model', 'strategy'], how='left')

# ✅ Reorder columns
desired_order = [
    'In-Distribution', 'All Corruptions', 'Lowest', 'Mid-Range', 'Highest'
]
summary_df = summary_df[['model', 'strategy', 'epochs_run'] + [col for col in desired_order if col in summary_df.columns]]

# ✅ Print the final table
print(summary_df.to_string(index=False))
