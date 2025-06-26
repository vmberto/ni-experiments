import pandas as pd
import numpy as np
import re
from lib.helpers import bootstrap_confidence_interval

# Load dataset
# df = pd.read_csv('../output/experiment_2025-06-04_11-20-50/output.csv')
# df = pd.read_csv('../output/experiment_2025-06-04_15-56-24/output.csv')
df = pd.concat([
    pd.read_csv('../output/randaugment_simulation_v5/output.csv'),
    pd.read_csv('../output/experiment_2025-06-10_17-24-01/output.csv')
], ignore_index=True)

# Clean corruption names
def get_base_corruption(name):
    return re.sub(r'\s+\d+$', '', name.strip())

df['corruption_group'] = df['evaluation_set'].apply(get_base_corruption)

# CI formatters
def format_no_leading_zero(x):
    return "{:.2f}".format(x * 100)

def bootstrap_ci_format(series, n_bootstrap=1000, ci=95):
    values = series.dropna().values
    if len(values) == 0:
        return "–", 0
    means = [np.mean(np.random.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    mean = np.mean(values)
    return f"{format_no_leading_zero(mean)} ({format_no_leading_zero(lower)}, {format_no_leading_zero(upper)})", mean

# Summarize per evaluation set
def summarize_with_ci(df_subset):
    grouped = df_subset.groupby(['model',  'machine', 'strategy', 'evaluation_set'])['auc_roc']
    rows = []
    for (model, strategy, machine, evaluation_set), series in grouped:
        formatted, _ = bootstrap_ci_format(series)
        rows.append({
            'model': model,
            'strategy': strategy,
            'machine': machine,
            'evaluation_set': evaluation_set,
            'auc_roc (CI)': formatted,
        })
    return pd.DataFrame(rows)

# Create summary
summary_df = summarize_with_ci(df)

# Convert 'machine' column to a categorical dtype with our full_order
summary_df['machine'] = pd.Categorical(
    summary_df['machine'],
    ordered=True
)

# Now sort by model, strategy, machine (in categorical order), evaluation_set
summary_df = summary_df.sort_values(
    by=['model', 'strategy', 'machine', 'evaluation_set']
)

# Display
print(summary_df.to_string(index=False))
