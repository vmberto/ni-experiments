import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lib.helpers import seaborn_styles, bootstrap_confidence_interval
seaborn_styles(sns)

# Define the order of strategies
STRATEGIES_ORDER = {
    'Baseline': 0,
    'RandAugment': 1,
    'RandAugment+S&P': 2,
    'RandAugment+Gaussian': 3,
    'Curriculum Learning': 4,
    'Curriculum Learning V2': 5,
}

results = pd.concat([
    pd.read_csv('../results/cifar10/resnet20_wrn2810_cct.csv'),
], ignore_index=False)
characterization_df = pd.read_csv('../results/cifar10/cifar_10_c_divergences_categories.csv')

results['evaluation_set'] = results['evaluation_set'].str.replace(' ', '_').str.lower()
results['evaluation_set'] = results['evaluation_set'].str.replace('-', '_').str.lower()

characterization_df['corruption_type'] = characterization_df['corruption_type'].str.replace(' ', '_').str.lower()

# Merge results with characterization data
merged_results = results.merge(
    characterization_df[['corruption_type', 'category']],
    left_on='evaluation_set',
    right_on='corruption_type',
    how='left'
)

merged_results['Severity'] = np.where(
    merged_results['evaluation_set'] == 'in_distribution',
    'In-Distribution',
    merged_results['category']
)

merged_results.drop(columns=['corruption_type', 'category'], inplace=True)
results = merged_results

# Remove numeric suffixes from evaluation_set
for i in range(1, 6):
    results['evaluation_set'] = results['evaluation_set'].str.replace(f' {i}', '')

# Split into in-distribution and out-of-distribution results
results_ood = results[results['evaluation_set'] != 'in_distribution']
results_in = results[results['evaluation_set'] == 'in_distribution']
results_in.loc[:, 'Severity'] = 'In-Distribution'

# Replication splits for 10-fold cross-validation
replications = []
for index in range(10):
    if index < 9:
        a = np.arange(1, 11)
        a = np.delete(a, [index, index + 1])
        replications.append(a)


# Function to generate miscoverage data
def generate_miscoverage(df, severity=None):
    global_10fold_df = df.copy()
    each_replication_array = []
    for i, selected_folds in enumerate(replications):
        splitted_dataframe = global_10fold_df[global_10fold_df['fold'].isin(selected_folds)].copy()
        splitted_dataframe['replication'] = i + 1
        if severity:
            splitted_dataframe['Severity'] = severity
        each_replication_array.append(splitted_dataframe)

    each_replication_df = pd.concat(each_replication_array, ignore_index=True)

    each_leave1out_df = []

    for i, selected_folds in enumerate(replications):
        splitted_dataframe = each_replication_df[each_replication_df['fold'].isin(selected_folds)].copy()
        mean_fscore_overall = each_replication_df.groupby(['strategy', 'model', 'replication', 'Severity'])[
            'f1-score(weighted avg)'].mean()

        def normalize_fscore(row):
            fscore = row['f1-score(weighted avg)']
            mean = mean_fscore_overall[row['strategy'], row['model'], row['replication'], row['Severity']]
            return fscore - mean

        dataframe = pd.DataFrame({
            'model': splitted_dataframe['model'],
            'replication': splitted_dataframe['replication'],
            'fold': splitted_dataframe['fold'],
            'severity': splitted_dataframe['Severity'],
            'evaluation_set': splitted_dataframe['evaluation_set'],
            'strategy': splitted_dataframe['strategy'] + '_' + str(i),
            'f1-score(weighted avg)': splitted_dataframe.apply(normalize_fscore, axis=1),
        })
        each_leave1out_df.append(dataframe)

    result_dataframe = pd.concat(each_leave1out_df, ignore_index=True)
    result_dataframe['sort_key'] = (result_dataframe['strategy']
                                    .apply(lambda strategy: STRATEGIES_ORDER[strategy.split('_')[0]]))
    result_dataframe_final = result_dataframe.sort_values(by=['sort_key', 'strategy']).drop(columns=['sort_key'])

    return result_dataframe_final, each_leave1out_df


# Generate miscoverage data for in-distribution and out-of-distribution
result_dataframe_in, each_dataframe_fscore_in = generate_miscoverage(results_in)
result_dataframe_in['severity'] = 'In-Distribution'

result_dataframe_ood, each_dataframe_fscore_ood = generate_miscoverage(results_ood)

_, each_dataframe_fscore_all_corruptions = generate_miscoverage(results_ood, severity='All Corruptions')

result_dataframe_ood_wo_noise, each_dataframe_fscore_ood_wo_noise = generate_miscoverage(results_ood[
    ~results_ood['evaluation_set'].str.contains('gaussian|impulse|speckle|shot|contrast|brightness', case=False, regex=True)
].copy(), severity='All Corruptions w/o Noise')




# Function to dynamically plot results for all models
def plot_results_all_dynamic(df_in, df_out, df_out_wo_noise, x_label='Mean Centered F-Score', figsize=(50, 25)):
    models = sorted(df_in['model'].unique())
    num_models = len(models)

    severity_labels = [
        "In-Distribution",
        "All Corruptions",
        "All Corruptions w/o Noise",
        "Lowest",
        "Mid-Range",
        "Highest",
    ]
    num_plot_types = len(severity_labels)

    fig, axes = plt.subplots(
        num_models, num_plot_types, figsize=(figsize[0], figsize[1] * num_models / 3),
        squeeze=False
    )

    x_min, x_max = -0.03, 0.03
    x_ticks = [x_min, 0, x_max]

    all_strategies = sorted(df_in['strategy'].str.extract(r'^([^_]+)')[0].unique())
    palette = sns.color_palette("husl", len(all_strategies))
    palette_dict = {strategy: color for strategy, color in zip(all_strategies, palette)}

    for row_idx, model in enumerate(models):
        model_results_in = df_in[df_in['model'] == model]
        model_results_out = df_out[df_out['model'] == model]
        model_results_out_wo_noise = df_out_wo_noise[df_out_wo_noise['model'] == model]

        all_corruptions = model_results_out.copy()
        all_corruptions['severity'] = "All Corruptions"

        without_noise = model_results_out_wo_noise.copy()
        without_noise['severity'] = "All Corruptions w/o Noise"

        severity_data = {
            "In-Distribution": model_results_in,
            "All Corruptions": all_corruptions,
            "All Corruptions w/o Noise": without_noise,
            "Lowest": model_results_out[model_results_out['severity'] == 'Lowest'],
            "Mid-Range": model_results_out[model_results_out['severity'] == 'Mid-Range'],
            "Highest": model_results_out[model_results_out['severity'] == 'Highest'],
        }

        for col_idx, severity_label in enumerate(severity_labels):
            severity_df = severity_data.get(severity_label, pd.DataFrame())
            ax = axes[row_idx, col_idx]

            sns.pointplot(
                data=severity_df,
                x='f1-score(weighted avg)',
                y='severity',
                hue='strategy',
                linestyles='none',
                dodge=.9,
                errorbar=("ci", 95),
                palette=[palette_dict[strategy.split('_')[0]] for strategy in severity_df['strategy'].unique()],
                err_kws={'linewidth': 3},
                ax=ax
            )

            ax.set_xlim(x_min, x_max)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks, fontsize=28)
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.axvline(x=0, color='k', linestyle='--')

            if row_idx == 0:
                ax.set_title(severity_label, fontsize=42)
            if row_idx == num_models - 1:
                ax.set_xlabel(x_label, fontsize=42)
            if col_idx == 0:
                ax.set_ylabel(model, fontsize=42)

            ax.get_legend().remove()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    labels = [label.split('_')[0] for label in labels]
    unique_labels, unique_handles = [], []

    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)

    for handle in unique_handles:
        handle.set_markersize(15)

    fig.legend(
        unique_handles,
        unique_labels,
        title="Strategy",
        loc='lower center',
        bbox_to_anchor=(0.5, -0.155),
        fontsize=38,
        title_fontsize=42,
        ncol=len(unique_handles)
    )

    plt.tight_layout()
    plt.savefig('../output/dynamic_miscoverage_plot_dynamic_palette.png', bbox_inches='tight')
    plt.show()


plot_results_all_dynamic(result_dataframe_in, result_dataframe_ood, result_dataframe_ood_wo_noise)

df = pd.concat(each_dataframe_fscore_in + each_dataframe_fscore_ood + each_dataframe_fscore_ood_wo_noise + each_dataframe_fscore_all_corruptions)

results_list = []

for model in df['model'].unique():
    for strategy in STRATEGIES_ORDER.keys():  # Use STRATEGIES_ORDER to maintain desired sort
        curr_df = df[(df['model'] == model) & (df['strategy'].str.startswith(strategy))]

        if not curr_df.empty:  # Only process if there's data for this strategy
            for severity, severity_df in curr_df.groupby('severity'):
                # Calculate mean and standard deviation for the metric
                std_fscore = severity_df['f1-score(weighted avg)'].std()

                # Bootstrap confidence interval for the standard deviation
                # Make sure there's enough data for bootstrap, otherwise it might fail
                if len(severity_df['f1-score(weighted avg)']) > 1:
                    lower_std, upper_std = bootstrap_confidence_interval(severity_df['f1-score(weighted avg)'],
                                                                         metric=np.std)
                else:  # Cannot compute std or CI with only one data point
                    lower_std, upper_std = np.nan, np.nan

                results_list.append({
                    'Model': model,
                    'Strategy': strategy,
                    'Severity': severity,
                    'STD F1-Score': std_fscore,
                    'STD Lower CI': lower_std,
                    'STD Upper CI': upper_std
                })

# Convert the list of results to a DataFrame
results_df = pd.DataFrame(results_list)

# Format the 'STD (Lower, Upper)' column for display
results_df['STD (Lower, Upper)'] = results_df.apply(
    lambda row: f"{row['STD F1-Score']:.4f} ({row['STD Lower CI']:.4f}, {row['STD Upper CI']:.4f})"
    if not pd.isna(row['STD F1-Score']) else 'N/A', axis=1
)

# Select and reorder columns for the final display
final_df = results_df[['Model', 'Strategy', 'Severity', 'STD (Lower, Upper)']]

# Sort the DataFrame for better readability, if desired
final_df = final_df.sort_values(by=['Model', 'Strategy', 'Severity'])

# Print the markdown table
print(final_df.to_string(index=False))