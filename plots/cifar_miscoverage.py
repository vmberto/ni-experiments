import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lib.helpers import seaborn_styles, bootstrap_confidence_interval

# Define the order of strategies
STRATEGIES_ORDER = {
    'Baseline': 0,
    'Salt&Pepper': 1,
    'Gaussian': 2,
    'RandAugment': 3,
    'RandAugment+Gaussian': 4,
    'RandAugment+S&P': 5,
    'Mixed': 6,
}

overlap_keywords = [
    "Gaussian Noise", "Shot Noise", "Impulse Noise",
    "Speckle Noise", "Brightness", "Contrast",
]

# Load results and characterization files
# results = pd.read_csv(f'../results/agnews/raw_lstm_all_strategies_result.csv')
results = pd.read_csv('../output/complete_resnet50_resnet18.csv')
characterization_df = pd.read_csv('../results/cifar10/cifar_10_c_divergences_categories.csv')

seaborn_styles(sns)

# Preprocess evaluation sets in results
results['evaluation_set'] = results['evaluation_set'].str.replace(' ', '_').str.lower()
results['evaluation_set'] = results['evaluation_set'].str.replace('-', '_').str.lower()

# Preprocess corruption types in characterization_df
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
def generate_miscoverage(df):
    global_10fold_df = df.copy()
    each_replication_array = []
    for i, selected_folds in enumerate(replications):
        splitted_dataframe = global_10fold_df[global_10fold_df['fold'].isin(selected_folds)].copy()
        splitted_dataframe['replication'] = i + 1
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


# Function to dynamically plot results for all models
def plot_results_all_dynamic(df_in, df_out, x_label='Mean Centered F-Score', figsize=(50, 25)):
    models = sorted(df_in['model'].unique())  # Dynamically detect models
    num_models = len(models)
    num_plot_types = 6  # In-Distribution, All Corruptions, etc.

    fig, axes = plt.subplots(
        num_models, num_plot_types, figsize=(figsize[0], figsize[1] * num_models / 3),
        squeeze=False  # Keep 2D structure even with one row
    )

    x_min, x_max = -0.03, 0.03
    x_ticks = [x_min, 0, x_max]

    # Define color palette for strategies
    unique_approaches = df_in['strategy'].unique()
    palette_dict = {}
    for strategy in unique_approaches:
        if strategy not in palette_dict:
            if strategy.split('_')[0] == 'Baseline':
                palette_dict['Baseline'] = '#5471ab'
            elif strategy.split('_')[0] == 'Gaussian':
                palette_dict['Gaussian'] = '#6aa66e'
            elif strategy.split('_')[0] == 'Salt&Pepper':
                palette_dict['Salt&Pepper'] = '#d1885c'
            elif strategy.split('_')[0] == 'RandAugment+S&P':
                palette_dict['RandAugment+S&P'] = '#7f73af'
            elif strategy.split('_')[0] == 'RandAugment+Gaussian':
                palette_dict['RandAugment+Gaussian'] = '#8f7963'
            elif strategy.split('_')[0] == 'RandAugment':
                palette_dict['RandAugment'] = '#b65655'
            elif strategy.split('_')[0] == 'Mixed':
                palette_dict['Mixed'] = '#D48AC7'
    palette = [palette_dict[strategy.split('_')[0]] for strategy in unique_approaches]

    for row_idx, model in enumerate(models):
        model_results_in = df_in[df_in['model'] == model]
        model_results_out = df_out[df_out['model'] == model]

        all_corruptions = model_results_out.copy()
        all_corruptions['severity'] = "All Corruptions"

        no_overlap = model_results_out[
            ~model_results_out['evaluation_set'].str.contains('|'.join(overlap_keywords), case=False, na=False)
        ].copy()
        no_overlap['severity'] = "All Corruptions"

        severities = {
            "In-Distribution": model_results_in,
            "All Corruptions": all_corruptions,
            "w/o Overlap": no_overlap,
            "Lowest": model_results_out[model_results_out['severity'] == 'Lowest'],
            "Mid-Range": model_results_out[model_results_out['severity'] == 'Mid-Range'],
            "Highest": model_results_out[model_results_out['severity'] == 'Highest'],
        }

        # Plot each type of result for the current model
        for col_idx, (severity_label, severity_df) in enumerate(severities.items()):
            ax = axes[row_idx, col_idx]

            sns.pointplot(
                data=severity_df,
                x='f1-score(weighted avg)',
                y='severity',
                hue='strategy',
                linestyles='none',
                dodge=.9,
                errorbar=("ci", 95),
                palette=palette,
                err_kws={'linewidth': 3},
                ax=ax
            )

            ax.set_xlim(x_min, x_max)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks, fontsize=28)
            ax.set_yticks([])
            ax.axvline(x=0, color='k', linestyle='--')

            # Add title only for the first row
            if row_idx == 0:
                ax.set_title(severity_label, fontsize=42)
            else:
                ax.set_title("")

            if row_idx == num_models - 1:
                ax.set_xlabel(x_label, fontsize=42)
            else:
                ax.set_xlabel("")

            # Add y-label only for the first column of plots
            if col_idx == 0:
                ax.set_ylabel(model, fontsize=42)
            else:
                ax.set_ylabel("")

            ax.get_legend().remove()

    # Generate a single legend for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    labels = [label.split('_')[0] for label in labels]
    unique_labels = []
    unique_handles = []
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
    plt.savefig('../output/dynamic_miscoverage_plot.pdf', bbox_inches='tight')
    plt.show()


plot_results_all_dynamic(result_dataframe_in, result_dataframe_ood)

df = pd.concat(each_dataframe_fscore_in + each_dataframe_fscore_ood)

for model in df['model'].unique():
    print('\n')
    for strategy in STRATEGIES_ORDER.keys():
        curr_df = df[(df['model'] == model) & (df['strategy'].str.startswith(strategy))]

        for severity, severity_df in curr_df.groupby('severity'):
            lower, upper = bootstrap_confidence_interval(severity_df['f1-score(weighted avg)'], metric=np.std)
            print(
                f"STD {model} - {strategy} - Severity {severity}: {severity_df['f1-score(weighted avg)'].std()} ({lower}, {upper})"
            )