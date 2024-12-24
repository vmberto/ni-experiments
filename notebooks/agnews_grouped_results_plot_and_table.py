import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from helpers import seaborn_styles, markers, bootstrap_confidence_interval, prepare_df

seaborn_styles(sns)

# RAW_RESULTS_PATH = '../output/output.csv'
RAW_RESULTS_PATH = '../results/agnews/raw_lstm_all_strategies_result.csv'
CATEGORIES_DF_PATH = '../results/agnews/agnews_c_divergences_categories.csv'


def plot_results(df):
    unique_models = df['model'].unique()
    num_models = len(unique_models)

    fig, axes = plt.subplots(1, num_models, figsize=(11 * num_models, 10))

    if num_models == 1:
        axes = [axes]

    for i, model_name in enumerate(unique_models):
        model_results = df[df['model'] == model_name]
        ax = sns.pointplot(
            data=model_results,
            linestyles='none',
            x='f1-score(weighted avg)',
            y='Severity',
            hue='strategy',
            markers=markers,
            dodge=0.7,
            err_kws={'linewidth': 3},
            errorbar=('ci', 95),
            ax=axes[i]
        )

        ax.set_xlabel("F1-Score", fontsize=42)
        ax.set_xlim(.5, 1)
        
        ax.axhline(y=0.5, color='grey', linestyle='--', linewidth=1)
        ax.axhline(y=1.5, color='grey', linestyle='--', linewidth=1)
        ax.axhline(y=2.5, color='grey', linestyle='--', linewidth=1)

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
        handle.set_markersize(30)  # Increase the marker size
    
    fig.legend(
        handles, labels, title="Strategy", loc='lower center',
        bbox_to_anchor=(0.5, -0.42), fontsize=42, title_fontsize=42, ncol=3
    )
    fig.suptitle('Distributions Domain Range', fontsize=50, y=1)
    plt.tight_layout()
    plt.savefig('../output/agnews_results_ci_each_domain.pdf', bbox_inches='tight')
    plt.show()


results = pd.read_csv(RAW_RESULTS_PATH)
merged_results = prepare_df(results, CATEGORIES_DF_PATH)

plot_results(merged_results)

grouped = merged_results.groupby(['model', 'strategy', 'Severity'])
confidence_intervals = grouped['f1-score(weighted avg)'].apply(lambda x: bootstrap_confidence_interval(x.values))
average_fscore = grouped['f1-score(weighted avg)'].mean()

confidence_intervals_df = pd.DataFrame({
    'Model': confidence_intervals.index.get_level_values('model'),
    'strategy': confidence_intervals.index.get_level_values('strategy'),
    'Severity': confidence_intervals.index.get_level_values('Severity'),
    'F-Score 95%': [f"{avg:.4f} ({lower:.4f}, {upper:.4f})" for (avg, (lower, upper)) in zip(average_fscore, confidence_intervals)]
})

print(confidence_intervals_df.to_latex())