import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from helpers import seaborn_styles, prepare_df, bootstrap_confidence_interval, markers

seaborn_styles(sns)

RAW_RESULTS_PATH = '../results/cifar10/official_10folds_results_resnet50_xception.csv'
CATEGORIES_DF_PATH = '../results/cifar10/cifar_10_c_divergences_categories.csv'

def plot_results(df):
    unique_values = df['model'].unique()

    resnet_results = df[df['model'] == unique_values[0]]
    xception_results = df[df['model'] == unique_values[1]]

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    x_min = 0.3
    x_max = 0.9

    for i, plot in enumerate([
        {"df": resnet_results, "model": "ResNet50"},
        {"df": xception_results, "model": "Xception"}
    ]):
        ax = sns.pointplot(
            data=plot['df'],
            x='f1-score(weighted avg)',
            y='Severity',
            hue='strategy',
            linestyles='none',
            markers=markers,
            dodge=0.7,
            err_kws={'linewidth': 3},
            errorbar=('ci', 95),
            ax=axes[i]
        )

        ax.set_xlabel("F1-Score", fontsize=42)
        ax.set_xlim(x_min, x_max)

        if i == 0:
            ax.set_ylabel("Severity", fontsize=42)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        ax.axhline(y=0.5, color='grey', linestyle='--', linewidth=1)
        ax.axhline(y=1.5, color='grey', linestyle='--', linewidth=1)
        ax.axhline(y=2.5, color='grey', linestyle='--', linewidth=1)
        ax.tick_params(axis='x', labelsize=32)
        ax.tick_params(axis='y', labelsize=24)
        ax.set_title(plot['model'], fontsize=42)

        x_ticks = ax.get_xticks()
        ax.set_xticks(x_ticks[1:-1])

    axes[0].legend_.remove()
    axes[1].legend_.remove()

    handles, labels = ax.get_legend_handles_labels()

    for handle in handles:
        handle.set_markersize(30)

    fig.legend(handles, labels, title="Strategy", loc='lower center', bbox_to_anchor=(0.5, -0.42), fontsize=42, title_fontsize=42, ncol=3)
    fig.suptitle('Distributions Domain Range', fontsize=50, y=1)
    plt.tight_layout()
    plt.savefig('../output/results_ci_each_domain.pdf', bbox_inches='tight')
    plt.show()


results = pd.read_csv(RAW_RESULTS_PATH)
results = prepare_df(results, CATEGORIES_DF_PATH)

plot_results(results)

grouped = results.groupby(['model', 'strategy', 'Severity'])
confidence_intervals = grouped['f1-score(weighted avg)'].apply(lambda x: bootstrap_confidence_interval(x.values))
average_fscore = grouped['f1-score(weighted avg)'].mean()

confidence_intervals_df = pd.DataFrame({
    'Model': confidence_intervals.index.get_level_values('model'),
    'strategy': confidence_intervals.index.get_level_values('strategy'),
    'Severity': confidence_intervals.index.get_level_values('Severity'),
    'F-Score 95%': [f"{avg:.4f} ({lower:.4f}, {upper:.4f})" for (avg, (lower, upper)) in zip(average_fscore, confidence_intervals)]
})




