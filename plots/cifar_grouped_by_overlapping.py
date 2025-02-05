import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from lib.helpers import seaborn_styles, bootstrap_confidence_interval, markers

seaborn_styles(sns)

RAW_RESULTS_PATH = '../output/complete_resnet50_resnet18.csv'

overlap_keywords = [
    "Gaussian Noise", "Shot Noise", "Impulse Noise",
    "Speckle Noise", "Brightness", "Contrast",
]


def is_overlap_corruption(corr_name: str) -> bool:
    return any(kw in corr_name for kw in overlap_keywords)


results = pd.read_csv(RAW_RESULTS_PATH)


def categorize_severity(row):
    if row['evaluation_set'] == 'In-Distribution':
        return 'In-Distribution'
    else:
        is_overlap = is_overlap_corruption(row['evaluation_set'])
        if is_overlap:
            return 'All Corruptions'
        else:
            return row['category'] if 'category' in row else 'All Corruptions'


results['Severity'] = results.apply(categorize_severity, axis=1)

without_overlap = results[
    (results['Severity'] == 'All Corruptions') &
    ~results['evaluation_set'].apply(is_overlap_corruption)
    ].copy()
without_overlap['Severity'] = 'Without Overlap'

filtered_results = pd.concat([results, without_overlap])

severity_order = ['In-Distribution', 'All Corruptions', 'Without Overlap']
filtered_results['Severity'] = pd.Categorical(filtered_results['Severity'],
                                              categories=severity_order,
                                              ordered=True)


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
            order=severity_order,
            ax=axes[i]
        )

        # Customize plot appearance
        ax.set_xlabel("F1-Score", fontsize=42)
        ax.set_xlim(.4, .95)

        # Add horizontal separators
        for y in [0.5, 1.5, 2.5]:
            ax.axhline(y=y, color='grey', linestyle='--', linewidth=1)

        # Set labels and ticks
        if i == 0:
            ax.set_ylabel("Severity", fontsize=42)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        ax.set_title(model_name, fontsize=42)

        # Remove individual legends
        ax.legend_.remove()

    # Create a single legend at the bottom
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_markersize(30)

    fig.legend(
        handles, labels,
        title="Strategy",
        loc='lower center',
        bbox_to_anchor=(0.5, -0.42),
        fontsize=42,
        title_fontsize=42,
        ncol=3
    )

    fig.suptitle('Distributions Domain Range', fontsize=50, y=1)
    plt.tight_layout()

    plt.savefig('../output/cifar_results_by_overlapping.pdf', bbox_inches='tight')
    plt.show()


plot_results(filtered_results)

grouped = filtered_results.groupby(['model', 'strategy', 'Severity'])
confidence_intervals = grouped['f1-score(weighted avg)'].apply(
    lambda x: bootstrap_confidence_interval(x.values)
)
average_fscore = grouped['f1-score(weighted avg)'].mean()

# Create summary DataFrame
confidence_intervals_df = pd.DataFrame({
    'Model': confidence_intervals.index.get_level_values('model'),
    'Severity': confidence_intervals.index.get_level_values('Severity'),
    'Strategy': confidence_intervals.index.get_level_values('strategy'),
    'F-Score 95%': [
        f"{avg:.4f} ({lower:.4f}, {upper:.4f})"
        for (avg, (lower, upper)) in zip(average_fscore, confidence_intervals)
    ]
})

print("\n=== Summary Statistics ===")
print(confidence_intervals_df.to_latex())

print("\n=== Evaluation Sets by Category ===")
for severity in severity_order:
    print(f"\n{severity}:")
    print(filtered_results[filtered_results['Severity'] == severity]['evaluation_set'].unique())