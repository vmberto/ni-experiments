import numpy as np
from scipy import stats
import pandas as pd


def bootstrap_confidence_interval(data, num_samples=1000, ci=0.95):
    data = np.array(data)

    res = stats.bootstrap((data,), np.mean, confidence_level=ci, n_resamples=num_samples, method='basic')
    return res.confidence_interval.low, res.confidence_interval.high


def seaborn_styles(sns):
    sns.set_style("whitegrid")
    sns.set(font='serif')
    sns.set_style("white", {
      "font.family": "serif",
      "font.serif": ["Times", "Palatino", "serif"],
    })


def prepare_df(df, categories_df_path):
    df['evaluation_set'] = df['evaluation_set'].str.replace(' ', '_').str.lower()
    df['evaluation_set'] = df['evaluation_set'].str.replace('-', '_').str.lower()
    characterization_df = pd.read_csv(categories_df_path)
    characterization_df['corruption_type'] = characterization_df['corruption_type'].str.replace(' ', '_').str.lower()
    merged_results = df.merge(
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
    merged_results.drop(columns=['corruption_type'], inplace=True)
    merged_results.drop(columns=['category'], inplace=True)

    return merged_results



markers = ['o', 's', '^', 'v', '<', '>', 'p']
