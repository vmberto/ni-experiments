import pandas as pd
import numpy as np
from lib.helpers import bootstrap_confidence_interval
import re

# estimator = 'f1-score(weighted avg)'
estimator = 'auc_roc'

# df = pd.concat([
#     pd.read_csv('../output/output.csv'),
#     pd.read_csv('../output/output_merged.csv')
# ], ignore_index=False)
df = pd.read_csv('../output/output_mimii.csv')
# df = pd.read_csv('../output/output.csv')

# Extrai raiz da corrupção
def get_base_corruption(name):
    return re.sub(r'\s+\d+$', '', name.strip())

df['corruption_group'] = df['evaluation_set'].apply(get_base_corruption)

# Define corrupções do tipo ruído
noise_like = {"Gaussian Noise", "Shot Noise", "Speckle Noise", "Impulse Noise"}
# noise_like = {"Gaussian Noise", "Shot Noise", "Speckle Noise", "Impulse Noise", "Contrast"}

# Função de bootstrap com formatação
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

# Resumo com média
def summarize_with_ci(df_subset, column_name):
    grouped = df_subset.groupby(['model', 'strategy'])[estimator]
    rows = []
    for (model, strategy), series in grouped:
        formatted, mean_val = bootstrap_ci_format(series)
        rows.append({'model': model, 'strategy': strategy, column_name: formatted, f'{column_name}_mean': mean_val})
    return pd.DataFrame(rows)

# 📌 F1-scores com intervalos
in_dist = df[df['evaluation_set'] == 'In-Distribution']
corruptions = df[df['evaluation_set'] != 'In-Distribution']
corruptions_wo_noise = corruptions[~corruptions['corruption_group'].isin(noise_like)]

in_dist_summary = summarize_with_ci(in_dist, 'In-Distribution')
corruptions_all = summarize_with_ci(corruptions, 'All Corruptions')
corruptions_wo_noise_summary = summarize_with_ci(corruptions_wo_noise, 'All Corruptions w/o Noise')

# 📌 Tempo médio de treinamento e épocas por modelo+estratégia
agg_time = df[df['evaluation_set'] == 'In-Distribution'].groupby(['model', 'strategy']).agg({
    'training_time': 'mean',
    'epochs_run': 'mean'
}).reset_index()

# Junta tudo
summary_df = in_dist_summary \
    .merge(corruptions_all, on=['model', 'strategy']) \
    .merge(corruptions_wo_noise_summary, on=['model', 'strategy']) \
    .merge(agg_time, on=['model', 'strategy'])

# Remove colunas auxiliares
summary_df = summary_df.drop(columns=[c for c in summary_df.columns if c.endswith('_mean')])

# Mostra resultado final
print(summary_df.to_string(index=False))

