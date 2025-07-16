import torch
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from lib.consts import AGNEWS_CORRUPTIONS
from lib.helpers import seaborn_styles
from dataset.agnewsdataset import AGNewsDataset
from scipy.stats import entropy
import numpy as np


# ----------------------------
# CONFIG
# ----------------------------
MAX_SEQUENCE_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# DISTILBERT ENCODER
# ----------------------------
def get_distilbert_encoder():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE)
    model.eval()

    def encode_texts(text_batch):
        if not isinstance(text_batch, list):
            text_batch = text_batch.tolist()

        clean_batch = [str(t) for t in text_batch if isinstance(t, str) and t.strip() != ""]
        if not clean_batch:
            raise ValueError("All texts are empty after cleaning. Check your corruption method.")

        inputs = tokenizer(clean_batch, padding=True, truncation=True,
                           max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings.cpu().numpy()

    return encode_texts


# ----------------------------
# KL DIVERGENCE CALCULATION
# ----------------------------
def compute_kl_divergence(embeddings_a, embeddings_b, bins=100):
    """
    Compute KL divergence between two sets of embeddings.
    We use histogram-based approximation for probability densities.
    """
    # Flatten features to 1D distribution for simplicity
    hist_a, _ = np.histogram(embeddings_a.flatten(), bins=bins, density=True)
    hist_b, _ = np.histogram(embeddings_b.flatten(), bins=bins, density=True)

    # Avoid zeros for KL calculation
    hist_a += 1e-10
    hist_b += 1e-10

    return entropy(hist_a, hist_b)  # KL(P || Q)


# ----------------------------
# PIPELINE
# ----------------------------
def compute_kl_for_all_corruptions(dataset, encode_fn):
    results = []
    print("Calculating KL divergences for DistilBERT embeddings...")
    clean_texts = dataset.get_testset_for_autoencoder()
    clean_embeddings = encode_fn(clean_texts)

    for corruption_type in AGNEWS_CORRUPTIONS:
        print(f"Processing corruption: {corruption_type}")
        corrupted_texts = dataset.get_corruptedset_for_autoencoder(corruption_type)
        corrupted_embeddings = encode_fn(corrupted_texts)
        kld = compute_kl_divergence(clean_embeddings, corrupted_embeddings)
        print(kld)
        print('---------------')
        results.append({"corruption_type": corruption_type, "divergence": kld})

    return pd.DataFrame(results)


# ----------------------------
# ANALYSIS & PLOTTING
# ----------------------------
def categorize_by_percentiles(pct_value):
    if pct_value <= 25:
        return 'Lowest'
    elif 25 < pct_value <= 75:
        return 'Mid-Range'
    else:
        return 'Highest'


def analyze_and_plot(results_df):
    grouped = results_df.groupby('corruption_type')['divergence'].mean().to_frame(name='mean')
    percentiles = grouped['mean'].rank(pct=True) * 100
    grouped['category'] = percentiles.apply(categorize_by_percentiles)

    seaborn_styles(sns)
    rcParams['font.size'] = 24
    rcParams['axes.titlesize'] = 32
    rcParams['axes.labelsize'] = 28
    rcParams['xtick.labelsize'] = 26
    rcParams['ytick.labelsize'] = 26
    rcParams['legend.fontsize'] = 18

    grouped = grouped.sort_values('mean')
    counts = grouped['category'].value_counts()
    means_by_cat = grouped.groupby('category')['mean'].mean().to_dict()

    plt.figure(figsize=(14, 10))
    sns.scatterplot(x=range(len(grouped)), y=grouped['mean'], hue=grouped['category'], palette="coolwarm", s=100)

    for i, (cat, mean) in enumerate(zip(grouped['category'], grouped['mean'])):
        plt.text(i, mean + 0.1, f"{cat}", ha="center", fontsize=14)

    plt.title("KL Divergence: DistilBERT (AG News vs Corruptions)")
    plt.xlabel("Corruptions (sorted)")
    plt.ylabel("KL Divergence")
    plt.xticks([])
    plt.tight_layout()
    plt.show()


# ----------------------------
# MAIN
# ----------------------------
def main():
    dataset = AGNewsDataset(max_sequence_length=MAX_SEQUENCE_LENGTH, vocab_size=None, batch_size=32)
    encode_fn = get_distilbert_encoder()
    results_df = compute_kl_for_all_corruptions(dataset, encode_fn)
    results_df.to_csv('distilbert_ood_characterization.csv')
    analyze_and_plot(results_df)


if __name__ == "__main__":
    main()
