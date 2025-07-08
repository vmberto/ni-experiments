from collections import Counter
import numpy as np
from scipy.special import rel_entr


def compute_kl_divergence(original_texts, corrupted_texts, top_k=10000):
    """
    Compute KL Divergence between token distributions of original and corrupted datasets.

    Args:
        original_texts (list of str): List of original text samples.
        corrupted_texts (list of str): List of corrupted text samples.
        top_k (int): The maximum number of most common tokens to consider.

    Returns:
        float: KL Divergence value between original and corrupted token distributions.
    """
    # Tokenize texts
    original_tokens = " ".join(original_texts).split()
    corrupted_tokens = " ".join(corrupted_texts).split()

    # Count word frequencies
    original_counts = Counter(original_tokens)
    corrupted_counts = Counter(corrupted_tokens)

    # Get the top-k common tokens
    original_common = dict(original_counts.most_common(top_k))
    corrupted_common = dict(corrupted_counts.most_common(top_k))
    all_tokens = set(original_common.keys()).union(set(corrupted_common.keys()))

    # Create aligned probability distributions
    p = np.array([original_common.get(token, 0) for token in all_tokens], dtype=float)
    q = np.array([corrupted_common.get(token, 0) for token in all_tokens], dtype=float)

    # Normalize to get probability distributions
    p = p / p.sum()
    q = q / q.sum()

    # Add epsilon to avoid log(0) errors
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    # Compute KL Divergence
    kl_div = np.sum(rel_entr(p, q))
    return kl_div


# Example Usage
if __name__ == "__main__":
    import pandas as pd

    # Simulate original and corrupted datasets
    original_df = pd.DataFrame({"text": [
        "Scientists discovered a new planet orbiting a distant star.",
        "The stock market experienced significant gains this week.",
        "A new breakthrough in artificial intelligence was announced.",
        "Sports teams around the world are preparing for the championship.",
        "Economy reports show positive growth in multiple sectors."
    ]})

    corrupted_df = pd.DataFrame({"text": [
        "Scientts disovered  a new planett orbitng  distant  starrrr.",
        "Th stock mrket experinced gains this wkeek.",
        "New breakthrough  AI announced!",
        "Sports teamss prep for chamionship worldwide.",
        "Economy showss groth  multi sectors."
    ]})

    # Compute KL Divergence
    kl_divergence = compute_kl_divergence(original_df["text"], corrupted_df["text"])
    print(f"KL Divergence between original and corrupted datasets: {kl_divergence:.4f}")
