import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.linalg import sqrtm


def compute_frechet_distance(original_texts, corrupted_texts):
    """
    Compute Fréchet Distance between two text datasets based on TF-IDF embeddings.

    Args:
        original_texts (list of str): List of original text samples.
        corrupted_texts (list of str): List of corrupted text samples.

    Returns:
        float: Fréchet Distance value between the original and corrupted datasets.
    """
    # Convert texts into TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_original = vectorizer.fit_transform(original_texts).toarray()
    tfidf_corrupted = vectorizer.transform(corrupted_texts).toarray()

    # Compute means and covariance matrices
    mu_original = np.mean(tfidf_original, axis=0)
    mu_corrupted = np.mean(tfidf_corrupted, axis=0)
    cov_original = np.cov(tfidf_original, rowvar=False)
    cov_corrupted = np.cov(tfidf_corrupted, rowvar=False)

    # Compute squared difference of means
    mean_diff = np.sum((mu_original - mu_corrupted) ** 2)

    # Compute square root of the product of covariance matrices
    cov_sqrt, _ = sqrtm(cov_original @ cov_corrupted, disp=False)

    # Handle numerical issues with complex values
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    # Compute trace term
    trace_term = np.trace(cov_original + cov_corrupted - 2 * cov_sqrt)

    # Return Fréchet Distance
    return mean_diff + trace_term