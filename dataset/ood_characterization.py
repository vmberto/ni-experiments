import numpy as np
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from typing import Dict
from abc import ABC, abstractmethod


class DistributionComparer(ABC):
    """
    Abstract base class for comparing distributions from latent representations.
    """

    @abstractmethod
    def compare(self, dist_in: np.ndarray, dist_out: np.ndarray) -> float:
        """Compare two distributions and return a metric."""
        pass


class KLDivergenceComparer(DistributionComparer):
    """
    Compare distributions using KL divergence with proper normalization.
    Implements the algorithm as specified in the paper.
    """

    def __init__(self, epsilon: float = 1e-10, method: str = 'histogram'):
        """
        Args:
            epsilon: Small constant to avoid log(0)
            method: 'histogram' (default), 'flatten', or 'feature_wise'
        """
        self.epsilon = epsilon
        self.method = method

    def _normalize_to_distribution(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to form a valid probability distribution.
        Ensures all values are positive and sum to 1.
        """
        # Shift to ensure all values are positive
        data_shifted = data - np.min(data) + self.epsilon
        # Normalize to sum to 1
        return data_shifted / np.sum(data_shifted)

    def _histogram_method(self, dist_in: np.ndarray, dist_out: np.ndarray,
                          bins: int = 100) -> float:
        """
        Create histograms of the distributions and compute KL divergence.
        This handles different sized distributions properly.
        """
        # Flatten both distributions
        flat_in = dist_in.flatten()
        flat_out = dist_out.flatten()

        # Determine common range for binning
        min_val = min(np.min(flat_in), np.min(flat_out))
        max_val = max(np.max(flat_in), np.max(flat_out))

        # Create histograms with the same bins
        hist_in, bin_edges = np.histogram(flat_in, bins=bins,
                                          range=(min_val, max_val),
                                          density=False)
        hist_out, _ = np.histogram(flat_out, bins=bins,
                                   range=(min_val, max_val),
                                   density=False)

        # Normalize histograms to probability distributions
        p = self._normalize_to_distribution(hist_in.astype(float))
        q = self._normalize_to_distribution(hist_out.astype(float))

        return entropy(p, q)

    def _flatten_method(self, dist_in: np.ndarray, dist_out: np.ndarray) -> float:
        """
        Direct flattening method as in your algorithm.
        WARNING: Requires same size distributions.
        """
        flat_in = dist_in.flatten()
        flat_out = dist_out.flatten()

        if len(flat_in) != len(flat_out):
            raise ValueError(
                f"Distributions must have same size for flatten method. "
                f"Got {len(flat_in)} vs {len(flat_out)}"
            )

        # Normalize following the algorithm
        p = self._normalize_to_distribution(flat_in)
        q = self._normalize_to_distribution(flat_out)

        return entropy(p, q)

    def _feature_wise_method(self, dist_in: np.ndarray,
                             dist_out: np.ndarray) -> float:
        """
        Compute KL divergence for each latent dimension and average.
        This is useful when distributions have different numbers of samples.
        """
        if dist_in.shape[1] != dist_out.shape[1]:
            raise ValueError(
                f"Latent dimensions must match. "
                f"Got {dist_in.shape[1]} vs {dist_out.shape[1]}"
            )

        kl_per_feature = []
        num_features = dist_in.shape[1]

        for i in range(num_features):
            # Use histogram method for each feature
            feat_in = dist_in[:, i]
            feat_out = dist_out[:, i]

            # Create histograms
            hist_in, bin_edges = np.histogram(feat_in, bins=50, density=False)
            hist_out, _ = np.histogram(feat_out, bins=bin_edges, density=False)

            # Normalize
            p = self._normalize_to_distribution(hist_in.astype(float))
            q = self._normalize_to_distribution(hist_out.astype(float))

            kl_per_feature.append(entropy(p, q))

        return np.mean(kl_per_feature)

    def compare(self, dist_in: np.ndarray, dist_out: np.ndarray) -> float:
        """
        Compare two distributions using KL divergence.

        Args:
            dist_in: In-distribution latent representations (n_in, latent_dim)
            dist_out: Out-of-distribution latent representations (n_out, latent_dim)

        Returns:
            KL divergence value
        """
        if self.method == 'histogram':
            return self._histogram_method(dist_in, dist_out)
        elif self.method == 'flatten':
            return self._flatten_method(dist_in, dist_out)
        elif self.method == 'feature_wise':
            return self._feature_wise_method(dist_in, dist_out)
        else:
            raise ValueError(f"Unknown method: {self.method}")

class WassersteinComparer(DistributionComparer):
    """
    Compare distributions using Wasserstein distance in latent space.

    Modes:
      - 'per_feature': computes 1D Wasserstein per latent dim, then averages.
      - 'sliced': Sliced Wasserstein with random projections (no extra deps).
    """

    def __init__(self, mode: str = 'per_feature', n_projections: int = 128, random_state: int = 42):
        """
        Args:
            mode: 'per_feature' or 'sliced'
            n_projections: number of random projections (sliced mode)
            random_state: random seed for reproducibility (sliced mode)
        """
        assert mode in ('per_feature', 'sliced'), "mode must be 'per_feature' or 'sliced'"
        self.mode = mode
        self.n_projections = n_projections
        self.rng = np.random.default_rng(random_state)

    def _per_feature(self, dist_in: np.ndarray, dist_out: np.ndarray) -> float:
        if dist_in.ndim == 1:
            # Fall back to single-dimension case
            return float(wasserstein_distance(dist_in, dist_out))

        if dist_in.shape[1] != dist_out.shape[1]:
            raise ValueError(
                f"Latent dimensions must match. Got {dist_in.shape[1]} vs {dist_out.shape[1]}"
            )

        wds = []
        for d in range(dist_in.shape[1]):
            wds.append(wasserstein_distance(dist_in[:, d], dist_out[:, d]))
        return float(np.mean(wds))

    def _sliced(self, dist_in: np.ndarray, dist_out: np.ndarray) -> float:
        """
        Sliced Wasserstein: project both clouds onto random 1D directions and
        average the 1D Wasserstein distances.
        """
        # Ensure 2D arrays: (n_samples, latent_dim)
        dist_in = np.atleast_2d(dist_in)
        dist_out = np.atleast_2d(dist_out)

        if dist_in.shape[1] != dist_out.shape[1]:
            raise ValueError(
                f"Latent dimensions must match. Got {dist_in.shape[1]} vs {dist_out.shape[1]}"
            )

        d = dist_in.shape[1]
        total = 0.0
        for _ in range(self.n_projections):
            # Random unit vector in R^d
            v = self.rng.normal(size=d)
            v /= (np.linalg.norm(v) + 1e-12)

            proj_in = dist_in @ v
            proj_out = dist_out @ v
            total += wasserstein_distance(proj_in, proj_out)

        return float(total / self.n_projections)

    def compare(self, dist_in: np.ndarray, dist_out: np.ndarray) -> float:
        if self.mode == 'per_feature':
            return self._per_feature(dist_in, dist_out)
        else:  # 'sliced'
            return self._sliced(dist_in, dist_out)


class GenericOODCharacterization:
    """
    Generic algorithm for characterizing OOD data using latent representations.
    """

    def __init__(self, encoder, comparer: DistributionComparer):
        """
        Args:
            encoder: Trained encoder model (e.g., autoencoder.encoder)
            comparer: DistributionComparer instance for comparing distributions
        """
        self.encoder = encoder
        self.comparer = comparer

    def characterize(self, x_in: np.ndarray, x_out: np.ndarray) -> Dict:
        """
        Characterize OOD data by comparing latent representations.

        Args:
            x_in: In-distribution data (n_in, height, width, channels)
            x_out: Out-of-distribution data (n_out, height, width, channels)

        Returns:
            Dictionary with results including metric value and metadata
        """
        # Encode both datasets
        z_in = self.encoder.predict(x_in)
        z_out = self.encoder.predict(x_out)

        # Compare distributions
        metric_value = self.comparer.compare(z_in, z_out)

        return {
            'metric_value': metric_value,
            'z_in_shape': z_in.shape,
            'z_out_shape': z_out.shape,
            'comparer_type': self.comparer.__class__.__name__
        }


# Updated function for your existing code
def calculate_kl_divergence(latent_clean: np.ndarray,
                            latent_corrupted: np.ndarray,
                            method: str = 'histogram',
                            epsilon: float = 1e-10) -> float:
    """
    Calculate KL divergence between two latent distributions with proper normalization.

    Args:
        latent_clean: Clean latent representations (n_clean, latent_dim)
        latent_corrupted: Corrupted latent representations (n_corrupted, latent_dim)
        method: 'histogram', 'flatten', or 'feature_wise'
        epsilon: Small constant to avoid numerical issues

    Returns:
        KL divergence value
    """
    comparer = KLDivergenceComparer(epsilon=epsilon, method=method)
    return comparer.compare(latent_clean, latent_corrupted)


# Example usage showing all methods
if __name__ == "__main__":
    # Simulate latent representations with different sizes
    np.random.seed(42)
    latent_clean = np.random.randn(1000, 128)  # 1000 samples, 128 dimensions
    latent_corrupted = np.random.randn(500, 128)  # 500 samples, 128 dimensions

    print("Testing different methods:")
    print("=" * 60)

    # Method 1: Histogram (recommended for different sized distributions)
    kl_hist = calculate_kl_divergence(latent_clean, latent_corrupted,
                                      method='histogram')
    print(f"Histogram method KL: {kl_hist:.6f}")

    # Method 2: Feature-wise (also handles different sizes)
    kl_feat = calculate_kl_divergence(latent_clean, latent_corrupted,
                                      method='feature_wise')
    print(f"Feature-wise method KL: {kl_feat:.6f}")

    # Method 3: Flatten (only works with same size)
    latent_corrupted_same = np.random.randn(1000, 128)
    kl_flat = calculate_kl_divergence(latent_clean, latent_corrupted_same,
                                      method='flatten')
    print(f"Flatten method KL (same size): {kl_flat:.6f}")

    print("\n" + "=" * 60)
    print("Shapes:")
    print(f"  latent_clean: {latent_clean.shape}")
    print(f"  latent_corrupted: {latent_corrupted.shape}")
