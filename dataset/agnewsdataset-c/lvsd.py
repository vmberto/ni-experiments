import Levenshtein


def compute_levenshtein_distance(original_texts, corrupted_texts):
    """
    Computes the average Levenshtein distance and normalized severity between
    original and corrupted texts.

    Args:
        original_texts (list): List of original text samples.
        corrupted_texts (list): List of corresponding corrupted text samples.

    Returns:
        dict: Average Levenshtein distance and normalized severity score.
    """
    total_distance = 0
    total_normalized = 0
    num_samples = len(original_texts)

    for original, corrupted in zip(original_texts, corrupted_texts):
        # Calculate Levenshtein Distance
        distance = Levenshtein.distance(original, corrupted)
        total_distance += distance

        # Normalize by original text length
        normalized_distance = distance / max(1, len(original))
        total_normalized += normalized_distance

    # Average metrics
    avg_distance = total_distance / num_samples
    avg_normalized_severity = total_normalized / num_samples

    return {
        "average_distance": avg_distance,
        "normalized_severity": avg_normalized_severity
    }


# Example Usage
if __name__ == "__main__":
    # Simulate datasets
    original_texts = [
        "Scientists discovered a new planet orbiting a distant star.",
        "The economy is growing steadily with strong employment rates.",
        "The championship finals will take place next Sunday."
    ]

    corrupted_texts = [
        "ScIenTists disovered  a new plaNET orbitng  distant starrr.",
        "Th ecnomy growig steadly with emploment rates.",
        "The chmpionship finls wil tak place Sundy."
    ]

    # Compute Levenshtein severity
    results = compute_levenshtein_distance(original_texts, corrupted_texts)
    print(f"Average Levenshtein Distance: {results['average_distance']:.2f}")
    print(f"Normalized Severity Score: {results['normalized_severity']:.4f}")