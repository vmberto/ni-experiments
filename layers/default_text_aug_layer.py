import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf

# Define augmentations

# RandomFlip Equivalent: Random word swap
random_swap_aug = naw.RandomWordAug(action="swap", name="Random_Swap_Aug", aug_min=1, aug_max=3, aug_p=0.3)

# RandomRotation Equivalent: Synonym replacement
synonym_aug = naw.SynonymAug(aug_src="wordnet", name="Synonym_Aug", aug_min=1, aug_max=3, aug_p=0.3)

# RandomZoom Equivalent: Random word deletion
random_delete_aug = naw.RandomWordAug(action="delete", name="Random_Delete_Aug", aug_min=1, aug_max=3, aug_p=0.3)


def get_default_aug_layers():
    """
    Returns a sequential NLP augmentation pipeline with:
    - Random word swap (RandomFlip equivalent)
    - Synonym replacement (RandomRotation equivalent)
    - Random word deletion (RandomZoom equivalent)
    """
    return naf.Sequential([
        random_swap_aug,   # RandomFlip equivalent
        synonym_aug,       # RandomRotation equivalent
        random_delete_aug  # RandomZoom equivalent
    ])


# Example usage
if __name__ == "__main__":
    # Create the augmentation pipeline
    aug_pipeline = get_default_aug_layers()

    # Test text
    text = "This is a sample sentence for augmentation testing."

    # Apply the augmentation pipeline
    augmented_text = aug_pipeline.augment(text)

    print("Original Text:")
    print(text)
    print("\nAugmented Text:")
    print(augmented_text)