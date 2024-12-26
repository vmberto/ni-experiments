import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf


random_swap_aug = naw.RandomWordAug(action="swap", name="Random_Swap_Aug", aug_min=1, aug_max=5, aug_p=0.1)
random_substitute_aug = naw.RandomWordAug(action="substitute", name="Random_Substitute_Aug", aug_min=1, aug_max=5, aug_p=0.1)
random_char_aug = nac.RandomCharAug(action='substitute', name='RandomChar_Aug', aug_char_min=1, aug_char_max=5, aug_char_p=0.1)

def get_default_aug_layers():
    return naf.Sequential([
        random_swap_aug,
        random_substitute_aug,
        random_char_aug,
    ])


def get_default_aug_layers_mixed():
    return naf.Sometimes([
        random_swap_aug,
        random_substitute_aug,
        random_char_aug,
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