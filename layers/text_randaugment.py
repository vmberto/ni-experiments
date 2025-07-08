import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf

def get_text_randaugment_layers(
    swap_min=1, swap_max=3, swap_p=0.1,
    sub_min=1, sub_max=3, sub_p=0.1,
    char_min=1, char_max=3, char_p=0.1
):
    random_swap_aug = naw.RandomWordAug(
        action="swap",
        name="Random_Swap_Aug",
        aug_min=swap_min,
        aug_max=swap_max,
        aug_p=swap_p
    )
    random_substitute_aug = naw.RandomWordAug(
        action="substitute",
        name="Random_Substitute_Aug",
        aug_min=sub_min,
        aug_max=sub_max,
        aug_p=sub_p
    )
    random_char_aug = nac.RandomCharAug(
        action='substitute',
        name='RandomChar_Aug',
        aug_char_min=char_min,
        aug_char_max=char_max,
        aug_char_p=char_p
    )

    return naf.Sequential([
        random_swap_aug,
        random_substitute_aug,
        random_char_aug,
    ])


# Example usage
if __name__ == "__main__":
    # Create the augmentation pipeline
    aug_pipeline = get_text_randaugment_layers()

    # Test text
    text = "This is a sample sentence for augmentation testing."

    # Apply the augmentation pipeline
    augmented_text = aug_pipeline.augment(text)

    print("Original Text:")
    print(text)
    print("\nAugmented Text:")
    print(augmented_text)
