import os
import pandas as pd
from keras import models
from lib.consts import AGNEWS_CORRUPTIONS
from models.text_autoencoder import TextAutoencoder
from dataset.agnewsdataset import AGNewsDataset

MAX_SEQUENCE_LENGTH = 128
VOCAB_SIZE = 10000
BATCH_SIZE = 256
EMBEDDING_DIM = 128
LATENT_DIM = 128
MODEL_PATH = "../saved_models/text_encoder/static_encoder.keras"
RESULT_PATH = "../output/agnews_encoder_kldiv.csv"


def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    dataset = AGNewsDataset(
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
        batch_size=BATCH_SIZE
    )

    x_test = dataset.get_testset_for_autoencoder()  # Just to get full x_test
    test_ds = dataset.get_dataset_for_autoencoder(x_test)

    # Load or build encoder
    encoder = TextAutoencoder(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        max_len=MAX_SEQUENCE_LENGTH,
        latent_dim=LATENT_DIM
    )

    # Compute KL divergence for each corruption
    results = []
    print("Calculating KL divergences...")
    for corruption_type in AGNEWS_CORRUPTIONS:
        print(f"  Processing corruption: {corruption_type}")
        kld = dataset.prepare_agnews_c_with_distances(encoder, corruption_type, test_ds)

        results.append({
            "corruption_type": corruption_type,
            "divergence": kld,
        })

        # Save incrementally
        pd.DataFrame(results).to_csv(RESULT_PATH, index=False)
        print(f"Saved interim results after corruption: {corruption_type}")

    print("\n=== Completed ===")
    print(f"KL Divergences saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
