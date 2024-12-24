import tensorflow as tf
from keras import layers
from layers.random_salt_and_pepper import RandomSaltAndPepper


def main():
    # A medium-to-large sentence:
    sentence = "Yahoo to buy online music company for $160 million \
    The acquisition is expected to close by the end of the year, pending regulatory approval, and will strengthen Yahoo’s move into paid music services."

    # We'll create a small dataset of 1 sample so we can adapt the vectorizer.
    text_ds = tf.data.Dataset.from_tensor_slices([sentence])

    # Define a TextVectorization layer to tokenize and integer-encode the sentence.
    # You can adjust max_tokens and output_sequence_length as needed.
    vectorizer = layers.TextVectorization(
        max_tokens=10000,
        output_mode='int',
    )

    # Adapt the vectorizer to the text. (Required so it learns the vocabulary)
    vectorizer.adapt(text_ds)

    print("Original Text:\n", sentence)
    # Vectorize the sentence
    # shape: (1, 20) because we have 1 sample with 20 tokens
    vectorized_text = vectorizer([sentence])
    print("\nVectorized text (int):\n", vectorized_text.numpy())

    # Cast to float, since our noise layer deals with floating-point replacements
    # shape: (1, 20)
    float_vectorized_text = tf.cast(vectorized_text, dtype=tf.float32)

    # Instantiate our RandomSaltAndPepper layer
    # Let's use factor=0.3 for a heavier corruption, and a fixed seed for reproducibility
    salt_pepper_layer = RandomSaltAndPepper(factor=0.3, seed=42)

    # Apply it in 'training=True' mode to see the noise effect.
    # We can do this simply by calling the layer inside a GradientTape,
    # or building a small model. Here we'll just use 'call(..., training=True)'
    noisy_text = salt_pepper_layer(float_vectorized_text, training=True)

    print("\nNoisy vectorized text (float):\n", noisy_text.numpy())


if __name__ == "__main__":
    main()
