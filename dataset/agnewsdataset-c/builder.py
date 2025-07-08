import string
from kldiv import compute_kl_divergence
from lib.consts import AGNEWS_CORRUPTIONS
from lvsd import compute_levenshtein_distance
import tensorflow_datasets as tfds
import pandas as pd
import os
import random
import nltk
from nltk.corpus import wordnet
import re
import spacy

nltk.download('wordnet')

# -----------------------------
# Character-Level Corruptions
# -----------------------------
def typo_injection(text, severity):
    """Introduce random typos by swapping adjacent characters."""
    text = list(text)
    num_typos = severity * 12
    for _ in range(num_typos):
        if len(text) < 2:
            break
        idx = random.randint(0, len(text) - 2)
        text[idx], text[idx + 1] = text[idx + 1], text[idx]
    return ''.join(text)


def whitespace_noise(text, severity):
    words = text.split()

    # Severity controls the number of spaces to add/remove
    for _ in range(severity):
        noise_type = random.choice(["add", "remove", "split"])

        if noise_type == "add":
            # Add random extra spaces between words
            idx = random.randint(0, len(words) - 2)
            words[idx] += " " * random.randint(1, severity)

        elif noise_type == "remove":
            # Remove spaces between words
            idx = random.randint(0, len(words) - 2)
            words[idx] += words.pop(idx + 1)  # Merge with next word

        elif noise_type == "split":
            # Split a word by adding spaces in the middle
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            if len(word) > 3:  # Ensure the word is long enough
                split_pos = random.randint(1, len(word) - 2)
                words[idx] = word[:split_pos] + " " + word[split_pos:]

    corrupted_text = " ".join(words)
    corrupted_text = re.sub(r"\s+", lambda m: " " * random.randint(1, severity * 5), corrupted_text)
    return corrupted_text


def case_randomization(text, severity):
    """
    Randomly alter the case of letters in the text.

    Args:
        text (str): Input text.
        severity (int): Severity level (1-5), determines the frequency of randomization.

    Returns:
        str: Text with random case applied.
    """
    randomized_text = []
    for char in text:
        if char.isalpha() and random.random() < severity / 5:  # Severity controls randomness
            randomized_text.append(char.upper() if random.choice([True, False]) else char.lower())
        else:
            randomized_text.append(char)
    return "".join(randomized_text)


# -----------------------------
# Word-Level Corruptions
# -----------------------------
def synonym_replacement(text, severity):
    """Replace words with their synonyms."""
    words = text.split()
    for _ in range(severity * 20):
        if not words:
            break
        idx = random.randint(0, len(words) - 1)
        synonyms = wordnet.synsets(words[idx])
        if synonyms:
            replacement = synonyms[0].lemmas()[0].name()
            words[idx] = replacement.replace("_", " ")
    return " ".join(words)


def antonym_replacement(text, severity):
    words = text.split()
    for _ in range(severity * 20):
        idx = random.randint(0, len(words) - 1)
        antonyms = []
        for syn in wordnet.synsets(words[idx]):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.append(lemma.antonyms()[0].name())
        if antonyms:
            words[idx] = random.choice(antonyms).replace("_", " ")
    return " ".join(words)


def random_word_insertion(text, severity):
    words = text.split()
    num_insertions = severity * 20

    for _ in range(num_insertions):
        idx = random.randint(0, len(words))
        if words:
            original_word = random.choice(words)
            synonyms = wordnet.synsets(original_word)
            if synonyms:
                inserted_word = synonyms[0].lemmas()[0].name().replace("_", " ")
                words.insert(idx, inserted_word)
            else:
                words.insert(idx, random.choice(["foo", "bar", "baz", "qux", "noise", "xyz", "zxy", "abc", "cba"]))
    return " ".join(words)


def random_word_deletion(text, severity):
    """Randomly delete words based on severity."""
    words = text.split()
    num_deletions = severity * 20
    for _ in range(num_deletions):
        if words:
            del words[random.randint(0, len(words) - 1)]
    return " ".join(words)


def word_order_shuffling(text, severity):
    words = text.split()
    if len(words) < 2:
        return text  # Skip shuffling for short texts

    # Calculate the number of words to shuffle based on severity
    num_words_to_shuffle = max(1, int(len(words) * (severity / 5)))

    # Randomly select words to shuffle
    indices_to_shuffle = random.sample(range(len(words)), num_words_to_shuffle)
    shuffled_words = [words[i] for i in indices_to_shuffle]
    random.shuffle(shuffled_words)

    # Replace the selected words in the original order with shuffled ones
    for idx, shuffled_word in zip(indices_to_shuffle, shuffled_words):
        words[idx] = shuffled_word

    return " ".join(words)


# -----------------------------
# Sentence-Level Corruptions
# -----------------------------
def sentence_noise_injection(text, severity):
    """
    Inject random noise tokens into a sentence based on severity.

    Args:
        text (str): The input sentence.
        severity (int): The severity level (1-5), controls the amount of noise injected.

    Returns:
        str: Text with noise tokens injected.
    """
    words = text.split()
    num_noise_tokens = severity * max(1, len(words) // 10) * 20  # Proportional to severity

    noise_types = ["gibberish", "symbols", "numbers", "random_word"]

    for _ in range(num_noise_tokens):
        noise_type = random.choice(noise_types)

        noise = ''
        if noise_type == "gibberish":
            # Generate a random gibberish word
            noise = ''.join(random.choices(string.ascii_letters, k=random.randint(3, 8)))
        elif noise_type == "symbols":
            # Inject random special symbols
            noise = ''.join(random.choices(string.punctuation, k=random.randint(2, 4)))
        elif noise_type == "numbers":
            # Inject random numbers
            noise = str(random.randint(0, 9999))
        elif noise_type == "random_word":
            # Use a placeholder word
            noise = random.choice(["foo", "bar", "baz", "qux", "noise"])

        # Insert noise at a random position
        insert_position = random.randint(0, len(words))
        words.insert(insert_position, noise)

    return " ".join(words)


# -----------------------------
# Semantic Corruptions
# -----------------------------
nlp = spacy.load("en_core_web_sm")


def entity_masking(text, severity):
    """
    Replace a proportion of named entities in text with placeholders based on severity.

    Args:
        text (str): Input text.
        severity (int): Severity level (1-5), determines the proportion of entities to mask.

    Returns:
        str: Text with named entities replaced by placeholders.
    """
    doc = nlp(text)
    masked_text = text
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]]

    # If no entities are found, return the original text
    if not entities:
        return masked_text

    # Calculate the number of entities to mask based on severity
    num_to_mask = min(len(entities), max(1, int(len(entities) * (severity / 5.0))))

    # Randomly sample entities to mask
    entities_to_mask = random.sample(entities, num_to_mask)

    # Replace selected entities with placeholders
    for ent_text, ent_label in entities_to_mask:
        placeholder = f"<{ent_label.lower()}>"
        masked_text = re.sub(rf"\b{re.escape(ent_text)}\b", placeholder, masked_text)

    return masked_text

def load_ag_news_from_tfds():
    dataset = tfds.load("ag_news_subset", split="test", as_supervised=True)
    test_texts, test_labels = [], []
    for text, label in tfds.as_numpy(dataset):
        test_texts.append(text.decode("utf-8"))
        test_labels.append(label)

    return pd.DataFrame({"label": test_labels, "text": test_texts})


# Apply Corruptions
def apply_corruption(data, corruption_func, severity):
    corrupted_data = data.copy()
    corrupted_data["text"] = data["text"].apply(lambda x: corruption_func(x, severity))
    return corrupted_data


# Main Function
def main():
    output_dir = "./"
    severities = [1, 2, 3, 4, 5]
    dataset = load_ag_news_from_tfds()

    # Define corruptions
    corruptions = {
        # "typo": typo_injection,
        # "whitespace": whitespace_noise,
        # "case_randomization": case_randomization,
        # "synonym": synonym_replacement,
        # "deletion": random_word_deletion,
        # "insertion": random_word_insertion,
        # "antonym": antonym_replacement,
        # "sentence_noise_injection": sentence_noise_injection,
        "shuffling": word_order_shuffling,
    }

    for name, func in corruptions.items():
        for severity in severities:
            corrupted_dataset = apply_corruption(dataset, func, severity)
            output_path = f"{output_dir}ag_news_{name}_{severity}.csv"

            kl_divergence = compute_kl_divergence(dataset['text'], corrupted_dataset['text'])
            levenshtein_distance = compute_levenshtein_distance(dataset['text'], corrupted_dataset['text'])

            print(f"KL Divergence between original and {name}_{severity}: {kl_divergence:.4f}")
            print(f"Average Levenshtein Distance between original and {name}_{severity}:: {levenshtein_distance['average_distance']:.2f}")

            corrupted_dataset.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")


def calculate_distances_with_characterization():
    ds = load_ag_news_from_tfds()

    results = []

    # Iterate over all corruptions
    for name in AGNEWS_CORRUPTIONS:
        corrupted_ds_path = f"{os.getcwd()}/ag_news_{name}.csv"
        corrupted_ds = pd.read_csv(corrupted_ds_path)

        # Extract texts
        original_texts = ds['text'].astype(str).tolist()
        corrupted_texts = corrupted_ds['text'].astype(str).tolist()

        # Compute the distance
        distance = compute_levenshtein_distance(original_texts, corrupted_texts)
        distance = distance['average_distance']

        # Append results
        results.append({
            "corruption": name,
            "divergence": distance
        })

    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    output_path = "../../results/agnews/agnews_c_divergences.csv"
    df.to_csv(output_path, index=False)

    print(f"Characterized CSV saved at {output_path}")


if __name__ == "__main__":
    calculate_distances_with_characterization()
