import string
import tensorflow_datasets as tfds
import pandas as pd
import nltk
from nltk.corpus import wordnet
import spacy

nltk.download('wordnet')

# -----------------------------
# Character-Level Corruptions
# -----------------------------
def typo_injection(text, severity):
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

    for _ in range(severity):
        noise_type = random.choice(["add", "remove", "split"])

        if noise_type == "add":
            idx = random.randint(0, len(words) - 2)
            words[idx] += " " * random.randint(1, severity)

        elif noise_type == "remove":
            idx = random.randint(0, len(words) - 2)
            words[idx] += words.pop(idx + 1)  # Merge with next word

        elif noise_type == "split":
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            if len(word) > 3:  # Ensure the word is long enough
                split_pos = random.randint(1, len(word) - 2)
                words[idx] = word[:split_pos] + " " + word[split_pos:]

    corrupted_text = " ".join(words)
    corrupted_text = re.sub(r"\s+", lambda m: " " * random.randint(1, severity * 5), corrupted_text)
    return corrupted_text


def case_randomization(text, severity):
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
    words = text.split()
    num_deletions = severity * 2
    for _ in range(num_deletions):
        if words:
            del words[random.randint(0, len(words) - 1)]
    return " ".join(words)


def truncation(text, severity):
    words = text.split()
    cutoff = max(1, len(words) - severity * 3)
    return " ".join(words[:cutoff])


def word_order_shuffling(text, severity):
    words = text.split()
    if len(words) < 2:
        return text

    num_words_to_shuffle = max(1, int(len(words) * (severity / 5)))

    indices_to_shuffle = random.sample(range(len(words)), num_words_to_shuffle)
    shuffled_words = [words[i] for i in indices_to_shuffle]
    random.shuffle(shuffled_words)

    for idx, shuffled_word in zip(indices_to_shuffle, shuffled_words):
        words[idx] = shuffled_word

    return " ".join(words)


# -----------------------------
# Sentence-Level Corruptions
# -----------------------------
def sentence_noise_injection(text, severity):
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
    doc = nlp(text)
    masked_text = text
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]]

    if not entities:
        return masked_text

    num_to_mask = min(len(entities), max(1, int(len(entities) * (severity / 5.0))))
    entities_to_mask = random.sample(entities, num_to_mask)

    for ent_text, ent_label in entities_to_mask:
        placeholder = f"<{ent_label.lower()}>"
        masked_text = re.sub(rf"\b{re.escape(ent_text)}\b", placeholder, masked_text)

    return masked_text


from nltk.corpus import opinion_lexicon
nltk.download('opinion_lexicon')
def sentiment_masking(text, severity):
    words = text.split()
    new_words = []
    num_to_mask = int(len(words) * severity)  # Severity controls fraction

    pos_words = set(opinion_lexicon.positive())
    neg_words = set(opinion_lexicon.negative())
    sentiment_words = pos_words.union(neg_words)

    masked = 0
    for word in words:
        lower_word = word.lower().strip(string.punctuation)
        if lower_word in sentiment_words and masked < num_to_mask:
            new_words.append("<mask>")
            masked += 1
        else:
            new_words.append(word)

    return " ".join(new_words)


import re
import random
from num2words import num2words
def digits_to_words(text, severity):
    def replace_digits_in_word(word):
        return re.sub(r'\d+(\.\d+)?', lambda m: convert_number(m.group()), word)

    def convert_number(number):
        try:
            if '.' in number:
                return num2words(float(number))
            else:
                return num2words(int(number))
        except Exception:
            return number  # In case of error, return original

    if not (1 <= severity <= 5):
        raise ValueError("Severity must be between 1 and 5.")

    scale = severity / 5.0  # Normalize severity to 0.2 – 1.0

    words = re.findall(r'\w+|\W+', text)
    digit_word_indices = [i for i, w in enumerate(words) if re.search(r'\d', w)]

    num_to_replace = int(len(digit_word_indices) * scale)
    if num_to_replace == 0:
        return text  # No corruption if no matches

    selected_indices = random.sample(digit_word_indices, num_to_replace)

    for i in selected_indices:
        words[i] = replace_digits_in_word(words[i])

    return ''.join(words)


# -----------------------------
# Main Code
# -----------------------------
def load_ag_news_from_tfds():
    dataset = tfds.load("ag_news_subset", split="test", as_supervised=True)
    test_texts, test_labels = [], []
    for text, label in tfds.as_numpy(dataset):
        test_texts.append(text.decode("utf-8"))
        test_labels.append(label)

    return pd.DataFrame({"label": test_labels, "text": test_texts})

def apply_corruption(data, corruption_func, severity):
    corrupted_data = data.copy()
    corrupted_data["text"] = data["text"].apply(lambda x: corruption_func(x, severity))
    return corrupted_data


def main():
    output_dir = "./"
    severities = [1, 2, 3, 4, 5]
    dataset = load_ag_news_from_tfds()

    corruptions = {
        "typo": typo_injection,
        "whitespace": whitespace_noise,
        "synonym": synonym_replacement,
        "deletion": random_word_deletion,
        "insertion": random_word_insertion,
        "antonym": antonym_replacement,
        "sentence_noise_injection": sentence_noise_injection,
        "shuffling": word_order_shuffling,
        "truncation": truncation,
    }

    for name, func in corruptions.items():
        for severity in severities:
            corrupted_dataset = apply_corruption(dataset, func, severity)
            output_path = f"{output_dir}ag_news_{name}_{severity}.csv"

            corrupted_dataset.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
