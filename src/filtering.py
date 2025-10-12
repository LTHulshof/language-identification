"""
Script for reducing the wili-2018 dataset to datapoints to only contain desired languages. Split up into train, evaluation and testing data.
"""

import os
import random
from collections import defaultdict

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, "..", "data", "wili-2018")
OUTPUT_DIR = os.path.join(SRC_DIR, "..", "data", "wili-preprocessed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_LANGS = {"deu", "spa", "nld", "ell", "eng"}


def filter_train():
    """
    Filter and save the training set
    """
    x_path = os.path.join(DATA_DIR, "x_train.txt")
    y_path = os.path.join(DATA_DIR, "y_train.txt")

    with open(x_path, encoding="utf-8") as xf, open(y_path, encoding="utf-8") as yf:
        texts = xf.readlines()
        labels = yf.readlines()

    filtered_texts = []
    filtered_labels = []
    for text, label in zip(texts, labels):
        label = label.strip()
        if label in TARGET_LANGS:
            filtered_texts.append(text.strip())
            filtered_labels.append(label)

    # Save filtered train set
    with open(os.path.join(OUTPUT_DIR, "x_train.txt"), "w", encoding="utf-8") as xf:
        xf.write("\n".join(filtered_texts))
    with open(os.path.join(OUTPUT_DIR, "y_train.txt"), "w", encoding="utf-8") as yf:
        yf.write("\n".join(filtered_labels))

    print(f"Saved {len(filtered_texts)} train samples")



def filter_and_split_test_eval():
    """
    Filter, split, and shuffle the test set
    """
    x_path = os.path.join(DATA_DIR, "x_test.txt")
    y_path = os.path.join(DATA_DIR, "y_test.txt")

    with open(x_path, encoding="utf-8") as xf, open(y_path, encoding="utf-8") as yf:
        texts = xf.readlines()
        labels = yf.readlines()

    # Collect samples per language
    lang_to_samples = defaultdict(list)
    for text, label in zip(texts, labels):
        label = label.strip()
        if label in TARGET_LANGS:
            lang_to_samples[label].append((text.strip(), label))

    eval_samples = []
    test_samples = []

    # Split per language: 100 test, remaining (400) eval
    for lang in TARGET_LANGS:
        samples = lang_to_samples[lang]
        random.seed(42)
        random.shuffle(samples)
        test_samples.extend(samples[:100])
        eval_samples.extend(samples[100:])

    # Shuffle across all languages
    random.seed(42)
    random.shuffle(eval_samples)
    random.shuffle(test_samples)

    # Unzip
    eval_texts, eval_labels = zip(*eval_samples)
    test_texts, test_labels = zip(*test_samples)

    # Save eval set
    with open(os.path.join(OUTPUT_DIR, "x_eval.txt"), "w", encoding="utf-8") as xf:
        xf.write("\n".join(eval_texts))
    with open(os.path.join(OUTPUT_DIR, "y_eval.txt"), "w", encoding="utf-8") as yf:
        yf.write("\n".join(eval_labels))

    # Save final test set
    with open(os.path.join(OUTPUT_DIR, "x_test.txt"), "w", encoding="utf-8") as xf:
        xf.write("\n".join(test_texts))
    with open(os.path.join(OUTPUT_DIR, "y_test.txt"), "w", encoding="utf-8") as yf:
        yf.write("\n".join(test_labels))


if __name__ == "__main__":
    filter_train()
    filter_and_split_test_eval()
    print("Saved filtered data")