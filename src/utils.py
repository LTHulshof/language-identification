"""
Script containing helper functions for train.py
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def load_data(x_path, y_path):
    """
    Load training data into lists
    """
    with open(x_path, encoding="utf-8") as f_x, open(y_path, encoding="utf-8") as f_y:
        texts = [line.strip().split() for line in f_x]
        labels = [line.strip().split() for line in f_y]
    return texts, labels



def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize the datapoints into sub-word tokens using the (roberta by default) tokenizer
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )

    all_labels = []
    for i, label_seq in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Special tokens
            else:
                label_ids.append(label_seq[word_id])
            previous_word_id = word_id

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

def compute_metrics(p):
    """
    Compute metrics on evaluation data during training
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_preds = []
    for pred, lab in zip(predictions, labels):
        for p_, l_ in zip(pred, lab):
            if l_ != -100:
                true_labels.append(l_)
                true_preds.append(p_)

    return {
        "accuracy": accuracy_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds, average="macro"),
    }
