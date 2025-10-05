"""
Script for evaluating the finetuned roberta model on generated test sets. Print token and sentence-level metrics.
"""

import os
import torch
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from tqdm import tqdm


SRC_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "model", "roberta-finetuned"))
MODEL_DIR = os.path.normpath(MODEL_DIR)
DATA_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "data", "wili-preprocessed"))
DATA_DIR = os.path.normpath(DATA_DIR)


# Load tokenizer and model
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR, add_prefix_space=True)
model = RobertaForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()
id2label = model.config.id2label
label2id = {v: k for k, v in id2label.items()}


def predict_tokens(words):
    """
    Predicts token-level labels for a list of words.
    """
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=-1).squeeze().tolist()
    word_ids = inputs.word_ids()

    word_preds = []
    last_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != last_word_id:
            last_word_id = word_id
            word_preds.append(id2label[preds[idx]])
        else:
            word_preds[-1] = id2label[preds[idx]]
    return word_preds



def evaluate_dataset(x_path, y_path):
    """
    Evaluate dataset and print metrics
    """
    print(f"\n📘 Evaluating:\nX: {x_path}\nY: {y_path}\n")
    with open(x_path, encoding="utf-8") as f:
        x_lines = [line.strip() for line in f if line.strip()]
    with open(y_path, encoding="utf-8") as f:
        y_lines = [line.strip() for line in f if line.strip()]

    assert len(x_lines) == len(y_lines), "x and y files must have the same number of lines"

    all_preds, all_labels = [], []
    sent_level_preds, sent_level_labels = [], []

    for sent, labels in tqdm(zip(x_lines, y_lines), total=len(x_lines), desc="Processing sentences", ncols=100):
        words = sent.split()
        gold_labels = labels.split()
        preds = predict_tokens(words)

        # Tokenizer truncation
        min_len = min(len(preds), len(gold_labels))
        preds, gold_labels = preds[:min_len], gold_labels[:min_len]

        # token-level
        all_preds.extend(preds)
        all_labels.extend(gold_labels)

        # sentence-level majority vote
        gold_majority = Counter(gold_labels).most_common(1)[0][0]
        pred_majority = Counter(preds).most_common(1)[0][0]

        sent_level_labels.append(gold_majority)
        sent_level_preds.append(pred_majority)

    # Token-level
    acc_token = accuracy_score(all_labels, all_preds)
    print(f"\nToken-level accuracy: {acc_token * 100:.2f}%\n")
    print("Token-level Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Sentence-level
    acc_sent = accuracy_score(sent_level_labels, sent_level_preds)
    print(f"\nSentence-level accuracy: {acc_sent * 100:.2f}%\n")
    print("Sentence-level Classification Report:")
    print(classification_report(sent_level_labels, sent_level_preds, digits=4))

    return acc_token, acc_sent


if __name__ == "__main__":
    X_FILE = os.path.join(DATA_DIR, "x_test_mixed_sentences_n50.txt")
    Y_FILE = os.path.join(DATA_DIR, "y_test_mixed_sentences_n50.txt")

    evaluate_dataset(X_FILE, Y_FILE)