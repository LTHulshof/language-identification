"""
Evaluate all fine-tuned RoBERTa/XLM-RoBERTa models on the same test set.
Print token- and sentence-level metrics for each.
"""

import os
import torch
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForTokenClassification,
    RobertaTokenizerFast,
    RobertaForTokenClassification,
)
from tqdm import tqdm


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "model"))
DATA_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "data", "wili-preprocessed"))

# Test files
X_FILE = os.path.join(DATA_DIR, "x_test_mixed_sentences_n50.txt")
Y_FILE = os.path.join(DATA_DIR, "y_test_mixed_sentences_n50.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir):
    """
    Load correct model/tokenizer type based on directory name.
    """
    if "xlm-roberta" in model_dir.lower():
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_dir)
        model = XLMRobertaForTokenClassification.from_pretrained(model_dir)
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
        model = RobertaForTokenClassification.from_pretrained(model_dir)

    model.to(device)
    model.eval()
    return tokenizer, model


def predict_tokens(words, tokenizer, model, id2label):
    """
    Predicts token-level labels for a list of words.
    """
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True).to(device)
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


def evaluate_dataset(model_dir, x_path, y_path):
    """
    Evaluate dataset and print metrics
    """
    tokenizer, model = load_model(model_dir)
    id2label = model.config.id2label

    with open(x_path, encoding="utf-8") as f:
        x_lines = [line.strip() for line in f if line.strip()]
    with open(y_path, encoding="utf-8") as f:
        y_lines = [line.strip() for line in f if line.strip()]


    all_preds, all_labels = [], []
    sent_level_preds, sent_level_labels = [], []

    for sent, labels in tqdm(zip(x_lines, y_lines), total=len(x_lines), desc="Processing", ncols=100):
        words = sent.split()
        gold_labels = labels.split()
        preds = predict_tokens(words, tokenizer, model, id2label)

        # Truncation handling
        min_len = min(len(preds), len(gold_labels))
        preds, gold_labels = preds[:min_len], gold_labels[:min_len]

        # Token-level
        all_preds.extend(preds)
        all_labels.extend(gold_labels)

        # Sentence-level majority
        gold_majority = Counter(gold_labels).most_common(1)[0][0]
        pred_majority = Counter(preds).most_common(1)[0][0]
        sent_level_labels.append(gold_majority)
        sent_level_preds.append(pred_majority)

    # Token-level metrics
    acc_token = accuracy_score(all_labels, all_preds)
    print(f"\nToken-level accuracy: {acc_token * 100:.2f}%")
    print("Token-level Metrics:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Sentence-level metrics
    acc_sent = accuracy_score(sent_level_labels, sent_level_preds)
    print(f"\nSentence-level accuracy: {acc_sent * 100:.2f}%")
    print("Sentence-level Metrics:")
    print(classification_report(sent_level_labels, sent_level_preds, digits=4))
    print("\n\n")


if __name__ == "__main__":
    # List of models to evaluate
    model_dirs = [
        "xlm-roberta-easy",
        "xlm-roberta-medium",
        "xlm-roberta-hard",
        # "roberta-easy",
        # "roberta-medium",
        # "roberta-hard",
    ]

    for m in model_dirs:
        model_path = os.path.join(MODEL_ROOT, m)
        evaluate_dataset(model_path, X_FILE, Y_FILE)