"""
Script for comparing label files against the human ground truth (gpt data analysis).
Prints token-level accuracy and classification reports for each.
"""

import os
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "data", "wili-preprocessed"))

GROUND_TRUTH = os.path.join(DATA_DIR, "y_test_gpt_human.txt")
PRED_FILES = {
    "Gemini": os.path.join(DATA_DIR, "y_test_gpt_gemini.txt"),
    "FastText": os.path.join(DATA_DIR, "y_test_gpt_fasttext.txt"),
    "GoogleTranslate": os.path.join(DATA_DIR, "y_test_gpt_googletranslate.txt"),
}


def load_labels(path):
    """
    Loads label sequences per line.
    """
    with open(path, encoding="utf-8") as f:
        return [line.strip().split() for line in f if line.strip()]


def evaluate_labels(true_file, pred_file, name):
    """
    Evaluate predicted labels against ground truth.
    """
    true_labels = load_labels(true_file)
    pred_labels = load_labels(pred_file)

    if len(true_labels) != len(pred_labels):
        print(f"{name}: line count mismatch (truth={len(true_labels)}, pred={len(pred_labels)})")
        return

    all_true, all_pred = [], []
    mismatched_lines = 0

    for t, p in tqdm(zip(true_labels, pred_labels), total=len(true_labels), desc=f"Evaluating {name}", ncols=100):
        min_len = min(len(t), len(p))
        if len(t) != len(p):
            mismatched_lines += 1
        all_true.extend(t[:min_len])
        all_pred.extend(p[:min_len])

    acc = accuracy_score(all_true, all_pred)
    print(f"\n{name} Token-level Accuracy: {acc * 100:.2f}%")
    print(f"{mismatched_lines} lines had different lengths.\n")
    print(classification_report(all_true, all_pred, digits=4))
    print("=" * 100 + "\n")


def main():
    if not os.path.exists(GROUND_TRUTH):
        print("Ground truth file not found")
        return

    for name, path in PRED_FILES.items():
        if os.path.exists(path):
            evaluate_labels(GROUND_TRUTH, path, name)
        else:
            print(f"File not found at {path}")


if __name__ == "__main__":
    main()