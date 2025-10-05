"""
Script for generating mixed-language datapoints by mixing existing x and y datapoints.
"""

import os
import random
from collections import Counter, defaultdict
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "data", "wili-preprocessed"))
X_FILE = os.path.join(DATA_DIR, "x_test_n50.txt")
Y_FILE = os.path.join(DATA_DIR, "y_test_n50.txt")

OUTPUT_PREFIX = "x_test_mixed_sentences_n50"


N_OUTPUT = 2000             # Number of mixed datapoints to generate
MIN_SENTENCES = 2           # Minimum number of source sentences per mixed datapoint
MAX_SENTENCES = 4           # Maximum number of source sentences per mixed datapoint

NO_REUSE = True            # Do not reuse the same source sentence inside one mixed datapoint
ALLOW_DUP_LANG = True      # Allow multiple sentences of the same language in one mixed datapoint

RANDOM_SEED = 42           # Seed for reproducibility


def split_into_sentences(x_line, y_line):
    """
    Split a line into full sentences using ".".
    Return a list of dicts: {"tokens": [...], "labels": [...], "lang": majority_lang}
    """
    x_tokens = x_line.split()
    y_labels = y_line.split()
    assert len(x_tokens) == len(y_labels), "Token/label length mismatch"

    # Simple sentence splitting by "."
    sentence_indices = [i for i, tok in enumerate(x_tokens) if tok.endswith(".")]
    sentence_indices = [-1] + sentence_indices  # start index for first sentence

    sentences = []
    for start, end in zip(sentence_indices[:-1], sentence_indices[1:]):
        s_tokens = x_tokens[start+1:end+1]
        s_labels = y_labels[start+1:end+1]
        if s_tokens:
            maj_label = Counter(s_labels).most_common(1)[0][0]
            sentences.append({"tokens": s_tokens, "labels": s_labels, "lang": maj_label})
    return sentences

def load_xy_by_sentence(x_path, y_path):
    with open(x_path, encoding="utf-8") as fx, open(y_path, encoding="utf-8") as fy:
        x_lines = [l.strip() for l in fx if l.strip()]
        y_lines = [l.strip() for l in fy if l.strip()]
    assert len(x_lines) == len(y_lines), "x and y must have same number of non-empty lines"

    all_sentences = []
    for x, y in zip(x_lines, y_lines):
        sents = split_into_sentences(x, y)
        all_sentences.extend(sents)
    return all_sentences

def build_index_by_lang(sentences):
    by_lang = defaultdict(list)
    for i, s in enumerate(sentences):
        by_lang[s["lang"]].append(i)
    return by_lang

def construct_mixed_sentences(
    sentences,
    by_lang,
    n_output,
    min_sentences,
    max_sentences,
    allow_reuse_sentences=True,
    require_distinct_langs=True
):
    available_langs = list(by_lang.keys())
    if not available_langs:
        raise ValueError("No languages found in input data.")

    mixed = []
    for _ in tqdm(range(n_output), desc="Constructing mixed datapoints", ncols=100):
        n_sents = random.randint(min_sentences, max_sentences)

        if require_distinct_langs and len(available_langs) >= n_sents:
            chosen_langs = random.sample(available_langs, n_sents)
        else:
            chosen_langs = [random.choice(available_langs) for _ in range(n_sents)]

        dp_tokens = []
        dp_labels = []
        used_indices = set()

        for lang in chosen_langs:
            candidates = by_lang.get(lang, [])
            if not candidates:
                candidates = list(range(len(sentences)))
            possible = [c for c in candidates if (allow_reuse_sentences or c not in used_indices)]
            if not possible:
                possible = candidates
            sid = random.choice(possible)
            if not allow_reuse_sentences:
                used_indices.add(sid)
            dp_tokens.extend(sentences[sid]["tokens"])
            dp_labels.extend(sentences[sid]["labels"])

        mixed.append((dp_tokens, dp_labels))
    return mixed



def save_mixed(mixed, x_out, y_out):
    with open(x_out, "w", encoding="utf-8") as fx, open(y_out, "w", encoding="utf-8") as fy:
        for tokens, labels in tqdm(mixed, desc="Writing mixed files", ncols=100):
            fx.write(" ".join(tokens) + "\n")
            fy.write(" ".join(labels) + "\n")



def main():
    random.seed(RANDOM_SEED)

    x_path = os.path.abspath(X_FILE)
    y_path = os.path.abspath(Y_FILE)
    assert os.path.exists(x_path) and os.path.exists(y_path), "Input files must exist"

    sentences = load_xy_by_sentence(x_path, y_path)
    print(f"Loaded {len(sentences)} sentences from {x_path} and {y_path}")

    by_lang = build_index_by_lang(sentences)
    print("Languages found and counts:", {k: len(v) for k, v in by_lang.items()})

    mixed = construct_mixed_sentences(
        sentences,
        by_lang,
        n_output=N_OUTPUT,
        min_sentences=MIN_SENTENCES,
        max_sentences=MAX_SENTENCES,
        allow_reuse_sentences=not NO_REUSE,
        require_distinct_langs=not ALLOW_DUP_LANG,
    )

    out_dir = os.path.dirname(x_path)
    x_out = os.path.join(out_dir, f"{OUTPUT_PREFIX}.txt")
    y_out = os.path.join(out_dir, f"{OUTPUT_PREFIX.replace('x_', 'y_')}.txt")

    save_mixed(mixed, x_out, y_out)
    print(f"\nSaved {len(mixed)} mixed datapoints to:\n  {x_out}\n  {y_out}")


if __name__ == "__main__":
    main()