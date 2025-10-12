"""
Script for preprocessing the wili-2018 data by translating nouns and word-spans to a listed non-dominant language.
"""

import os
import spacy
from deep_translator import GoogleTranslator
import random
import re
from langdetect import detect, DetectorFactory
from tqdm import tqdm
import pickle

DetectorFactory.seed = 0

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, "..", "data")
WILI_SUBSET_DIR = os.path.join(DATA_DIR, "wili-preprocessed")
os.makedirs(WILI_SUBSET_DIR, exist_ok=True)


TARGET_LANGS = ["eng", "spa", "nld", "ell", "deu"]

wili_to_google = {
    "eng": "en",
    "nld": "nl",
    "ell": "el",
    "spa": "es",
    "deu": "de"
}
google_to_wili = {v: k for k, v in wili_to_google.items()}

# Load spaCy models once, disable unnecessary pipes
spacy_models = {
    "eng": spacy.load("en_core_web_sm", disable=["ner", "parser"]),
    "spa": spacy.load("es_core_news_sm", disable=["ner", "parser"]),
    "nld": spacy.load("nl_core_news_sm", disable=["ner", "parser"]),
    "ell": spacy.load("el_core_news_sm", disable=["ner", "parser"]),
    "deu": spacy.load("de_core_news_sm", disable=["ner", "parser"])
}

_detection_cache = {}

# Persistent translation cache
try:
    with open("translation_cache.pkl", "rb") as f:
        _translation_cache = pickle.load(f)
except FileNotFoundError:
    _translation_cache = {}



def get_pos_tags(tokens, language):
    """
    Word-by-word POS tagging to avoid mapping issues.
    """
    model = spacy_models[language]
    pos_tags = []
    for token in tokens:
        doc = model(token)
        # take the first token's POS if spaCy splits it further
        pos_tags.append(doc[0].pos_ if len(doc) > 0 else "X")
    return pos_tags



def sanitize_token(token, expected_wili):
    """
    For each token, return its token, or "unk" for numerical and unrecognized languages.
    """
    if re.search(r"\d", token):
        return "unk"

    core = "".join(ch for ch in token if ch.isalpha())
    if not core:
        return "unk"

    # never mark <=2 character tokens as unk
    core_lower = core.lower()
    if len(core_lower) <= 2:
        return expected_wili

    # check cache for token
    if core_lower in _detection_cache:
        detected = _detection_cache[core_lower]
    else:
        try:
            detected = detect(core_lower)
        except Exception:
            detected = None
        _detection_cache[core_lower] = detected

    expected_google = wili_to_google.get(expected_wili)
    if detected in google_to_wili:
        return expected_wili if detected == expected_google else "unk"
    return expected_wili



def batch_translate(tokens, source_lang, target_lang, use_cache=True):
    """
    Translate a list of tokens in a single GoogleTranslator call
    """
    key = (tuple(tokens), source_lang, target_lang)

    if use_cache and key in _translation_cache:
        return _translation_cache[key]

    try:
        sentence = " ".join(tokens)
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(sentence)
        translated_tokens = translated.split()
        # only cache if not empty
        if translated_tokens:  
            _translation_cache[key] = translated_tokens
    except Exception:
        translated_tokens = tokens

    return translated_tokens



def save_translation_cache():
    with open("translation_cache.pkl", "wb") as f:
        pickle.dump(_translation_cache, f)



def process_and_translate(x_file, y_file, out_prefix, replace_nouns=True, replace_spans=True, replace_noun_freq=0.2, replace_span_freq=0.3):
    """
    Replace nouns and word-spans for every datapoint, write to a new file that contains all partially translated datapoints.
    """
    # build suffix string
    suffix_parts = []
    if replace_nouns:
        suffix_parts.append(f"n{int(replace_noun_freq*100)}")
    if replace_spans:
        suffix_parts.append(f"s{int(replace_span_freq*100)}")
    if not suffix_parts:
        suffix_parts.append("orig")
    suffix = "_".join(suffix_parts)

    # construct output file names
    x_out = os.path.join(WILI_SUBSET_DIR, f"{out_prefix}_{suffix}.txt")
    y_out = os.path.join(WILI_SUBSET_DIR, f"{out_prefix.replace('x_', 'y_')}_{suffix}.txt")

    with open(x_file, "r", encoding="utf-8") as f_x, open(y_file, "r", encoding="utf-8") as f_y:
        x_data = f_x.readlines()
        y_data = f_y.readlines()

    with open(x_out, "w", encoding="utf-8") as f_x_out, open(y_out, "w", encoding="utf-8") as f_y_out:
        for x, y in tqdm(zip(x_data, y_data), total=len(x_data), desc=f"Processing {x_file}"):
            x_split = x.strip().split()
            expected_wili = y.strip()
            new_y = [sanitize_token(tok, expected_wili) for tok in x_split]

            pos_tags = get_pos_tags(x_split, expected_wili)

            # Replace nouns
            if replace_nouns:
                noun_indices = [i for i, pos in enumerate(pos_tags) if pos == "NOUN"]
                if noun_indices:
                    k = max(1, int(len(noun_indices) * replace_noun_freq))
                    indices = random.sample(noun_indices, min(k, len(noun_indices)))
                    for i in indices:
                        if new_y[i] == "unk":
                            continue
                        # For each noun, choose a one of the non-dominant languages and translate.
                        source_lang = wili_to_google[expected_wili]
                        target_lang = random.choice([l for l in wili_to_google.values() if l != source_lang])
                        translated_tokens = batch_translate([x_split[i]], source_lang, target_lang, use_cache=False)
                        
                        # Skip if translation failed
                        if not translated_tokens:
                            continue  
                        
                        # If multiple words are returned by translator, add the token for the translated words multiple times
                        target_wili = google_to_wili[target_lang]
                        x_split[i:i+1] = translated_tokens
                        new_y[i:i+1] = [target_wili] * len(translated_tokens)

            # Replace spans
            if replace_spans and len(x_split) > 5 and random.random() < replace_span_freq:
                span_len = random.choice([2, 3, 4, 5])
                start = random.randint(0, len(x_split) - span_len)
                end = start + span_len
                source_lang = wili_to_google[expected_wili]
                target_lang = random.choice([l for l in wili_to_google.values() if l != source_lang])
                translated_tokens = batch_translate(x_split[start:end], source_lang, target_lang)
                x_split[start:end] = translated_tokens[:span_len]
                new_y[start:end] = [google_to_wili[target_lang]] * span_len

            if len(x_split) != len(new_y):
                min_len = min(len(x_split), len(new_y))
                x_split = x_split[:min_len]
                new_y = new_y[:min_len]

            f_x_out.write(" ".join(x_split) + "\n")
            f_y_out.write(" ".join(new_y) + "\n")

    save_translation_cache()
    print(f"Saved {x_out} and {y_out}")



# # Training data preprocessing- easy
# process_and_translate(
#     os.path.join(WILI_SUBSET_DIR, "x_train.txt"),
#     os.path.join(WILI_SUBSET_DIR, "y_train.txt"),
#     out_prefix="x_train",
#     replace_nouns=False,
#     replace_spans=False,
#     replace_noun_freq=0.0,
#     replace_span_freq=0.0
# )


# # Evaluation data preprocessing - easy
# process_and_translate(
#     os.path.join(WILI_SUBSET_DIR, "x_eval.txt"),
#     os.path.join(WILI_SUBSET_DIR, "y_eval.txt"),
#     out_prefix="x_eval",
#     replace_nouns=False,
#     replace_spans=False,
#     replace_noun_freq=0.0,
#     replace_span_freq=0.0
# )


# # Training data preprocessing - medium
# process_and_translate(
#     os.path.join(WILI_SUBSET_DIR, "x_train.txt"),
#     os.path.join(WILI_SUBSET_DIR, "y_train.txt"),
#     out_prefix="x_train",
#     replace_nouns=True,
#     replace_spans=False,
#     replace_noun_freq=0.5,
#     replace_span_freq=0.0
# )

# # Evaluation data preprocessing - medium
# process_and_translate(
#     os.path.join(WILI_SUBSET_DIR, "x_eval.txt"),
#     os.path.join(WILI_SUBSET_DIR, "y_eval.txt"),
#     out_prefix="x_eval",
#     replace_nouns=True,
#     replace_spans=False,
#     replace_noun_freq=0.5,
#     replace_span_freq=0.0
# )


# # Training data preprocessing - hard
# process_and_translate(
#     os.path.join(WILI_SUBSET_DIR, "x_train.txt"),
#     os.path.join(WILI_SUBSET_DIR, "y_train.txt"),
#     out_prefix="x_train",
#     replace_nouns=True,
#     replace_spans=True,
#     replace_noun_freq=0.5,
#     replace_span_freq=1
# )

# # Evaluation data preprocessing  - hard
# process_and_translate(
#     os.path.join(WILI_SUBSET_DIR, "x_eval.txt"),
#     os.path.join(WILI_SUBSET_DIR, "y_eval.txt"),
#     out_prefix="x_eval",
#     replace_nouns=True,
#     replace_spans=True,
#     replace_noun_freq=0.5,
#     replace_span_freq=1
# )


# # Default testing data preprocessing
# process_and_translate(
#     os.path.join(WILI_SUBSET_DIR, "x_test.txt"),
#     os.path.join(WILI_SUBSET_DIR, "y_test.txt"),
#     out_prefix="x_test",
#     replace_nouns=True,
#     replace_spans=True,
#     replace_noun_freq=0.5,
#     replace_span_freq=0.5
# )

# Additional data generation to check translation span performance.
for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    process_and_translate(
        os.path.join(WILI_SUBSET_DIR, "x_test.txt"),
        os.path.join(WILI_SUBSET_DIR, "y_test.txt"),
        out_prefix="x_test",
        replace_nouns=False,
        replace_spans=True,
        replace_noun_freq=i,
        replace_span_freq=i
    )