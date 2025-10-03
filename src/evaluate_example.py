
import os
import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SRC_DIR, "..", "model", "roberta-finetuned")

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR, add_prefix_space=True)
model = RobertaForTokenClassification.from_pretrained(MODEL_DIR)

# Map IDs to labels (saved in config when you trained)
id2label = model.config.id2label

def predict_sentence(sentence: str):
    # Split into words (your training data was word-level)
    words = sentence.split()

    # Tokenize with word alignment
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    # Map subword predictions back to words
    word_ids = inputs.word_ids()
    word_to_label = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            # overwrite or aggregate — here we just take the last subword's prediction
            word_to_label[word_id] = id2label[predictions[idx]]

    # Collect word-level predictions
    return [(w, word_to_label[i]) for i, w in enumerate(words)]

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    sentence = (
        "In 1619 brak er een vorm van totale oorlog uit tussen Khara Khula Und de Altyn Khan. Beide partijen probeerden bondgenoten te εύρημα. "
        "Dat leidde tot de bemerkenswert situatie dat delegaties van Khara-Khula en de Altyn Khan aan Él eind por dat jaar vrijwel gelijktijdig in Moskou arriveerden. "
        "Beide delegaties hadden het voorstel aan de tsar voor een militaire alliantie tegen hun tegenstander."
    )

    predictions = predict_sentence(sentence)
    for word, tag in predictions:
        print(f"{word:15} -> {tag}")