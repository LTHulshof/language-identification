"""
Script for Finetuning RoBERTa or XLM-RoBERTa on the preprocessed WiLI-2018 dataset.
"""

import sys
import os
import argparse
from datasets import Dataset
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForTokenClassification,
    RobertaTokenizerFast,
    RobertaForTokenClassification,
    TrainingArguments,
    Trainer,
)
import torch
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa/XLM-RoBERTa on WiLI-2018.")
    parser.add_argument("--train_texts", type=str, required=True, help="Path to training texts file")
    parser.add_argument("--train_labels", type=str, required=True, help="Path to training labels file")
    parser.add_argument("--eval_texts", type=str, required=True, help="Path to eval texts file")
    parser.add_argument("--eval_labels", type=str, required=True, help="Path to eval labels file")
    parser.add_argument("--model_type", type=str, choices=["roberta", "xlm-roberta"], default="xlm-roberta", help="Model type")
    parser.add_argument("--output_dir", type=str, default="roberta-finetuned", help="Directory to save the fine-tuned model")
    return parser.parse_args()




def main():


    
    args = parse_args()
    
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_NAME = "xlm-roberta-base" if args.model_type == "xlm-roberta" else "roberta-base"

    MAIN_MODEL_DIR = os.path.join(SRC_DIR, "..", "model")
    MODEL_DIR = os.path.join(MAIN_MODEL_DIR, args.output_dir)
    os.makedirs(MODEL_DIR, exist_ok=True)  # create the directory if it doesn't exist

    DATA_DIR = os.path.join(SRC_DIR, "..", "data", "wili-preprocessed")

    train_texts_path = os.path.join(DATA_DIR, args.train_texts)
    train_labels_path = os.path.join(DATA_DIR, args.train_labels)
    eval_texts_path = os.path.join(DATA_DIR, args.eval_texts)
    eval_labels_path = os.path.join(DATA_DIR, args.eval_labels)


    # Redirect all console output to logs.txt in MODEL_DIR
    log_file_path = os.path.join(MODEL_DIR, "logs.txt")
    log_file = open(log_file_path, "w")

    class LoggerWriter:
        def __init__(self, original, log_file):
            self.original = original
            self.log_file = log_file

        def write(self, message):
            if message.strip() != "":
                if not message.endswith("\n"):
                    message += "\n"
                self.original.write(message)
                self.original.flush()
                self.log_file.write(message)
                self.log_file.flush()

        def flush(self):
            self.original.flush()
            self.log_file.flush()

    sys.stdout = LoggerWriter(sys.stdout, log_file)
    sys.stderr = LoggerWriter(sys.stderr, log_file)


    # Load data
    train_texts, train_labels = load_data(train_texts_path, train_labels_path)
    eval_texts, eval_labels = load_data(eval_texts_path, eval_labels_path)
    print(f"Loaded {len(train_texts)} training examples, {len(eval_texts)} eval examples")

    # Label mapping
    label_list = ["deu", "ell", "eng", "nld", "spa", "unk"]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    # Convert labels
    train_labels = [[label2id[l] for l in seq] for seq in train_labels]
    eval_labels = [[label2id[l] for l in seq] for seq in eval_labels]

    train_dataset = Dataset.from_dict({"tokens": train_texts, "labels": train_labels})
    eval_dataset = Dataset.from_dict({"tokens": eval_texts, "labels": eval_labels})

    # Choose tokenizer
    if args.model_type == "xlm-roberta":
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(MODEL_NAME)
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    # Tokenize and align labels
    train_dataset = train_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    eval_dataset = eval_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    # Remove original token lists
    train_dataset = train_dataset.remove_columns(["tokens"])
    eval_dataset = eval_dataset.remove_columns(["tokens"])

    # Load pretrained model
    if args.model_type == "xlm-roberta":
        model = XLMRobertaForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
        )
    else:
        model = RobertaForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
        )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Training setup
    args_train = TrainingArguments(
        output_dir=MODEL_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        load_best_model_at_end=True,   
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=50,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=1,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train and save
    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()