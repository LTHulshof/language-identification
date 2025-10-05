"""
Script for finetuning a roberta model on the preprocessed wili-2018 dataset
"""

import os
from datasets import Dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForTokenClassification,
    TrainingArguments,
    Trainer,
)
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from utils import *


# Data paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, "..", "data", "wili-preprocessed")
MODEL_DIR = os.path.join(SRC_DIR, "..", "model", "roberta-finetuned")

train_texts_path = os.path.join(DATA_DIR, "x_train_n50.txt")
train_labels_path = os.path.join(DATA_DIR, "y_train_n50.txt")
eval_texts_path = os.path.join(DATA_DIR, "x_eval_n50.txt")
eval_labels_path = os.path.join(DATA_DIR, "y_eval_n50.txt")


# Load raw data
train_texts, train_labels = load_data(train_texts_path, train_labels_path)
eval_texts, eval_labels = load_data(eval_texts_path, eval_labels_path)

print(f"Loaded {len(train_texts)} training examples, {len(eval_texts)} eval examples")


# Label mapping
label_list = ["deu", "ell", "eng", "nld", "spa", "unk"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}


# Convert labels from strings to IDs
train_labels = [[label2id[l] for l in seq] for seq in train_labels]
eval_labels = [[label2id[l] for l in seq] for seq in eval_labels]


# Create Hugging Face datasets and define Tokenizer
train_dataset = Dataset.from_dict({"tokens": train_texts, "labels": train_labels})
eval_dataset = Dataset.from_dict({"tokens": eval_texts, "labels": eval_labels})

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

# Pass tokenizer to util function using lambda function
train_dataset = train_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
eval_dataset = eval_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

# Remove original word-level tokens after mapping
train_dataset = train_dataset.remove_columns(["tokens"])
eval_dataset = eval_dataset.remove_columns(["tokens"])



# Load pretrained Roberta Model
model = RobertaForTokenClassification.from_pretrained(
    "roberta-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

# Training
args = TrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(MODEL_DIR)