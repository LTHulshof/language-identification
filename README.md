# Language Identification with Deep Learning

This project explores **automatic language identification** using deep learning techniques, developed as part of the *Deep Learning for Natural Language Processing* course at the University of Amsterdam (UvA).

The goal is to build a model that can correctly classify the language of a given text sample.

## Features

- Preprocessing of multilingual text data
- Blablabla
- Evaluation on multiple languages with NLP metrics

## Project Structure

```text
├── data/                 # Dataset (raw and processed)
│   ├── raw/              # Original data
│   └── processed/        # Cleaned/preprocessed data
│
├── src/                  # Source code
│   ├── preprocessing.py  # Text cleaning, tokenization, etc.
│   ├── models.py         # Model architectures (RNN, Transformer, etc.)
│   ├── train.py          # Training loop
│   ├── evaluate.py       # Evaluation scripts
│   └── utils.py          # Helper functions
│
├── notebooks/            # Jupyter notebooks for exploration & experiments
│
├── results/              # Saved models, logs, evaluation outputs
│
├── download_data.py      # Download dataset
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
