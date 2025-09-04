# Language Identification with Deep Learning

This project explores **automatic language identification** using deep learning techniques, developed as part of the *Deep Learning for Natural Language Processing* course at the University of Amsterdam (UvA).

The goal is to build a model that can correctly classify the language of a given text sample.

## Features

- Preprocessing of multilingual text data
- Blablabla
- Evaluation on multiple languages with NLP metrics

## Dataset

This project uses the [WiLI-2018 dataset](https://zenodo.org/records/841984), a publicly available dataset for **written language identification** covering 235 languages.

### Option 1: Quick Setup (recommended)
Run the downloader script to fetch and extract the dataset automatically:

```bash
python download_data.py
```

This will place the files inside the `data/raw/` directory.

### Option 2: Manual Setup
1. Download the dataset from [Zenodo](https://zenodo.org/records/841984).
2. Extract the archive.
3. Place the files inside the `data/raw/` directory of this project.

## Project Structure

```text
├── data/                 # Dataset (raw and processed)
│   ├── raw/              # Original data
│   └── processed/        # Cleaned/preprocessed data
│
├── src/                  # Source code
│   ├── preprocessing.py  # Text cleaning, tokenization, etc.
│   ├── models.py         # Model architectures
│   ├── train.py          # Training loop
│   ├── evaluate.py       # Evaluation scripts
│   └── utils.py          # Helper functions
│
├── notebooks/            # Jupyter notebooks for exploration & experiments
│
├── results/              # Saved models, logs, evaluation outputs
│
├── main.py               # Main execute file
├── download_data.py      # Download dataset
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```
