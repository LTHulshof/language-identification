# Language Identification with Deep Learning

This project explores **automatic language identification** using deep learning techniques, developed as part of the *Deep Learning for Natural Language Processing* course at the University of Amsterdam (UvA).

The goal is to build a model that can correctly classify the language of a given text sample.

## Features

- Preprocessing of multilingual text data
- Finetuning a XLM-RoBERTa model for language classification on the preprocessed data
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
├── data/                           # Dataset (raw and processed)
│   ├── wili-2018/                  # Original data
│   └── wili_preprocessed/          # Cleaned/preprocessed data
│
├── src/                            # Source code
│   ├── preprocessing.py            # Text cleaning, tokenization, etc.
│   ├── models.py                   # Model architectures
│   ├── train.py                    # Training loop
│   ├── evaluate.py                 # Evaluation scripts for testing model
│   ├── evaluate_labels.py          # Evaluation script for comparing two label files.
│   └── utils.py                    # Helper functions for training
│   └── download_data.py            # Download and unzip the WiLi-2018 data
│   └── mix_datapoint_sentence.py   # Mix sentences from a preprocessed dataset 
│
├── model/                        # Saved models, logs, evaluation outputs
│
├── pipeline.bat/pipeline.sh        # Main execute file, train a single XLM roberta model
├── download_data.py                # Download dataset
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```
