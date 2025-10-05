#!/bin/bash
cd src

# REM Download dataset
python download_data.py

# Filter dataset
python filtering.py

# Preprocess dataset
python preprocessing.py

# Finetune Roberta
python train.py

cd ..