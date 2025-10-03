#!/bin/bash

# REM Download dataset
python download_data.py

cd src

# Filter dataset
python filtering.py

# Preprocess dataset
python preprocessing.py

# Finetune Roberta
python train.py

cd ..