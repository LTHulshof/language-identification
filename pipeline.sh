#!/bin/bash
cd src

# Filter dataset
python filtering.py

# Preprocess dataset
python preprocessing.py

# Finetune Roberta
python train.py

cd ..