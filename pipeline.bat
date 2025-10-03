cd src

REM Filter dataset
python filtering.py

REM Preprocess dataset
python preprocessing.py

REM Finetune Roberta
python train.py

cd ..