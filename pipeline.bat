cd src

REM Download dataset
python download_data.py

REM Filter dataset
python filtering.py

REM Preprocess dataset
python preprocessing.py

REM Finetune XLM-Roberta
python train.py --train_texts "x_train_n50_s100.txt"  --train_labels "y_train_n50_s100.txt" --eval_texts "x_eval_n50_s100.txt" --eval_labels "y_eval_n50_s100.txt" --model_type "xlm-roberta" --output_dir "xlm-roberta-hard"

cd ..