REM Bat file that trains RoBERTa and XML-RoBERTa on different datasets.

cd src

python preprocessing.py

python train.py --train_texts "x_train_orig.txt"  --train_labels "y_train_orig.txt" --eval_texts "x_eval_orig.txt" --eval_labels "y_eval_orig.txt" --model_type "xlm-roberta" --output_dir "xlm-roberta-easy"
python train.py --train_texts "x_train_n50.txt"  --train_labels "y_train_n50.txt" --eval_texts "x_eval_n50.txt" --eval_labels "y_eval_n50.txt" --model_type "xlm-roberta" --output_dir "xlm-roberta-medium"
python train.py --train_texts "x_train_n50_s100.txt"  --train_labels "y_train_n50_s100.txt" --eval_texts "x_eval_n50_s100.txt" --eval_labels "y_eval_n50_s100.txt" --model_type "xlm-roberta" --output_dir "xlm-roberta-hard"

python train.py --train_texts "x_train_orig.txt"  --train_labels "y_train_orig.txt" --eval_texts "x_eval_orig.txt" --eval_labels "y_eval_orig.txt" --model_type "roberta" --output_dir "roberta-easy"
python train.py --train_texts "x_train_n50.txt"  --train_labels "y_train_n50.txt" --eval_texts "x_eval_n50.txt" --eval_labels "y_eval_n50.txt" --model_type "roberta" --output_dir "roberta-medium"
python train.py --train_texts "x_train_n50_s100.txt"  --train_labels "y_train_n50_s100.txt" --eval_texts "x_eval_n50_s100.txt" --eval_labels "y_eval_n50_s100.txt" --model_type "roberta" --output_dir "roberta-hard"

cd ..