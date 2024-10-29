@echo off
python train.py --model_name_or_path roberta-base --output_dir ./results --do_train --do_eval --evaluation_strategy epoch --logging_strategy steps --logging_steps 10 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --num_train_epochs 3 --seed 42 --dataset "restaurant_sup"
