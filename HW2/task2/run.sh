#!/bin/bash

# 设置 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 定义模型和数据集数组
models=("roberta-base" "bert-base-uncased" "allenai/scibert_scivocab_uncased")
datasets=("restaurant_sup" "acl_sup" "agnews_sup")

# 循环遍历每个模型、每个数据集，并重复运行 5 次
for model in "${models[@]}"
do
  for dataset in "${datasets[@]}"
  do
    for run in {1..5}
    do
      echo "Running experiment with model: $model on dataset: $dataset - Repeat $run"
      
      # 执行 Python 脚本并保存输出到对应目录
      python train.py \
        --model_name_or_path $model \
        --output_dir ./results/${model}_${dataset}_run$run \
        --do_train \
        --do_eval \
        --evaluation_strategy epoch \
        --logging_strategy steps \
        --logging_steps 10 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --num_train_epochs 3 \
        --seed 42 \
        --dataset $dataset \
        --report_to wandb
    done
  done
done
