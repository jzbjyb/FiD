#!/usr/bin/env bash

python train_reader.py \
  --train_data open_domain_data/NQ/train.json \
  --eval_data open_domain_data/NQ/dev.json \
  --model_size base \
  --per_gpu_batch_size 1 \
  --n_context 100 \
  --name my_experiment \
  --checkpoint_dir checkpoint
