#!/usr/bin/env bash

#train_data=open_domain_data/NQ/train.json
#eval_data=open_domain_data/NQ/dev.json
train_data=open_domain_data/scifact/train.json
eval_data=open_domain_data/scifact/test.json

python train_reader.py \
  --train_data ${train_data} \
  --eval_data ${eval_data} \
  --model_size base \
  --use_checkpoint \
  --text_maxlength 250 \
  --per_gpu_batch_size 1 \
  --accumulation_steps 1 \
  --n_context 100 \
  --name scifact_reader_base_v11lm \
  --checkpoint_dir trained_reader \
  --lr 0.0005 \
  --optim adamw \
  --scheduler linear \
  --weight_decay 0.01 \
  --total_step 401 \
  --warmup_step 40 \
  --save_freq 400 \
  --eval_freq 500
