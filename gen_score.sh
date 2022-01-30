#!/usr/bin/env bash

model=trained_reader/scifact_reader_base
ckpt_dir=${model}.scifact_train
data=open_domain_data/scifact/train.json

python test_reader.py \
  --model_path ${model} \
  --eval_data ${data} \
  --per_gpu_batch_size 16 \
  --n_context 100 \
  --name distill \
  --checkpoint_dir ${ckpt_dir} \
  --write_crossattention_scores
