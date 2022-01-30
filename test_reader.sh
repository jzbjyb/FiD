#!/usr/bin/env bash

model_dir=trained_reader/scifact_reader_base_v11lm/checkpoint/latest
train_data=open_domain_data/scifact/train.json
eval_data=open_domain_data/scifact/test.json

python test_reader.py \
  --model_path ${model_dir} \
  --eval_data ${eval_data} \
  --per_gpu_batch_size 1 \
  --n_context 100 \
  --name my_test \
  --checkpoint_dir checkpoint \
  --write_results