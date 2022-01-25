#!/usr/bin/env bash

model=pretrained_models/nq_reader_base
ckpt_dir=${model}.nq_train_10000
data=open_domain_data/NQ/train.10000.json

python test_reader.py \
  --model_path ${model} \
  --eval_data ${data} \
  --per_gpu_batch_size 16 \
  --n_context 100 \
  --name distill \
  --checkpoint_dir ${ckpt_dir} \
  --write_crossattention_scores
