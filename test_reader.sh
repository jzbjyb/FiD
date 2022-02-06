#!/usr/bin/env bash

model=trained_reader/sciq_reader_base_v11lm_separate_layer9/checkpoint/latest
data=open_domain_data/SciQ/test.json

python test_reader.py \
  --model_path ${model} \
  --eval_data ${data} \
  --per_gpu_batch_size 1 \
  --n_context 100 \
  --name my_test \
  --checkpoint_dir checkpoint \
  --write_results
