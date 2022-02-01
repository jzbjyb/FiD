#!/usr/bin/env bash

model=trained_reader/sciq_reader_base_v11lm/checkpoint/latest
data=open_domain_data/SciQ/test.json
ckpt_dir=${model}.sciq_test

MAX_NUM_GPU_PER_NODE=8
num_gpu=$1

if (( ${num_gpu} == 1 )); then
  echo 'single-GPU'
  prefix=""
elif (( ${num_gpu} <= ${MAX_NUM_GPU_PER_NODE} )); then
  echo 'single-node'
  export NGPU=${num_gpu}
  prefix="-m torch.distributed.launch --nproc_per_node=${num_gpu}"
else
  echo 'multi-node'
  prefix=""
  exit  # TODO: not implemented
fi

python ${prefix} test_reader.py \
  --model_path ${model} \
  --eval_data ${data} \
  --per_gpu_batch_size 16 \
  --n_context 100 \
  --name distill \
  --checkpoint_dir ${ckpt_dir} \
  --write_crossattention_scores
