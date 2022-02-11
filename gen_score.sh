#!/usr/bin/env bash

model=trained_reader/nq_reader_base_v11lm_queryside/checkpoint/latest
#data=open_domain_data/SciQ/test.json
#data=open_domain_data/quasar_s/dev.json
data=open_domain_data/NQ/test.json
ckpt_dir=${model}.allhead.nq_test

MAX_NUM_GPU_PER_NODE=8
num_gpu=$1
attention_mask=query-side

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
  --per_gpu_batch_size 12 \
  --n_context 100 \
  --text_maxlength 250 \
  --answer_maxlength 50 \
  --attention_mask ${attention_mask} \
  --name distill \
  --checkpoint_dir ${ckpt_dir} \
  --write_crossattention_scores \
  --write_results
