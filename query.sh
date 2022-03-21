#!/usr/bin/env bash

export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

num_gpu=1
queries=open_domain_data/NQ/test.json
model_path=pretrained_models/nq_reader_base
passages=${model_path}.index/nq/embedding_*.npz
output_path=${model_path}.index/nq
per_gpu_batch_size=128

if (( ${num_gpu} == 1 )); then
  echo 'single-GPU'
  prefix=""
elif (( ${num_gpu} <= ${MAX_NUM_GPU_PER_NODE} )); then
  echo 'single-node'
  export NGPU=${num_gpu}
  random_port=$(shuf -i 10000-65000 -n 1)
  prefix="-m torch.distributed.launch --nproc_per_node=${num_gpu} --master_port=${random_port}"
else
  echo 'multi-node'
  prefix=""
  exit  # TODO: not implemented
fi

python ${prefix} retrieval.py \
  --queries ${queries} \
  --passages ${passages} \
  --model_path ${model_path} \
  --output_path ${output_path} \
  --per_gpu_batch_size ${per_gpu_batch_size} \
  --query_maxlength 50 \
  --hnsw_m 0 \
  --topk 100 \
  --save_or_load_index
