#!/usr/bin/env bash

export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

num_gpu=1
passages=open_domain_data2/NQ/psgs_w100.test_aggregate.tsv
model_path=trained_reader2/t5_base_v11lm/checkpoint/latest
output_path=${model_path}.index/nq_test
shard_id=0
num_shards=1
per_gpu_batch_size=32

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
  --model_path ${model_path} \
  --passages ${passages} \
  --output_path ${output_path} \
  --shard_id ${shard_id} \
  --num_shards ${num_shards} \
  --per_gpu_batch_size ${per_gpu_batch_size} \
  --passage_maxlength 200 \
  --indexing_dimension 64 \
