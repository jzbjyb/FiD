#!/usr/bin/env bash

export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

num_gpu=1
model_path=facebook/dpr-ctx_encoder-multiset-base
index_short_name=bioasq_500k_test

if [[ ${index_short_name} == 'nq_test_top10' ]]; then
  passages=open_domain_data/NQ/psgs_w100.test_top10_aggregate.tsv
  passage_maxlength=200
  per_gpu_batch_size=512
elif [[ ${index_short_name} == 'msmarcoqa_dev' ]]; then
  passages=open_domain_data/msmarco_qa/psgs_w100.dev_aggregate.tsv
  passage_maxlength=200
  per_gpu_batch_size=512
elif [[ ${index_short_name} == 'bioasq_500k_test' ]]; then
  passages=open_domain_data/bioasq_500k.nosummary/psgs_w100.test_aggregate.tsv
  passage_maxlength=512
  per_gpu_batch_size=256
else
  exit
fi

output_path=pretrained_models/dpr.index/${index_short_name}

shard_id=0
num_shards=1

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
  --model_type dpr \
  --model_path ${model_path} \
  --passages ${passages} \
  --output_path ${output_path} \
  --shard_id ${shard_id} \
  --num_shards ${num_shards} \
  --per_gpu_batch_size ${per_gpu_batch_size} \
  --passage_maxlength ${passage_maxlength}
