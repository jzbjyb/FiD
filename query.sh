#!/usr/bin/env bash

export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

num_gpu=1

index_short_name=$1
model_path=$2/checkpoint/latest
head_idx=$3
token_topk=$4
use_position_bias=$5
per_gpu_batch_size=128
query_maxlength=50

if [[ ${index_short_name} == 'nq_test_top10' ]]; then
  queries=open_domain_data/NQ/test.json
elif [[ ${index_short_name} == 'msmarcoqa_dev' ]]; then
  queries=open_domain_data/msmarco_qa/dev.json
elif [[ ${index_short_name} == 'bioasq_500k_test' ]]; then
  queries=open_domain_data/bioasq_500k.nosummary/test.json
elif [[ ${index_short_name} == 'fiqa' ]]; then
  queries=open_domain_data/fiqa/test.json
elif [[ ${index_short_name} == 'cqadupstack_mathematica' ]]; then
  queries=open_domain_data/cqadupstack/mathematica/test.json
elif [[ ${index_short_name} == 'cqadupstack_physics' ]]; then
  queries=open_domain_data/cqadupstack/physics/test.json
elif [[ ${index_short_name} == 'cqadupstack_programmers' ]]; then
  queries=open_domain_data/cqadupstack/programmers/test.json
else
  exit
fi

use_faiss_gpu="--use_faiss_gpu"
if (( ${token_topk} > 2048 )); then
  use_faiss_gpu=""
fi

if [[ ${use_position_bias} == 'true' ]]; then
  passages=${model_path}.index/${index_short_name}.position/embedding_*.npz
  output_path=${model_path}.index/${index_short_name}.position
  indexing_dimension=$((64 + ${query_maxlength}))
  extra="--use_position_bias"
else
  passages=${model_path}.index/${index_short_name}/embedding_*.npz
  output_path=${model_path}.index/${index_short_name}
  indexing_dimension=64
  extra=""
fi

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
  --model_type fid \
  --queries ${queries} \
  --passages ${passages} \
  --model_path ${model_path} \
  --output_path ${output_path} \
  --per_gpu_batch_size ${per_gpu_batch_size} \
  --indexing_dimension ${indexing_dimension} \
  --query_maxlength ${query_maxlength} \
  --hnsw_m 0 \
  --token_topk ${token_topk} \
  --doc_topk 10 \
  --head_idx ${head_idx} \
  --save_or_load_index \
  ${use_faiss_gpu} \
  ${extra}
