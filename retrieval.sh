#!/usr/bin/env bash
source utils.sh

export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

gpu=a100

model_type=$1  # fid dpr colbert

# default arguments
head_idx=""
extra=""
max_over_head=""

# model specific arguments
if [[ ${model_type} == 'fid' ]]; then
  model_path=$2/checkpoint/latest
  index_name=$3
  head_idx="--head_idx $4"
  use_position_bias=$5
  use_max_over_head=$6

  output_path=${model_path}.index/${index_name}
  if [[ ${use_position_bias} == 'true' ]]; then
    output_path=${output_path}.position
    extra="--use_position_bias"
  fi
  if [[ ${use_max_over_head} == 'true' ]]; then
    max_over_head="--max_over_head"
  fi
  get_dataset_settings ${index_name} 1024 ${gpu}  # t5's limit is 1024

elif [[ ${model_type} == 'dpr' ]]; then
  model_path=facebook/dpr-ctx_encoder-multiset-base
  index_name=$2

  output_path=pretrained_models/dpr.index/${index_name}
  get_dataset_settings ${index_name} 512 ${gpu}  # bert's limit is 512

elif [[ ${model_type} == 'colbert' ]]; then
  model_name=$2
  if [[ ${model_name} == 'ms' ]]; then
    model_path=../ColBERT/downloads/colbertv2.0
  elif [[ ${model_name} == 'nq' ]]; then
    model_path=../ColBERT/downloads/colbert-60000.dnn
  fi
  index_name=$3
  
  output_path=${model_path}.index/${index_name}
  get_dataset_settings ${index_name} 512 ${gpu}  # bert's limit is 512

else
  exit 1
fi

mkdir -p ${output_path}
for shard_id in $(seq 0 $((${num_shards} - 1))); do
  if (( ${num_shards} == 1 )); then
    python retrieval.py \
      --model_type ${model_type} \
      --model_path ${model_path} \
      --passages ${passages} \
      --output_path ${output_path} \
      --shard_id ${shard_id} \
      --num_shards ${num_shards} \
      --save_every_n_doc ${save_every_n_doc} \
      --num_workers ${num_workers} \
      --per_gpu_batch_size ${passage_per_gpu_batch_size} \
      --passage_maxlength ${passage_maxlength} \
      --query_maxlength ${query_maxlength} \
      ${head_idx} ${extra} ${max_over_head}
  else
    CUDA_VISIBLE_DEVICES=${shard_id} python retrieval.py \
      --model_type ${model_type} \
      --model_path ${model_path} \
      --passages ${passages} \
      --output_path ${output_path} \
      --shard_id ${shard_id} \
      --num_shards ${num_shards} \
      --save_every_n_doc ${save_every_n_doc} \
      --num_workers ${num_workers} \
      --per_gpu_batch_size ${passage_per_gpu_batch_size} \
      --passage_maxlength ${passage_maxlength} \
      --query_maxlength ${query_maxlength} \
      ${head_idx} ${extra} ${max_over_head} &
  fi
done
wait
