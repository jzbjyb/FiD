#!/usr/bin/env bash
source utils.sh

export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

MAX_NUM_GPU_PER_NODE=8
gpu=a100
num_gpu=1
shard_id=0
num_shards=1

model_type=$1  # fid dpr colbert

# default arguments
head_idx=""
extra=""

# model specific arguments
if [[ ${model_type} == 'fid' ]]; then
  model_path=$2/checkpoint/latest
  index_name=$3
  head_idx="--head_idx $4"
  use_position_bias=$5

  output_path=${model_path}.index/${index_name}
  if [[ ${use_position_bias} == 'true' ]]; then
    output_path=${output_path}.position
    extra="--use_position_bias"
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
  --model_type ${model_type} \
  --model_path ${model_path} \
  --passages ${passages} \
  --output_path ${output_path} \
  --shard_id ${shard_id} \
  --num_shards ${num_shards} \
  --per_gpu_batch_size ${passage_per_gpu_batch_size} \
  --passage_maxlength ${passage_maxlength} \
  --query_maxlength ${query_maxlength} \
  ${head_idx} ${extra}
