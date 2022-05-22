#!/usr/bin/env bash
source utils.sh

export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

MAX_NUM_GPU_PER_NODE=4
gpu=a100
num_gpu=1

model_type=$1  # fid dpr colbert

# default arguments
head_idx=""
extra=""
token_topk=0
doc_topk=10
candidate_doc_topk=0
faiss_gpus="--faiss_gpus all"
rerank=""
max_over_head=""
files_per_run=16  # about 70G embs

# model specific arguments
if [[ ${model_type} == 'fid' ]]; then
  model_path=$2/checkpoint/latest
  index_name=$3
  head_idx="--head_idx $4"
  use_position_bias=$5
  token_topk=$6
  candidate_doc_topk=$7
  use_max_over_head=$8
  index_dim=64

  get_dataset_settings ${index_name} 1024 ${gpu}  # t5's limit is 1024
  output_path=${model_path}.index/${index_name}
  if [[ ${use_position_bias} == 'true' ]]; then
    index_dim=$(( 64 + ${query_maxlength} ))
    output_path=${output_path}.position
    extra="--use_position_bias"
  fi
  if (( ${token_topk} > 2048 )); then
    faiss_gpus="--faiss_gpus -1"
  fi
  if (( ${candidate_doc_topk} > 0 )); then
    rerank="--candidate_doc_topk ${candidate_doc_topk}"
  fi
  if [[ ${use_max_over_head} == 'true' ]]; then
    max_over_head="--max_over_head"
  fi

elif [[ ${model_type} == 'dpr' ]]; then
  model_path=facebook/dpr-question_encoder-multiset-base
  index_name=$2
  index_dim=768
  
  get_dataset_settings ${index_name} 512 ${gpu}  # bert's limit is 512
  output_path=pretrained_models/dpr.index/${index_name}
  if (( ${doc_topk} > 2048 )); then
    faiss_gpus="--faiss_gpus -1"
  fi
  
elif [[ ${model_type} == 'colbert' ]]; then
  model_name=$2
  if [[ ${model_name} == 'ms' ]]; then
    model_path=../ColBERT/downloads/colbertv2.0
  elif [[ ${model_name} == 'nq' ]]; then
    model_path=../ColBERT/downloads/colbert-60000.dnn
  else
    exit 1
  fi
  index_name=$3
  token_topk=$4
  candidate_doc_topk=$5
  index_dim=128

  get_dataset_settings ${index_name} 512 ${gpu}  # bert's limit is 512
  output_path=${model_path}.index/${index_name}
  if (( ${token_topk} > 2048 )); then
    faiss_gpus="--faiss_gpus -1"
  fi
  if (( ${candidate_doc_topk} > 0 )); then
    rerank="--candidate_doc_topk ${candidate_doc_topk}"
  fi
  
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
  --queries ${queries} \
  --model_path ${model_path} \
  --output_path ${output_path} \
  --per_gpu_batch_size ${query_per_gpu_batch_size} \
  --indexing_dimension ${index_dim} \
  --query_maxlength ${query_maxlength} \
  --hnsw_m 0 \
  --token_topk ${token_topk} \
  --doc_topk ${doc_topk} \
  --save_or_load_index \
  --files_per_run ${files_per_run} \
  ${rerank} ${head_idx} ${faiss_gpus} ${extra} ${max_over_head}
