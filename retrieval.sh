#!/usr/bin/env bash
#SBATCH --job-name=nq
#SBATCH --time=6:00:00
#SBATCH --partition=side
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --mem=1024GB

# env
eval "$(conda shell.bash hook)"
conda activate raat

# utils
source utils.sh

export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

gpu=$(get_gpu_type)

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

srun python retrieval.py \
  --model_type ${model_type} \
  --model_path ${model_path} \
  --passages ${passages} \
  --output_path ${output_path} \
  --save_every_n_doc ${save_every_n_doc} \
  --num_workers ${num_workers} \
  --per_gpu_batch_size ${passage_per_gpu_batch_size} \
  --passage_maxlength ${passage_maxlength} \
  --query_maxlength ${query_maxlength} \
  ${head_idx} ${extra} ${max_over_head}
