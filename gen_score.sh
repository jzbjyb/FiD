#!/usr/bin/env bash
#SBATCH --job-name=qa
#SBATCH --time=2:00:00
#SBATCH --partition=side
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=1024GB

# env
eval "$(conda shell.bash hook)"
conda activate raat

# utils
source utils.sh

# bioasq_1m uses 2 as per_gpu_batch_size, 800 as text_maxlength, and join_sep=', ' in EM evaluation

num_gpu=$1  # use all gpus
model=$2/checkpoint/latest  # pretrained_models/nq_reader_base
index_name=$3
ckpt_dir=${model}.rerank/specific
queries_another=$4
other="${@:5}"

n_context=100
attention_mask=separate

get_dataset_settings ${index_name} 1024 a100  # t5's limit is 1024
get_prefix ${num_gpu}

python ${prefix} test_reader.py \
  --model_path ${model} \
  --eval_data ${queries_another} \
  --per_gpu_batch_size ${fid_per_gpu_batch_size} \
  --n_context ${n_context} \
  --text_maxlength ${text_maxlength} \
  --answer_maxlength ${answer_maxlength} \
  --attention_mask ${attention_mask} \
  --name test \
  --checkpoint_dir ${ckpt_dir} \
  --write_results \
  --write_crossattention_scores \
  ${other}
