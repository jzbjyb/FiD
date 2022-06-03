#!/usr/bin/env bash
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0

set -e

in_datasets=(nq_test_top10)
out_datasets=(bioasq_500k_test cqadupstack_programmers msmarcoqa_dev fiqa cqadupstack_mathematica cqadupstack_physics)
all_datasets=("${in_datasets[@]}" "${out_datasets[@]}")
declare -a gpu_topks=(100 1000 2048)
declare -a cpu_topks=(4096 8192 16384)
declare -a reranks=(0 1000)

domain=$1
if [[ ${domain} == 'in' ]]; then
  datasets=("${in_datasets[@]}")
elif [[ ${domain} == 'out' ]]; then
  datasets=("${out_datasets[@]}")
elif [[ ${domain} == 'all' ]]; then
  datasets=("${all_datasets[@]}")
else
  exit 1
fi

topk_range=$2
if [[ ${topk_range} == 'gpu' ]]; then
  declare -a two_topks=(gpu_topks)
elif [[ ${topk_range} == 'cpu' ]]; then
  declare -a two_topks=(cpu_topks)
elif [[ ${topk_range} == 'all' ]]; then
  declare -a two_topks=(gpu_topks cpu_topks)
else
  exit 1
fi

model_type=$3
if [[ ${model_type} == 'fid' ]]; then
  echo FiD
  model=$4
  head=$5
  position=false
  use_max_over_head=false
elif [[ ${model_type} == 'dpr' ]]; then
  echo DPR
elif [[ ${model_type} == 'colbert' ]]; then
  echo ColBERT
  model=$4
else
  exit 1
fi

for group in "${two_topks[@]}"; do
  declare -n topks=$group
  for data in "${datasets[@]}"; do
    echo ${data} ${model}
    if [[ $group == 'gpu_topks' ]]; then
      ./retrieval.sh ${model_type} ${model} ${data} ${head} ${position} ${use_max_over_head}
    fi
    for topk in "${topks[@]}"; do
      for rerank in "${reranks[@]}"; do
        ./query.sh ${model_type} ${model} ${data} ${head} ${position} ${topk} ${rerank} ${use_max_over_head}  # TODO: DPR does not have rerank
      done
    done
  done
done
