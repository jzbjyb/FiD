#!/usr/bin/env bash
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0

set -e

in_datasets=(nq_test_top10)
nq_datasets=(nq)
pro=(cqadupstack_programmers)
out_datasets=(bioasq_500k_test cqadupstack_programmers msmarcoqa_dev fiqa cqadupstack_mathematica cqadupstack_physics)
all_datasets=("${in_datasets[@]}" "${out_datasets[@]}")
cqa_datasets=(cqadupstack_programmers cqadupstack_mathematica cqadupstack_physics cqadupstack_android cqadupstack_english cqadupstack_gaming cqadupstack_gis cqadupstack_stats cqadupstack_tex cqadupstack_unix cqadupstack_webmasters cqadupstack_wordpress)
cqa_bad_datasets=(cqadupstack_mathematica cqadupstack_android cqadupstack_gis cqadupstack_stats cqadupstack_tex cqadupstack_webmasters cqadupstack_wordpress)
cqa_good_datasets=(cqadupstack_programmers cqadupstack_physics cqadupstack_english cqadupstack_gaming cqadupstack_unix)
declare -a gpu_topks=(100 1000 2048)
declare -a cpu_topks=(4096 8192 16384)
declare -a reranks=(0 1000 2048)

domain=$1
if [[ ${domain} == 'in' ]]; then
  datasets=("${in_datasets[@]}")
elif [[ ${domain} == 'nq' ]]; then
  datasets=("${nq_datasets[@]}")
elif [[ ${domain} == 'out' ]]; then
  datasets=("${out_datasets[@]}")
elif [[ ${domain} == 'all' ]]; then
  datasets=("${all_datasets[@]}")
elif [[ ${domain} == 'cqa_bad' ]]; then
  datasets=("${cqa_bad_datasets[@]}")
elif [[ ${domain} == 'cqa_good' ]]; then
  datasets=("${cqa_good_datasets[@]}")  
else
  datasets=(${domain})
fi

topk_range=$2
if [[ ${topk_range} == 'gpu' ]]; then
  declare -a two_topks=(gpu_topks)
elif [[ ${topk_range} == 'cpu' ]]; then
  declare -a two_topks=(cpu_topks)
elif [[ ${topk_range} == 'all' ]]; then
  declare -a two_topks=(gpu_topks cpu_topks)
elif [[ ${topk_range} == 'best' ]]; then
  declare -a gpu_topks=(2048)
  declare -a two_topks=(gpu_topks)
  declare -a reranks=(2048)
else
  exit 1
fi

aug=""
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
  aug=true
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
        if (( ${rerank} <= ${topk} )); then
          ./query.sh ${model_type} ${model} ${data} ${head} ${position} ${topk} ${rerank} ${use_max_over_head} ${aug}  # TODO: DPR does not have rerank
        fi
      done
    done
  done
done
