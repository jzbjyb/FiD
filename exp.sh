#!/usr/bin/env bash
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0

in_datasets=(nq_test_top10)
out_datasets=(bioasq_500k_test cqadupstack_programmers msmarcoqa_dev fiqa cqadupstack_mathematica cqadupstack_physics)
topks=(100 1000 2048 4096 8192 16384)

domain=$1
if [[ ${domain} == 'in' ]]; then
  datasets=("${in_datasets[@]}")
elif [[ ${domain} == 'out' ]]; then
  datasets=("${out_datasets[@]}")
else
  exit 1
fi

model_type=$2
if [[ ${model_type} == 'fid' ]]; then
  echo FiD
  model=$3
  head=$4
  position=false
elif [[ ${model_type} == 'dpr' ]]; then
  echo DPR
elif [[ ${model_type} == 'colbert' ]]; then
  echo ColBERT
  model=$3
else
  exit 1
fi

for data in "${datasets[@]}"; do
  echo ${data} ${model}
  ./retrieval.sh ${model_type} ${model} ${data} ${head} ${position}
  for topk in "${topks[@]}"; do
    ./query.sh ${model_type} ${model} ${data} ${head} ${position} ${topk}
  done
done
