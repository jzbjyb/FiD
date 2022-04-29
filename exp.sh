#!/usr/bin/env bash
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:TITANX:1
#SBATCH --nodelist=tir-0-19
#SBATCH --time=0

model=$1
head=$2

for data in bioasq_500k_test cqadupstack_programmers msmarcoqa_dev fiqa cqadupstack_mathematica cqadupstack_physics; do
  echo ${data} ${model}
  ./retrieval.sh ${data} ${model} ${head} false
  ./query.sh ${data} ${model} ${head} 100 false
  ./query.sh ${data} ${model} ${head} 1000 false
  ./query.sh ${data} ${model} ${head} 2048 false
  ./query.sh ${data} ${model} ${head} 4096 false
done
