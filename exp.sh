#!/usr/bin/env bash
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:TITANX:1
#SBATCH --nodelist=tir-0-19
#SBATCH --time=0

for data in bioasq_500k_test cqadupstack_programmers; do
  for model in trained_reader/nq_reader_base_v11lm_separate_layer6_continue_kl1_tau0001_bs64_step8k_inbatchneg32; do
    echo ${data} ${model}
    ./retrieval.sh ${data} ${model} 3 false
    ./query.sh ${data} ${model} 3 100 false
    ./query.sh ${data} ${model} 3 1000 false
  done
done

for data in bioasq_500k_test cqadupstack_programmers; do
  for model in trained_reader/nq_reader_large_v11lm_separate_layer12_bs16_step3k_continue_kl1_step8k; do
    echo ${data} ${model}
    ./retrieval.sh ${data} ${model} 6 false
    ./query.sh ${data} ${model} 6 100 false
    ./query.sh ${data} ${model} 6 1000 false
  done
done
