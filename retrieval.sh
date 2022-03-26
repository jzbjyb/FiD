#!/usr/bin/env bash

export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

num_gpu=1
index_short_name=$1

#trained_reader/t5_base_v11lm
#trained_reader/nq_reader_base_v11lm_separate_layer6_continue
#trained_reader/nq_reader_base_v11lm_separate_layer6_continue_kl1_tau0001
#trained_reader/nq_reader_base_v11lm_separate_layer6_continue_decoder50_decattnnorm_tau0001
model_path=$2/checkpoint/latest
head_idx=$3

if [[ ${index_short_name} == 'nq_test_top10' ]]; then
  passages=open_domain_data/NQ/psgs_w100.test_top10_aggregate.tsv
  passage_maxlength=200
  per_gpu_batch_size=128
elif [[ ${index_short_name} == 'msmarcoqa_dev' ]]; then
  passages=open_domain_data/msmarco_qa/psgs_w100.dev_aggregate.tsv
  passage_maxlength=200
  per_gpu_batch_size=128
elif [[ ${index_short_name} == 'bioasq_500k_test' ]]; then
  passages=open_domain_data/bioasq_500k.nosummary/psgs_w100.test_aggregate.tsv
  passage_maxlength=1024  # TODO use 512?
  per_gpu_batch_size=32
elif [[ ${index_short_name} == 'fiqa' ]]; then
  passages=open_domain_data/fiqa/psgs.tsv
  passage_maxlength=512
  per_gpu_batch_size=64
elif [[ ${index_short_name} == 'cqadupstack_mathematica' ]]; then
  passages=open_domain_data/cqadupstack/mathematica/psgs.tsv
  passage_maxlength=512
  per_gpu_batch_size=64
elif [[ ${index_short_name} == 'cqadupstack_physics' ]]; then
  passages=open_domain_data/cqadupstack/physics/psgs.tsv
  passage_maxlength=512
  per_gpu_batch_size=64
elif [[ ${index_short_name} == 'cqadupstack_programmers' ]]; then
  passages=open_domain_data/cqadupstack/programmers/psgs.tsv
  passage_maxlength=512
  per_gpu_batch_size=64
else
  exit
fi

output_path=${model_path}.index/${index_short_name}

shard_id=0
num_shards=1

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
  --model_type fid \
  --model_path ${model_path} \
  --passages ${passages} \
  --output_path ${output_path} \
  --shard_id ${shard_id} \
  --num_shards ${num_shards} \
  --per_gpu_batch_size ${per_gpu_batch_size} \
  --passage_maxlength ${passage_maxlength} \
  --head_idx ${head_idx}
