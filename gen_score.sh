#!/usr/bin/env bash
source utils.sh

num_gpu=$1  # use all gpus
model=$2/checkpoint/latest  # pretrained_models/nq_reader_base
index_name=$3
ckpt_dir=${model}.rerank/${index_name}
other="${@:4}"

n_context=100
attention_mask=separate

get_dataset_settings ${index_name} 1024 a100  # t5's limit is 1024
get_prefix ${num_gpu}

python ${prefix} test_reader.py \
  --model_path ${model} \
  --eval_data ${queries} \
  --per_gpu_batch_size ${fid_per_gpu_batch_size} \
  --n_context ${n_context} \
  --text_maxlength ${text_maxlength} \
  --answer_maxlength ${answer_maxlength} \
  --attention_mask ${attention_mask} \
  --name test \
  --checkpoint_dir ${ckpt_dir} \
  --write_crossattention_scores \
  --write_results \
  ${other}
