#!/usr/bin/env bash

MAX_NUM_GPU_PER_NODE=8
num_gpu=$1
model=$2/checkpoint/latest  # pretrained_models/nq_reader_base
dataname=$3
n_context=$4
ckpt_dir=${model}.rerank/${dataname}
other="${@:5}"

if [[ ${dataname} == 'nq' ]]; then
  data=open_domain_data/NQ/test.json
  text_maxlength=250
  per_gpu_batch_size=8
elif [[ ${dataname} == 'bioasq' ]]; then
  data=open_domain_data/bioasq_500k.nosummary/test.json
  text_maxlength=1024
  per_gpu_batch_size=1
elif [[ ${dataname} == 'msmarcoqa' ]]; then
  data=open_domain_data/msmarco_qa/dev.json
  text_maxlength=250
  per_gpu_batch_size=8
else
  exit
fi

#data=open_domain_data/NQ/test.json
#ckpt_dir=${model}.allhead_softmax.nq_test
#text_maxlength=250
#per_gpu_batch_size=8
#n_context=100

#data=open_domain_data/SciQ/test.json
#ckpt_dir=${model}.allhead_softmax.sciq_test
#text_maxlength=250
#per_gpu_batch_size=12
#n_context=100

#data=open_domain_data/quasar_s/dev.json
#ckpt_dir=${model}.allhead_softmax.quasars_dev
#text_maxlength=250
#per_gpu_batch_size=12
#n_context=100

#data=open_domain_data/bioasq_500k.nosummary/test.json
#ckpt_dir=${model}.allhead_softmax.bioasq_test
#text_maxlength=1024
#per_gpu_batch_size=1
#n_context=100

#data=open_domain_data/msmarco_qa/dev.json
#ckpt_dir=${model}.allhead_softmax.msmarcoqa_dev
#text_maxlength=250
#per_gpu_batch_size=12
#n_context=10

attention_mask=separate
query_in_decoder=no

if [[ ${query_in_decoder} == 'no' ]]; then
  answer_maxlength=50
else
  answer_maxlength=100
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

python ${prefix} test_reader.py \
  --model_path ${model} \
  --eval_data ${data} \
  --per_gpu_batch_size ${per_gpu_batch_size} \
  --n_context ${n_context} \
  --text_maxlength ${text_maxlength} \
  --answer_maxlength ${answer_maxlength} \
  --attention_mask ${attention_mask} \
  --query_in_decoder ${query_in_decoder} \
  --name distill \
  --checkpoint_dir ${ckpt_dir} \
  --write_crossattention_scores \
  --write_results \
  ${other}
