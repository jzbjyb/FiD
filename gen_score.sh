#!/usr/bin/env bash

model=trained_reader/nq_reader_base_v11lm_separate_layer6_continue/checkpoint/latest
#model=pretrained_models/nq_reader_base
#data=open_domain_data/NQ/test.json
#ckpt_dir=${model}.allhead_softmax.nq_test
#data=open_domain_data/SciQ/test.json
#ckpt_dir=${model}.allhead_softmax.sciq_test
#data=open_domain_data/quasar_s/dev.json
#ckpt_dir=${model}.allhead_softmax.quasars_dev
#data=open_domain_data/bioasq_500k.nosummary/test.json
#ckpt_dir=${model}.allhead_softmax.bioasq_test
data=open_domain_data/msmarco_qa/dev.1000.json
ckpt_dir=${model}.allhead_softmax.msmarcoqa_dev

n_context=10
MAX_NUM_GPU_PER_NODE=8
num_gpu=$1
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
  prefix="-m torch.distributed.launch --nproc_per_node=${num_gpu}"
else
  echo 'multi-node'
  prefix=""
  exit  # TODO: not implemented
fi

python ${prefix} test_reader.py \
  --model_path ${model} \
  --eval_data ${data} \
  --per_gpu_batch_size 12 \
  --n_context ${n_context} \
  --text_maxlength 250 \
  --answer_maxlength ${answer_maxlength} \
  --attention_mask ${attention_mask} \
  --query_in_decoder ${query_in_decoder} \
  --name distill \
  --checkpoint_dir ${ckpt_dir} \
  --write_crossattention_scores \
  --write_results
