#!/usr/bin/env bash

export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

#train_file=pretrained_models/nq_reader_base.nq_train_10000/distill/dataset_wscores.json
#dev_file=open_domain_data/NQ/dev.json
train_file=trained_reader/sciq_reader_base_v11lm/checkpoint/latest.sciq_train/distill/dataset_wscores.json
dev_file=open_domain_data/SciQ/dev.json

ckpt_dir=trained_retriever
name=$2  # t5_base_v11lm_sciq_step2k_noscale, sciq_reader_base_v11lm_sciq_step2k_noscale
init_with=$3  # google/t5-base-lm-adapt, trained_reader/sciq_reader_base_v11lm/checkpoint/latest
n_context=90

MAX_NUM_GPU_PER_NODE=8
num_gpu=$1

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

python ${prefix} train_retriever.py \
  --lr 0.00005 \
  --optim adamw \
  --scheduler linear \
  --weight_decay 0.01 \
  --train_data ${train_file} \
  --eval_data ${dev_file} \
  --n_context ${n_context} \
  --total_steps 2001 \
  --scheduler_steps 2001 \
  --save_freq 500 \
  --warmup_steps 200 \
  --eval_freq 50 \
  --eval_num_examples 500 \
  --per_gpu_batch_size 1 \
  --accumulation_steps 1 \
  --checkpoint_dir ${ckpt_dir} \
  --name ${name} \
  --init_with ${init_with} \
  --indexing_dimension 768 \
  --no_projection \
  --wandb_name ${ckpt_dir}/${name}
