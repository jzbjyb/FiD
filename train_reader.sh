#!/usr/bin/env bash

export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

#train_data=open_domain_data/NQ/train.json
#eval_data=open_domain_data/NQ/dev.json
#train_data=open_domain_data/scifact/train.json
#eval_data=open_domain_data/scifact/test.json
train_data=open_domain_data/SciQ/train.json
eval_data=open_domain_data/SciQ/dev.json

ckpt_dir=trained_reader
name=sciq_reader_base_v11lm

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

python ${prefix} train_reader.py \
  --train_data ${train_data} \
  --eval_data ${eval_data} \
  --model_size google/t5-base-lm-adapt \
  --use_checkpoint \
  --text_maxlength 250 \
  --per_gpu_batch_size 1 \
  --accumulation_steps 1 \
  --n_context 100 \
  --name ${name} \
  --checkpoint_dir ${ckpt_dir} \
  --lr 0.00005 \
  --optim adamw \
  --scheduler linear \
  --weight_decay 0.01 \
  --total_step 1001 \
  --warmup_step 100 \
  --save_freq 500 \
  --eval_freq 30 \
  --wandb_name ${ckpt_dir}/${name}
