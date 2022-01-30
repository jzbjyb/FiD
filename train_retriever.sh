#!/usr/bin/env bash

train_file=pretrained_models/nq_reader_base.nq_train_10000/distill/dataset_wscores.json
dev_file=open_domain_data/NQ/dev.json

ckpt_dir=trained_retriever
name=scifact_reader_base_v11lm_raw
init_with=trained_reader/scifact_reader_base_v11lm/checkpoint/latest

python train_retriever.py \
  --lr 0.00005 \
  --optim adamw \
  --scheduler linear \
  --weight_decay 0.01 \
  --train_data ${train_file} \
  --eval_data ${dev_file} \
  --n_context 100 \
  --total_steps 0 \
  --scheduler_steps 30000 \
  --save_freq 2000 \
  --warmup_steps 1000 \
  --eval_freq 2000 \
  --accumulation_steps 8 \
  --checkpoint_dir ${ckpt_dir} \
  --name ${name} \
  --init_with ${init_with} \
  --indexing_dimension 768 \
  --no_projection  # TODO: debug
