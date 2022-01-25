#!/usr/bin/env bash

model_root=trained_retriever
name=bert_step2w
psgs_tsv_file=open_domain_data/NQ/train.1000/withtest.psgs_w100.tsv

python generate_passage_embeddings.py \
  --model_path ${model_root}/${name}/checkpoint/latest \
  --passages ${psgs_tsv_file} \
  --output_path ${psgs_tsv_file}.embedding/${name} \
  --shard_id 0 \
  --num_shards 1 \
  --per_gpu_batch_size 500
