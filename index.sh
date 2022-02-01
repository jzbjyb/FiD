#!/usr/bin/env bash

model_root=trained_retriever
name=sciq_reader_base_v11lm_sciq_step2k_noscale
#psgs_tsv_file=open_domain_data/NQ/train.1000/withtest.psgs_w100.tsv
#psgs_tsv_file=open_domain_data/scifact/psgs.tsv
psgs_tsv_file=open_domain_data/SciQ/psgs.tsv

python generate_passage_embeddings.py \
  --model_path ${model_root}/${name}/checkpoint/latest \
  --passages ${psgs_tsv_file} \
  --output_path ${psgs_tsv_file}.embedding/${name} \
  --shard_id 0 \
  --num_shards 1 \
  --per_gpu_batch_size 500 \
  --no_fp16
