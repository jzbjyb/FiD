#!/usr/bin/env bash

model_root=trained_retriever
name=scifact_reader_base_v11lm_raw
#data=open_domain_data/NQ/test.json
data=open_domain_data/scifact/test.json
#psgs_tsv_file=open_domain_data/NQ/train.1000/withtest.psgs_w100.tsv
psgs_tsv_file=open_domain_data/scifact/psgs.tsv

python passage_retrieval.py \
    --model_path ${model_root}/${name}/checkpoint/latest \
    --passages ${psgs_tsv_file} \
    --data ${data} \
    --passages_embeddings ${psgs_tsv_file}.embedding/${name}_00 \
    --output_path ${psgs_tsv_file}.embedding/${name}.retrieve/test.json \
    --n-docs 100 \
    --no_fp16
