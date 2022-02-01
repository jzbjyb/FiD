#!/usr/bin/env bash

model_root=trained_retriever
name=sciq_reader_base_v11lm_sciq_step2k_noscale
#data=open_domain_data/NQ/test.json
#psgs_tsv_file=open_domain_data/NQ/train.1000/withtest.psgs_w100.tsv
#data=open_domain_data/scifact/test.json
#psgs_tsv_file=open_domain_data/scifact/psgs.tsv
data=open_domain_data/SciQ/test.json
psgs_tsv_file=open_domain_data/SciQ/psgs.tsv

python passage_retrieval.py \
    --model_path ${model_root}/${name}/checkpoint/latest \
    --passages ${psgs_tsv_file} \
    --data ${data} \
    --passages_embeddings ${psgs_tsv_file}.embedding/${name}_00 \
    --output_path ${psgs_tsv_file}.embedding/${name}.retrieve/test.json \
    --n-docs 100 \
    --no_fp16
