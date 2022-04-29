#!/usr/bin/env bash

index_short_name=$1
model_path=$2
topk=$3
pkl=${model_path}.index/${index_short_name}/qid2rank_${topk}.pkl
json=${model_path}.index/${index_short_name}/qid2rank_${topk}.json

split=test
if [[ ${index_short_name} == 'nq_test_top10' ]]; then
  queries=open_domain_data/NQ/test.json
  passages=open_domain_data/NQ/psgs_w100.test_top10_aggregate.tsv
elif [[ ${index_short_name} == 'msmarcoqa_dev' ]]; then
  queries=open_domain_data/msmarco_qa/dev.json
  passages=open_domain_data/msmarco_qa/psgs.dev_aggregate.tsv
  beir=open_domain_data/msmarco_qa_beir
  split=dev
elif [[ ${index_short_name} == 'bioasq_500k_test' ]]; then
  queries=open_domain_data/bioasq_500k.nosummary/test.json
  passages=open_domain_data/bioasq_500k.nosummary/psgs.test_aggregate.tsv
  beir=open_domain_data/bioasq_500k.nosummary_beir
elif [[ ${index_short_name} == 'fiqa' ]]; then
  queries=open_domain_data/fiqa/test.json
  passages=open_domain_data/fiqa/psgs.tsv
  beir=open_domain_data/fiqa_beir
elif [[ ${index_short_name} == 'cqadupstack_mathematica' ]]; then
  queries=open_domain_data/cqadupstack/mathematica/test.json
  passages=open_domain_data/cqadupstack/mathematica/psgs.tsv
  beir=open_domain_data/cqadupstack_beir/mathematica
elif [[ ${index_short_name} == 'cqadupstack_physics' ]]; then
  queries=open_domain_data/cqadupstack/physics/test.json
  passages=open_domain_data/cqadupstack/physics/psgs.tsv
  beir=open_domain_data/cqadupstack_beir/physics
elif [[ ${index_short_name} == 'cqadupstack_programmers' ]]; then
  queries=open_domain_data/cqadupstack/programmers/test.json
  passages=open_domain_data/cqadupstack/programmers/psgs.tsv
  beir=open_domain_data/cqadupstack_beir/programmers
else
  exit
fi

python prep.py --task rank2json --inp ${pkl} ${queries} ${passages} --out ${json}
if [[ ${index_short_name} == 'nq_test_top10' ]]; then
  ./rerank.sh false ${json} score default
else
  ./rerank.sh true ${json} ${beir} ${split} score default
fi
