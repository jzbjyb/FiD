#!/usr/bin/env bash
source utils.sh

index_name=$1
model_path=$2
topk=$3
index_name_suffix=$4
metric=$5

pkl=${model_path}.index/${index_name}${index_name_suffix}/qid2rank_${topk}.pkl
json=${model_path}.index/${index_name}${index_name_suffix}/qid2rank_${topk}.json

get_dataset_settings ${index_name} 1024 ${gpu}  # 1024 doesn't matter

python prep.py --task rank2json --inp ${pkl} ${queries} ${passages} --out ${json}
if [[ ${index_name} == 'nq_test_top10' || ${index_name} == 'nq' || ${index_name} == 'nq_test' || ${index_name} == 'nq_test_005' || ${index_name} == 'nq_train3k_top10' ]]; then
  ./rerank.sh false ${json} ${metric} score default
else
  ./rerank.sh true ${json} ${beir} ${split} ${metric} score default
fi
