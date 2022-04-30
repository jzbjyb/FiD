#!/usr/bin/env bash
source utils.sh

index_name=$1
model_path=$2
topk=$3
pkl=${model_path}.index/${index_name}/qid2rank_${topk}.pkl
json=${model_path}.index/${index_name}/qid2rank_${topk}.json

get_dataset_settings ${index_name} 1024 ${gpu}  # 1024 doesn't matter

python prep.py --task rank2json --inp ${pkl} ${queries} ${passages} --out ${json}
if [[ ${index_name} == 'nq_test_top10' ]]; then
  ./rerank.sh false ${json} score default
else
  ./rerank.sh true ${json} ${beir} ${split} score default
fi
