data_root=open_domain_data

function min() {
  echo $(( $1 < $2 ? $1 : $2 ))
}

function get_dataset_settings() {
  index_name=$1
  length_limit=$2
  gpu=$3

  query_maxlength=$( min 50 ${length_limit} )  # 50 query tokens
  if [[ ${index_name} == 'nq_test_top10' ]]; then
    passages=${data_root}/NQ/psgs_w100.test_top10_aggregate.tsv
    queries=${data_root}/NQ/test.json
    passage_maxlength=$( min 200 ${length_limit} )
  elif [[ ${index_name} == 'msmarcoqa_dev' ]]; then
    passages=${data_root}/msmarco_qa/psgs.dev_aggregate.tsv
    queries=${data_root}/msmarco_qa/dev.json
    beir=${data_root}/msmarco_qa_beir
    split=dev
    passage_maxlength=$( min 200 ${length_limit} )
  elif [[ ${index_name} == 'bioasq_500k_test' ]]; then
    passages=${data_root}/bioasq_500k.nosummary/psgs.test_aggregate.tsv
    queries=${data_root}/bioasq_500k.nosummary/test.json
    beir=${data_root}/bioasq_500k.nosummary_beir
    split=test
    passage_maxlength=$( min 1024 ${length_limit} )
  elif [[ ${index_name} == 'fiqa' ]]; then
    passages=${data_root}/fiqa/psgs.tsv
    queries=${data_root}/fiqa/test.json
    beir=${data_root}/fiqa_beir
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
  elif [[ ${index_name} == 'cqadupstack_mathematica' ]]; then
    passages=${data_root}/cqadupstack/mathematica/psgs.tsv
    queries=${data_root}/cqadupstack/mathematica/test.json
    beir=${data_root}/cqadupstack_beir/mathematica
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
  elif [[ ${index_name} == 'cqadupstack_physics' ]]; then
    passages=${data_root}/cqadupstack/physics/psgs.tsv
    queries=${data_root}/cqadupstack/physics/test.json
    beir=${data_root}/cqadupstack_beir/physics
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
  elif [[ ${index_name} == 'cqadupstack_programmers' ]]; then
    passages=${data_root}/cqadupstack/programmers/psgs.tsv
    queries=${data_root}/cqadupstack/programmers/test.json
    beir=${data_root}/cqadupstack_beir/programmers
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
  else
    exit 1
  fi

  if [[ ${gpu} == 'a100' ]]; then  # 80G
    passage_per_gpu_batch_size=256
    query_per_gpu_batch_size=512
  elif [[ ${gpu} == 'v100' ]]; then  # 32G
    passage_per_gpu_batch_size=128
    query_per_gpu_batch_size=256
  else  # 12G
    passage_per_gpu_batch_size=64
    query_per_gpu_batch_size=128
  fi
}
