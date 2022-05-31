data_root=open_domain_data

function min() {
  echo $(( $1 < $2 ? $1 : $2 ))
}

function get_dataset_settings() {
  index_name=$1
  length_limit=$2
  gpu=$3
  
  num_shards=1
  save_every_n_doc=0
  num_workers=4
  query_maxlength=$( min 50 ${length_limit} )  # 50 query tokens
  answer_maxlength=50
  if [[ ${index_name} == 'nq_test_top10' ]]; then
    passages=${data_root}/NQ/psgs_w100.test_top10_aggregate.tsv
    queries=${data_root}/NQ/test.json
    passage_maxlength=$( min 200 ${length_limit} )
    text_maxlength=250
  elif [[ ${index_name} == 'nq' ]]; then
    passages=${data_root}/NQ/psgs_w100.tsv
    queries=${data_root}/NQ/test.json
    passage_maxlength=$( min 200 ${length_limit} )
    text_maxlength=250
    num_shards=$( nvidia-smi --query-gpu=name --format=csv,noheader | wc -l )  # use all gpus
    save_every_n_doc=100000
    num_workers=0
  elif [[ ${index_name} == 'msmarcoqa_dev' ]]; then
    passages=${data_root}/msmarco_qa/psgs.dev_aggregate.tsv
    queries=${data_root}/msmarco_qa/dev.json
    beir=${data_root}/msmarco_qa_beir
    split=dev
    passage_maxlength=$( min 200 ${length_limit} )
    text_maxlength=250
  elif [[ ${index_name} == 'bioasq_500k_test' ]]; then
    passages=${data_root}/bioasq_500k.nosummary/psgs.test_aggregate.tsv
    queries=${data_root}/bioasq_500k.nosummary/test.json
    beir=${data_root}/bioasq_500k.nosummary_beir
    split=test
    passage_maxlength=$( min 1024 ${length_limit} )
    text_maxlength=${passage_maxlength}
  elif [[ ${index_name} == 'fiqa' ]]; then
    passages=${data_root}/fiqa/psgs.tsv
    queries=${data_root}/fiqa/test.json
    beir=${data_root}/fiqa_beir
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
    text_maxlength=${passage_maxlength}
  elif [[ ${index_name} == 'cqadupstack_mathematica' ]]; then
    passages=${data_root}/cqadupstack/mathematica/psgs.tsv
    queries=${data_root}/cqadupstack/mathematica/test.json
    beir=${data_root}/cqadupstack_beir/mathematica
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
    text_maxlength=${passage_maxlength}
  elif [[ ${index_name} == 'cqadupstack_physics' ]]; then
    passages=${data_root}/cqadupstack/physics/psgs.tsv
    queries=${data_root}/cqadupstack/physics/test.json
    beir=${data_root}/cqadupstack_beir/physics
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
    text_maxlength=${passage_maxlength}
  elif [[ ${index_name} == 'cqadupstack_programmers' ]]; then
    passages=${data_root}/cqadupstack/programmers/psgs.tsv
    queries=${data_root}/cqadupstack/programmers/test.json
    beir=${data_root}/cqadupstack_beir/programmers
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
    text_maxlength=${passage_maxlength}
  else
    exit 1
  fi

  if [[ ${gpu} == 'a100' ]]; then  # 80G
    fid_per_gpu_batch_size=8
    passage_per_gpu_batch_size=128
    query_per_gpu_batch_size=256
  elif [[ ${gpu} == 'v100' ]]; then  # 32G
    fid_per_gpu_batch_size=4
    passage_per_gpu_batch_size=128
    query_per_gpu_batch_size=256
  else  # 12G
    fid_per_gpu_batch_size=2
    passage_per_gpu_batch_size=64
    query_per_gpu_batch_size=128
  fi
}

function get_prefix() {
  MAX_NUM_GPU_PER_NODE=8
  num_gpu=$1
  if (( ${num_gpu} == 1 )); then
    echo 'single-GPU'
    prefix=""
  elif (( ${num_gpu} <= ${MAX_NUM_GPU_PER_NODE} )); then
    echo 'single-node'
    export NGPU=${num_gpu}
    random_port=$(shuf -i 10000-65000 -n 1)
    prefix="-m torch.distributed.launch --nproc_per_node=${num_gpu} --master_port=${random_port}"
  else
    echo 'multi-node'
    prefix=""
    exit  # TODO: not implemented
  fi
}
