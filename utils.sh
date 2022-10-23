data_root=open_domain_data

function min() {
  echo $(( $1 < $2 ? $1 : $2 ))
}

function get_gpu_type() {
  gpu_info=$( nvidia-smi --query-gpu=name --format=csv )
  if [[ ${gpu_info} == *"V100"* ]]; then
    echo v100
  elif [[ ${gpu_info} == *"A100"* ]]; then
    echo a100
  else
    echo none
  fi
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
  
  elif [[ ${index_name} == 'nq_train3k_top10' ]]; then
    passages=${data_root}/NQ/psgs_w100.train3k_top10_aggregate.tsv
    queries=${data_root}/NQ/train.3k.json
    passage_maxlength=$( min 200 ${length_limit} )
    text_maxlength=250
  
  elif [[ ${index_name} == 'nq_test' ]]; then
    passages=${data_root}/NQ/psgs_w100.test_aggregate.tsv
    queries=${data_root}/NQ/test.json
    passage_maxlength=$( min 200 ${length_limit} )
    text_maxlength=250
    num_shards=$( nvidia-smi --query-gpu=name --format=csv,noheader | wc -l )  # use all gpus
    save_every_n_doc=100000
    num_workers=0
  
  elif [[ ${index_name} == 'nq_test_005' ]]; then
    passages=${data_root}/NQ/psgs_w100.test_aggregate_and_005.tsv
    queries=${data_root}/NQ/test.json
    passage_maxlength=$( min 200 ${length_limit} )
    text_maxlength=250
    num_shards=$( nvidia-smi --query-gpu=name --format=csv,noheader | wc -l )  # use all gpus
    save_every_n_doc=100000
    num_workers=0
  
  elif [[ ${index_name} == 'nq' ]]; then
    passages=${data_root}/NQ/psgs_w100.tsv
    queries=${data_root}/NQ/test.json
    #queries=${data_root}/NQ/train.json
    passage_maxlength=$( min 200 ${length_limit} )
    text_maxlength=250
    num_shards=$( nvidia-smi --query-gpu=name --format=csv,noheader | wc -l )  # use all gpus
    save_every_n_doc=100000
    num_workers=0
  
  elif [[ ${index_name} == 'sciq' ]]; then
    passages=${data_root}/SciQ/psgs.tsv
    queries=${data_root}/SciQ/test.json
    passage_maxlength=$( min 200 ${length_limit} )
    text_maxlength=250
  
  elif [[ ${index_name} == 'msmarcoqa_dev' ]]; then
    passages=${data_root}/msmarco_qa/psgs.dev_aggregate.tsv
    queries=${data_root}/msmarco_qa/dev.json
    beir=${data_root}/msmarco_qa_beir
    split=dev
    passage_maxlength=$( min 200 ${length_limit} )
    text_maxlength=250
  
  elif [[ ${index_name} == 'msmarco' ]]; then
    passages=${data_root}/msmarco/psgs.tsv
    queries=${data_root}/msmarco/dev.json
    beir=${data_root}/msmarco_beir
    split=dev
    passage_maxlength=$( min 200 ${length_limit} )
    text_maxlength=250
    num_shards=$( nvidia-smi --query-gpu=name --format=csv,noheader | wc -l )  # use all gpus
    save_every_n_doc=100000
    num_workers=0
  
  elif [[ ${index_name} == 'bioasq_1m' ]]; then
    passages=${data_root}/bioasq_1m/psgs.tsv
    queries=${data_root}/bioasq_1m/test.json
    beir=${data_root}/bioasq_1m_beir
    split=test
    passage_maxlength=$( min 1024 ${length_limit} )
    text_maxlength=${passage_maxlength}
    num_shards=$( nvidia-smi --query-gpu=name --format=csv,noheader | wc -l )  # use all gpus
    save_every_n_doc=100000
    num_workers=0
  
  elif [[ ${index_name} == 'bioasq_500k' ]]; then
    passages=${data_root}/bioasq_500k.nosummary/psgs.tsv
    queries=${data_root}/bioasq_500k.nosummary/test.json
    beir=${data_root}/bioasq_500k.nosummary_beir
    split=test
    passage_maxlength=$( min 1024 ${length_limit} )
    text_maxlength=${passage_maxlength}
    num_shards=$( nvidia-smi --query-gpu=name --format=csv,noheader | wc -l )  # use all gpus
    save_every_n_doc=100000
    num_workers=0
  
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

  elif [[ ${index_name} == 'scifact' ]]; then
    passages=${data_root}/scifact/psgs.tsv
    queries=${data_root}/scifact/test.json
    beir=${data_root}/scifact_beir
    split=test
    passage_maxlength=$( min 1024 ${length_limit} )
    text_maxlength=${passage_maxlength}
  
  elif [[ ${index_name} == 'nfcorpus' ]]; then
    passages=${data_root}/nfcorpus/psgs.tsv
    queries=${data_root}/nfcorpus/test.json
    beir=${data_root}/nfcorpus_beir
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
    text_maxlength=${passage_maxlength}
  
  elif [[ ${index_name} == 'scidocs' ]]; then
    passages=${data_root}/scidocs/psgs.tsv
    queries=${data_root}/scidocs/test.json
    beir=${data_root}/scidocs_beir
    split=test
    passage_maxlength=$( min 1024 ${length_limit} )
    text_maxlength=${passage_maxlength}
  
  elif [[ ${index_name} == 'trec_covid' ]]; then
    passages=${data_root}/trec_covid/psgs.tsv
    queries=${data_root}/trec_covid/test.json
    beir=${data_root}/trec_covid_beir
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
    text_maxlength=${passage_maxlength}
  
  elif [[ ${index_name} == 'touche2020' ]]; then
    passages=${data_root}/touche2020/psgs.tsv
    queries=${data_root}/touche2020/test.json
    beir=${data_root}/touche2020_beir
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
    text_maxlength=${passage_maxlength}
    num_shards=$( nvidia-smi --query-gpu=name --format=csv,noheader | wc -l )  # use all gpus
    save_every_n_doc=100000
    num_workers=0
  
   elif [[ ${index_name} == 'quora' ]]; then
    passages=${data_root}/quora/psgs.tsv
    queries=${data_root}/quora/test.json
    beir=${data_root}/quora_beir
    split=test
    passage_maxlength=$( min 200 ${length_limit} )
    text_maxlength=${passage_maxlength}

  elif [[ ${index_name} == 'cqadupstack_android' ]]; then
    passages=${data_root}/cqadupstack/android/psgs.tsv
    queries=${data_root}/cqadupstack/android/test.json
    beir=${data_root}/cqadupstack_beir/android
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
    text_maxlength=${passage_maxlength}
  
  elif [[ ${index_name} == 'cqadupstack_english' ]]; then
    passages=${data_root}/cqadupstack/english/psgs.tsv
    queries=${data_root}/cqadupstack/english/test.json
    beir=${data_root}/cqadupstack_beir/english
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
    text_maxlength=${passage_maxlength}
  
  elif [[ ${index_name} == 'cqadupstack_gaming' ]]; then
    passages=${data_root}/cqadupstack/gaming/psgs.tsv
    queries=${data_root}/cqadupstack/gaming/test.json
    beir=${data_root}/cqadupstack_beir/gaming
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
    text_maxlength=${passage_maxlength}
  
  elif [[ ${index_name} == 'cqadupstack_gis' ]]; then
    passages=${data_root}/cqadupstack/gis/psgs.tsv
    queries=${data_root}/cqadupstack/gis/test.json
    beir=${data_root}/cqadupstack_beir/gis
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
  
  elif [[ ${index_name} == 'cqadupstack_stats' ]]; then
    passages=${data_root}/cqadupstack/stats/psgs.tsv
    queries=${data_root}/cqadupstack/stats/test.json
    beir=${data_root}/cqadupstack_beir/stats
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
    text_maxlength=${passage_maxlength}
  
  elif [[ ${index_name} == 'cqadupstack_tex' ]]; then
    passages=${data_root}/cqadupstack/tex/psgs.tsv
    queries=${data_root}/cqadupstack/tex/test.json
    beir=${data_root}/cqadupstack_beir/tex
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
    text_maxlength=${passage_maxlength}
  
  elif [[ ${index_name} == 'cqadupstack_unix' ]]; then
    passages=${data_root}/cqadupstack/unix/psgs.tsv
    queries=${data_root}/cqadupstack/unix/test.json
    beir=${data_root}/cqadupstack_beir/unix
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
    text_maxlength=${passage_maxlength}
  
  elif [[ ${index_name} == 'cqadupstack_webmasters' ]]; then
    passages=${data_root}/cqadupstack/webmasters/psgs.tsv
    queries=${data_root}/cqadupstack/webmasters/test.json
    beir=${data_root}/cqadupstack_beir/webmasters
    split=test
    passage_maxlength=$( min 512 ${length_limit} )
    text_maxlength=${passage_maxlength}
  
  elif [[ ${index_name} == 'cqadupstack_wordpress' ]]; then
    passages=${data_root}/cqadupstack/wordpress/psgs.tsv
    queries=${data_root}/cqadupstack/wordpress/test.json
    beir=${data_root}/cqadupstack_beir/wordpress
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
    passage_per_gpu_batch_size=64
    query_per_gpu_batch_size=128
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
