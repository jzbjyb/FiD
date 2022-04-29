#!/usr/bin/env bash

use_annotation=$1

if [[ ${use_annotation} == 'true' ]]; then
  ret_file=$2
  beir_dir=$3
  split=$4
  other="${@:5}"
  python prep.py --task eval_answer --inp ${ret_file} ${beir_dir} ${split} --other ${other}
else
  ret_file=$2
  other="${@:3}"
  python prep.py --task eval_answer --inp ${ret_file} --other ${other}
fi
