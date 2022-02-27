#!/usr/bin/env bash

inp=$1
other="${@:2}"

python prep.py --task eval_answer --inp ${inp} --other ${other}
