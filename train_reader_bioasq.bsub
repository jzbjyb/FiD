#!/usr/bin/env bash

#BSUB -J fid
#BSUB -o bosch/%J.stdout
#BSUB -e bosch/%J.stderr
#BSUB -W 48:00
#BSUB -n 4
#BSUB -M 50000
#BSUB -gpu "num=4:mode=exclusive_process:mps=no"
#BSUB -q batch_v100

source activate beir

module load proxy4server-access
source /fs/applications/p4s-access/1.0/ActivateP4S.sh -a

export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

#train_data=open_domain_data/bioasq_500k.nosummary/train.json
#eval_data=open_domain_data/bioasq_500k.nosummary/test.json
train_data=open_domain_data/bioasq_500k.nosummary.pseudo100k/train.json
eval_data=open_domain_data/bioasq_500k.nosummary.pseudo100k/dev.json
metric=em

MAX_NUM_GPU_PER_NODE=8
num_gpu=4
batch_size=1
accum=4
num_keep_ctx_in_decoder=0
combine_weight=0

init_model=google/t5-base-lm-adapt
ckpt_dir=trained_reader
name=nq_reader_base_v11lm_separate_layer6_continue_kl1_tau0001_adapt_bioasq_500k_nosummary_pseudo100k_kl_inbatchneg4_step5h
init_from=${ckpt_dir}/nq_reader_base_v11lm_separate_layer6_continue_kl1_tau0001/checkpoint/latest
n_context=100
only_topk_n_context=0
n_layer_two_tower=6
layer_for_retrieval=first
keep_ctx_in_decoder_with_head=3
keep_ctx_in_decoder_head_tau=0.001
head_weights_norm_func=softmax
encoder_decoder_kl_ratio=1.0
encoder_encoder_kl_ratio=0.0
encoder_encoder_kl="first|last=3|10"
encoder_encoder_kl_sparsity=0
retrieval_aggregation_method=all-avg-max
attention_mask=separate
query_in_decoder=no

if [[ ${query_in_decoder} == 'no' ]]; then
  answer_maxlength=50
else
  answer_maxlength=100
fi

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

python ${prefix} train_reader.py \
  --train_data ${train_data} \
  --eval_data ${eval_data} \
  --model_size ${init_model} \
  --use_checkpoint \
  --text_maxlength 512 \
  --answer_maxlength ${answer_maxlength} \
  --per_gpu_batch_size ${batch_size} \
  --accumulation_steps ${accum} \
  --n_context ${n_context} \
  --only_topk_n_context ${only_topk_n_context} \
  --name ${name} \
  --checkpoint_dir ${ckpt_dir} \
  --lr 0.00005 \
  --optim adamw \
  --scheduler linear \
  --weight_decay 0.01 \
  --n_layer_two_tower ${n_layer_two_tower} \
  --layer_for_retrieval ${layer_for_retrieval} \
  --num_keep_ctx_in_decoder ${num_keep_ctx_in_decoder} \
  --combine_weight ${combine_weight} \
  --keep_ctx_in_decoder_head_tau ${keep_ctx_in_decoder_head_tau} \
  --head_weights_norm_func ${head_weights_norm_func} \
  --encoder_decoder_kl_ratio ${encoder_decoder_kl_ratio} \
  --encoder_encoder_kl_ratio ${encoder_encoder_kl_ratio} \
  --encoder_encoder_kl ${encoder_encoder_kl} \
  --encoder_encoder_kl_sparsity ${encoder_encoder_kl_sparsity} \
  --retrieval_aggregation_method ${retrieval_aggregation_method} \
  --attention_mask ${attention_mask} \
  --query_in_decoder ${query_in_decoder} \
  --total_step 500 \
  --warmup_step 50 \
  --save_freq 500 \
  --eval_freq 100 \
  --eval_num_examples 200 \
  --metric ${metric} \
  --wandb_name ${ckpt_dir}/${name} \
  --init_from ${init_from} \
  --in_batch_negative

# --keep_ctx_in_decoder_with_head ${keep_ctx_in_decoder_with_head} \
# --decoder_attn_ctx_normalize \
# --encoder_attention_pre_softmax \
