#!/usr/bin/env bash
#SBATCH --mem=64000
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH --time=0
#SBATCH --output=slurm_out/%j.out

export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

train_data=open_domain_data/NQ/train.json
eval_data=open_domain_data/NQ/dev.json
#train_data=open_domain_data/scifact/train.json
#eval_data=open_domain_data/scifact/test.json
#train_data=open_domain_data/SciQ/train.json
#eval_data=open_domain_data/SciQ/dev.json
#train_data=open_domain_data/quasar_s/train.json
#eval_data=open_domain_data/quasar_s/dev.json
metric=em

MAX_NUM_GPU_PER_NODE=8
num_gpu=$1
batch_size=2
accum=$2
num_keep_ctx_in_decoder=0
combine_weight=0

init_model=google/t5-base-lm-adapt
ckpt_dir=trained_reader
name=nq_reader_base_v11lm_separate_layer6_continue_kl1_tau0001_bank50000_100_random_recompute
init_from=${ckpt_dir}/nq_reader_base_v11lm_separate_layer6/checkpoint/latest
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
  --text_maxlength 250 \
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
  --total_step 1000 \
  --warmup_step 100 \
  --save_freq 1000 \
  --eval_freq 200 \
  --eval_num_examples 200 \
  --metric ${metric} \
  --wandb_name ${ckpt_dir}/${name} \
  --init_from ${init_from} \
  --memory_bank 50000 \
  --memory_bank_topk 100 \
  --memory_use_random \
  --memory_bank_recompute

# --keep_ctx_in_decoder_with_head ${keep_ctx_in_decoder_with_head} \
# --decoder_attn_ctx_normalize \
# --encoder_attention_pre_softmax \

# NQ
# 3501
# 250
# 500
# 300
# 200

# 1001
# 100
# 1000
# 200
# 200

# SciQ
# 1001
# 100
# 1000
# 50
# 100

# 501
# 50
# 500
# 50
# 100
