#!/usr/bin/env bash
#SBATCH --job-name=iter
#SBATCH --cpus-per-task=80
#SBATCH --nodes=1
#SBATCH --time=128:00:00
#SBATCH --partition=side
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=512GB

# env
eval "$(conda shell.bash hook)"
conda activate raat

# utils
source utils.sh

# wandb
export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

#train_data=open_domain_data/NQ/train.json
#train_data=open_domain_data/NQ_bm25/train.json
#train_data=../ColBERT/experiments/colbert-60000.dnn/indexes/nq.2bits/result_train_2probe_8192cand.json
#train_data=../ColBERT/experiments/nq_reader_large_v11lm_separate_layer12_bs16_step3k_continue_bm25_kl1_inbatchneg64_accumulate4_ratio8_step8k/indexes/nq.8bits/result_train_4probe_16384cand_real_skiplargestview.json
train_data=open_domain_data/result_train_4probe_16384cand_real_skiplargestview.json
eval_data=open_domain_data/NQ/dev.json
#train_data=open_domain_data/SciQ/train.json
#eval_data=open_domain_data/SciQ/dev.json
metric=em

num_gpu=8
batch_size=8
accum=1
n_context=100
text_maxlength=250
answer_maxlength=50
num_keep_ctx_in_decoder=0
combine_weight=0

init_model=google/t5-large-lm-adapt
ckpt_dir=trained_reader
name=nq_reader_large_v11lm_separate_layer12_bs16_step3k_continue_bm25_kl1_inbatchneg64_accumulate4_ratio8_step8k_iterative_self_step8k_step8k
init_from=${ckpt_dir}/nq_reader_large_v11lm_separate_layer12_bs16_step3k_continue_bm25_kl1_inbatchneg64_accumulate4_ratio8_step8k_iterative_self_step8k/checkpoint/latest
only_topk_n_context=0
n_layer_two_tower=12
layer_for_retrieval=first
keep_ctx_in_decoder_with_head=0
keep_ctx_in_decoder_head_tau=0.001
head_weights_norm_func=softmax
encoder_decoder_kl_ratio=8.0
encoder_encoder_kl_ratio=0.0
encoder_encoder_kl="first|last=3|10"
encoder_encoder_kl_sparsity=0
retrieval_aggregation_method=all-avg-max
attention_mask=separate
query_in_decoder=no

get_prefix ${num_gpu}

cat ./train_reader_large.bsub &> ${ckpt_dir}/${name}.out
python ${prefix} train_reader.py \
  --train_data ${train_data} \
  --eval_data ${eval_data} \
  --model_size ${init_model} \
  --use_checkpoint \
  --text_maxlength ${text_maxlength} \
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
  --total_step 8000 \
  --warmup_step 800 \
  --save_freq 2000 \
  --eval_freq 400 \
  --eval_num_examples 200 \
  --metric ${metric} \
  --wandb_name ${ckpt_dir}/${name} \
  --wandb_log_freq 10 \
  --in_batch_negative \
  --accumulation_for_ibn 4 \
  --init_from ${init_from} &>> ${ckpt_dir}/${name}.out
