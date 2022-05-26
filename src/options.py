# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=1000)
        self.parser.add_argument('--total_steps', type=int, default=1000)
        self.parser.add_argument('--scheduler_steps', type=int, default=None, 
                        help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--scheduler', type=str, default='fixed')
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--fixed_lr', action='store_true')

    def add_eval_options(self):
        self.parser.add_argument('--write_results', action='store_true', help='save results')
        self.parser.add_argument('--write_crossattention_scores', action='store_true', help='save dataset with cross-attention scores')

    def add_reader_options(self):
        self.parser.add_argument('--train_data', type=str, default='none', help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default='none', help='path of eval data')
        self.parser.add_argument('--model_size', type=str, default='base')
        self.parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the encoder')
        self.parser.add_argument('--use_sharding', action='store_true', help='use ZeRO')
        self.parser.add_argument('--text_maxlength', type=int, default=200, 
                        help='maximum number of tokens in text segments (question+passage)')
        self.parser.add_argument('--answer_maxlength', type=int, default=-1, 
                        help='maximum number of tokens used to train the model, no truncation if -1')
        self.parser.add_argument('--no_title', action='store_true', 
                        help='article titles not included in passages')
        self.parser.add_argument('--n_context', type=int, default=1)
        self.parser.add_argument('--only_topk_n_context', type=int, default=None,
                                 help='mimic in-batch negatives in encoder')
        self.parser.add_argument('--n_layer_two_tower', type=int, default=0,
                                 help='number of layers used for two tower representation')
        self.parser.add_argument('--layer_for_retrieval', type=str, default='first',
                                 choices=['first', 'emb-first', 'emb', 'prev-first', 'after-first', 'last-first'],
                                 help='layers used for retrieval')
        self.parser.add_argument('--attention_mask', type=str, default=None,
                                 choices=[None, 'separate', 'query-side', 'no-query'],
                                 help='how to generate attention for query/doc')
        self.parser.add_argument('--retrieval_aggregation_method', type=str, default='all-avg-max',
                                 choices=['all-avg-max', 'all-avg-avg', 'all-max-all', 'all-all-all'],
                                 help='how to generate a single score for retrieval')
        self.parser.add_argument('--query_in_decoder', type=str, default='no', choices=['no', 'all'],
                                 help='use query at the beginning of the decoder')
        self.parser.add_argument('--num_keep_ctx_in_decoder', type=int, default=0, help='num of ctx used in decoder')
        self.parser.add_argument('--combine_weight', type=float, default=0,
                                 help='weight (before exp) to combine bi-encoder attn and decoder attn')
        self.parser.add_argument('--keep_ctx_in_decoder_head_tau', type=float, default=1.0,
                                 help='tau for head weight softmax')
        self.parser.add_argument('--head_weights_norm_func', type=str, default='softmax',
                                 choices=['softmax', 'sparsemax'],
                                 help='head weights normalization function')
        self.parser.add_argument('--keep_ctx_in_decoder_with_head', type=int, default=None,
                                 help='only use a specific head to keep ctx in decoder')
        self.parser.add_argument('--encoder_decoder_kl_ratio', type=float, default=0,
                                 help='the ratio of KL divergence between encoder and decoder attn')
        self.parser.add_argument('--encoder_encoder_kl_ratio', type=float, default=0,
                                 help='the ratio of KL divergence between encoder and encoder attn of different layers')
        self.parser.add_argument('--encoder_encoder_kl_sparsity', type=int, default=0, help='topk doc tokens per query token')
        self.parser.add_argument('--encoder_encoder_kl', type=str, default=None,
                                 help='specify the two layers and two heads to compute kl')
        self.parser.add_argument('--decoder_attn_ctx_normalize', action='store_true',
                                 help='normalize decoder attention for each context')
        self.parser.add_argument('--encoder_attention_pre_softmax', action='store_true',
                                 help='softmax encoder attention before KL or adding to decoder attention')
        self.parser.add_argument('--metric', type=str, default='em', choices=['em', 'rougel'])
        self.parser.add_argument('--in_batch_negative', action='store_true',
                                 help='activate in-batch negative in KL divergence')
        self.parser.add_argument('--in_batch_negative_size', type=int, default=0,
                                 help='the number of queries use in in-batch negative in total')
        self.parser.add_argument('--in_batch_negative_max_num_query', type=int, default=None,
                                 help='the number of queries to compute at one time for in-batch negative to save memory')
        self.parser.add_argument('--pairwise_loss', type=str, default=None, choices=[None, 'sigmoid'],
                                 help='pairwise loss function used when --in_batch_negative is activated')
        self.parser.add_argument('--memory_bank', type=int, default=0, help='number of documents kept in memory bank')
        self.parser.add_argument('--memory_bank_topk', type=int, default=0, help='number of documents retrieved from memory bank')
        self.parser.add_argument('--memory_use_random', action='store_true', help='use random doc from memory bank')
        self.parser.add_argument('--memory_bank_recompute', action='store_true',
                                 help='recompute the representation of docs from memory bank')
        self.parser.add_argument('--memory_bank_additional_encode', action='store_true',
                                 help='compute decoder-attention for docs from memory bank')
        self.parser.add_argument('--memory_bank_gpu', type=str, default=None, help='gpu index connected by , used for memory bank')
        self.parser.add_argument('--max_over_head', action='store_true', help='max over all attention heads in aggregation')
        self.parser.add_argument('--term_weight_parameter', action='store_true', help='use additional parameters to learn term weight')
        self.parser.add_argument('--embedding_normalize', action='store_true', help='normalize embedding before computing the bi-encoder score')

    def add_retriever_options(self):
        self.parser.add_argument('--train_data', type=str, default='none', help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default='none', help='path of eval data')
        self.parser.add_argument('--indexing_dimension', type=int, default=768)
        self.parser.add_argument('--no_projection', action='store_true', 
                        help='No addition Linear layer and layernorm, only works if indexing size equals 768')
        self.parser.add_argument('--question_maxlength', type=int, default=40, 
                        help='maximum number of tokens in questions')
        self.parser.add_argument('--passage_maxlength', type=int, default=200, 
                        help='maximum number of tokens in passages')
        self.parser.add_argument('--no_question_mask', action='store_true')
        self.parser.add_argument('--no_passage_mask', action='store_true')
        self.parser.add_argument('--extract_cls', action='store_true')
        self.parser.add_argument('--no_title', action='store_true', 
                        help='article titles not included in passages')
        self.parser.add_argument('--n_context', type=int, default=1)
        self.parser.add_argument('--init_with', type=str, default='bert-base-uncased')
        self.parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the encoder')
        self.parser.add_argument('--scale_dot_product', action='store_true', help='scale down the dot product')

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')
        self.parser.add_argument('--model_path', type=str, default='none', help='path for continue training')
        self.parser.add_argument('--init_from', type=str, default=None, help='path for loading initial parameteres')

        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
                        help="Batch size per GPU/CPU for training.")
        self.parser.add_argument('--maxload', type=int, default=-1)

        self.parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1,
                        help="Main port (for multi-node SLURM jobs)")
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        self.parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
        self.parser.add_argument('--augmentation', type=str, default=None, help='query augmentation')
        # training parameters
        self.parser.add_argument('--eval_freq', type=int, default=500,
                        help='evaluate model every <eval_freq> steps during training')
        self.parser.add_argument('--eval_num_examples', type=int, default=None,
                                 help='number of examples used in evaluation during training')
        self.parser.add_argument('--save_freq', type=int, default=5000,
                        help='save model every <save_freq> steps during training')
        self.parser.add_argument('--eval_print_freq', type=int, default=1000,
                        help='print intermdiate results of evaluation every <eval_print_freq> steps')
        # wandb
        self.parser.add_argument('--wandb_entity', type=str, default='jzbjyb')
        self.parser.add_argument('--wandb_project', type=str, default='adapt-knowledge')
        self.parser.add_argument('--wandb_name', type=str, default='test')
        self.parser.add_argument('--wandb_log_freq', type=int, default=1)


    def print_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        expr_dir = Path(opt.checkpoint_dir)/ opt.name
        model_dir = expr_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(expr_dir/'opt.log', 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        logger.info(message)

    def parse(self):
        opt = self.parser.parse_args()
        return opt


def get_options(use_reader=False,
                use_retriever=False,
                use_optim=False,
                use_eval=False):
    options = Options()
    if use_reader:
        options.add_reader_options()
    if use_retriever:
        options.add_retriever_options()
    if use_optim:
        options.add_optim_options()
    if use_eval:
        options.add_eval_options()
    return options.parse()
