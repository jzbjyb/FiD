# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import transformers
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler


import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model
from src.util import global_context


def evaluate(model, dataset, dataloader, tokenizer, opt):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention(opt.n_context)
        model.reset_score_storage() 
    total = 0
    exactmatch = []
    write_path = (Path(opt.checkpoint_dir) / opt.name / 'test_results' / ('%d.txt' % opt.global_rank)) \
      if opt.write_results else None
    with src.util.open_file(write_path, 'a') as fw, torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), disable=not opt.is_main):
            idx, labels, _, context_ids, context_mask, context_sep_mask = batch[:6]

            if opt.write_crossattention_scores:
                model.reset_score_storage()
            context_sep_mask = context_sep_mask.cuda() if context_sep_mask is not None else context_sep_mask
            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                attention_separate_mask=context_sep_mask,
                max_length=opt.answer_maxlength,
            )

            # extract answers and their starting positions
            queries, predictions, predictions_position = list(zip(*[
              src.util.extract_query_answer(o, tokenizer, query_in_decoder=opt.query_in_decoder) for o in outputs]))

            if opt.write_crossattention_scores:
                crossattention_scores, crossattention_encoder_scores = model.get_crossattention_scores(
                  predictions_position,
                  context_mask.cuda(),
                  sum_over_head_and_layer=False)
                retrieval = model.get_collected_for_retrieval()

            for k, o in enumerate(outputs):
                query, ans = queries[k], predictions[k]
                example = dataset.data[idx[k]]
                if 'answers' in example:
                    score = src.evaluation.ems(ans, example['answers'])
                    #score = src.evaluation.ems(queries[k].strip('question:').strip().lower(), [example['question'].strip().lower()])
                    exactmatch.append(score)

                if fw is not None:
                    fw.write(str(example['id']) + "\t" + ans + '\n')
                if opt.write_crossattention_scores:
                    for j in range(context_ids.size(1)):
                        cs = crossattention_scores[k, j]
                        cs = cs.item() if cs.dim() == 0 else cs.cpu().numpy().tolist()
                        example['ctxs'][j]['score'] = cs

                        if crossattention_encoder_scores is not None:
                          ces = crossattention_encoder_scores[k, j]
                          example['ctxs'][j]['encoder_score'] = ces.item()

                        if 'two_tower_attn_score' not in retrieval:
                          continue
                        tt = retrieval['two_tower_attn_score'][k, j]
                        tt = tt.item() if tt.dim() == 0 else tt.cpu().numpy().tolist()
                        example['ctxs'][j]['two_tower_attn_score'] = tt

                total += 1
            if (i + 1) % opt.eval_print_freq == 0:
                log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
                if len(exactmatch) == 0:
                    log += '| no answer to compute scores'
                else:
                    log += f' | average = {np.mean(exactmatch):.3f}'
                logger.warning(log)

    logger.warning(f'Process rank:{opt.global_rank}, total {total} | average = {np.mean(exactmatch):.3f}')
    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    
    return score, total


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    global_context['opt'] = opt

    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir) / opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)

    collator_function = src.data.Collator(
        opt.text_maxlength,
        tokenizer,
        answer_maxlength=opt.answer_maxlength,
        separate_query_passage=opt.attention_mask,
        query_in_decoder=opt.query_in_decoder)
    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank,  #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size,
        n_context=opt.n_context,
    )
    if opt.eval_num_examples:
      eval_examples = eval_examples[:opt.eval_num_examples]
    eval_dataset = src.data.Dataset(
        eval_examples, 
        opt.n_context, 
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=opt.num_workers,
        collate_fn=collator_function
    )
    
    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    logger.info("Start eval")
    exactmatch, total = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}')

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        src.util.write_output(glob_path, write_path) 
    if opt.write_crossattention_scores:
        src.util.save_distributed_dataset(eval_dataset.data, opt)
