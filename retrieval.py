# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Dict
import argparse
from collections import defaultdict
from tqdm import tqdm
import logging
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers

import src.model
import src.data
import src.util
import src.slurm
import src.index

logger = logging.getLogger(__name__)

def flatten_embedding(
     embeddings,  # (bs, n_layer, n_heads, seq_len, emb_size_per_head)
     input_ids,  # (bs, seq_len)
     attention_mask,  # (bs, seq_len)
     ids,  # (bs,)
     tokenizer,
     results: Dict[str, List],
     head_idx: int = 0):
  bs, seq_len = input_ids.size()
  for i in range(bs):
    for j in range(seq_len):
      if not attention_mask[i, j]:
        break
      results['ids'].append(ids[i])
      results['embeddings'].append(embeddings[i, 0, head_idx, j])  # TODO: add support for multi-layer and multi-heads
      results['words'].append(tokenizer.convert_ids_to_tokens([input_ids[i, j].item()])[0])

def encode_context(opt, passages: List[Tuple[str, str, str]], model, tokenizer) -> Dict[str, np.ndarray]:
  batch_size = opt.per_gpu_batch_size
  collator = src.data.TextCollator(tokenizer, opt.passage_maxlength)
  dataset = src.data.ContextDataset(passages)
  dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=8, collate_fn=collator)
  total = 0
  results: Dict[str, List] = {'ids': [], 'embeddings': [], 'words': []}
  with torch.no_grad():
    for k, (ids, input_ids, attention_mask) in tqdm(enumerate(dataloader)):
      embeddings = model.encode_context(
        input_ids=input_ids.cuda(),
        attention_mask=attention_mask.cuda())  # (num_docs, n_layer, n_heads, seq_len, emb_size_per_head)
      flatten_embedding(embeddings, input_ids, attention_mask, ids, tokenizer, results=results, head_idx=opt.head_idx)
      total += len(ids)
      if k % 100 == 0:
        logger.info(f'encoded passages {total}')
  results['embeddings']: np.ndarray = torch.stack(results['embeddings'], dim=0).cpu().numpy()
  results['ids']: np.ndarray = np.array(results['ids'], dtype=str)
  results['words']: np.ndarray = np.array(results['words'], dtype=str)
  return results

def encode_query_and_search(opt, queries: List[Tuple[str, str]], model, tokenizer, index) -> Dict[str, List[Tuple[str, float]]]:
  batch_size = opt.per_gpu_batch_size
  collator = src.data.TextCollator(tokenizer, opt.query_maxlength)
  dataset = src.data.QuestionDataset(queries)
  dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=8, collate_fn=collator)
  total = 0
  qid2tokens2did2score: Dict[str, List[Dict[str, float]]] = defaultdict(list)
  with torch.no_grad():
    for k, (ids, input_ids, attention_mask) in tqdm(enumerate(dataloader)):
      results: Dict[str, List] = {'ids': [], 'embeddings': [], 'words': []}
      embeddings = model.encode_context(
        input_ids=input_ids.cuda(),
        attention_mask=attention_mask.cuda())  # (num_docs, n_layer, n_heads, seq_len, emb_size_per_head)
      flatten_embedding(embeddings, input_ids, attention_mask, ids, tokenizer, results=results, head_idx=opt.head_idx)
      results['embeddings']: np.ndarray = torch.stack(results['embeddings'], dim=0).cpu().numpy()
      top_ids_and_scores = index.search_knn(results['embeddings'], opt.topk)
      assert len(top_ids_and_scores) == len(results['ids'])
      for i, (docids, scores, texts) in enumerate(top_ids_and_scores):
        qid = results['ids'][i]
        qid2tokens2did2score[qid].append(defaultdict(lambda: -1e10))
        for did, score, text in zip(docids, scores, texts):
          qid2tokens2did2score[qid][-1][did] = max(score, qid2tokens2did2score[qid][-1][did])
      if k % 100 == 0:
        logger.info(f'encoded queries {total}')
  qid2did2score: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0))
  for qid, tokens in qid2tokens2did2score.items():
    for token in tokens:
      for did, score in token.items():
        qid2did2score[qid][did] += score
  qid2rank: Dict[str, List[Tuple[str, float]]] = {}
  for qid, did2score in qid2did2score.items():
    qid2rank[qid] = list(sorted(did2score.items(), key=lambda x: -x[1]))
  return qid2rank

def main(opt):
  output_path = Path(args.output_path)
  output_path.mkdir(parents=True, exist_ok=True)

  tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)
  model = src.model.FiDT5.from_pretrained(opt.model_path)
  model.eval()
  model = model.to(opt.device)
  if opt.fp16:  # this could be dangerous for certain models
    model = model.half()

  if args.queries is not None:  # querying
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, output_path / 'query.log')
    # load data
    queries = src.data.load_data(args.queries)
    queries: List[Tuple[str, str]] = [(q['id'], q['question']) for q in queries]
    logger.info(f'embedding generation for {len(queries)} queries')
    # load index
    index = src.index.Indexer(
      opt.indexing_dimension,
      n_subquantizers=opt.n_subquantizers,
      n_bits=opt.n_bits,
      hnsw_m=opt.hnsw_m)
    index.load_from_npz(args.passages, args.save_or_load_index)
    # query
    qid2rank = encode_query_and_search(opt, queries, model, tokenizer, index)
    with open(output_path / f'qid2rank.pkl', mode='wb') as f:
      pickle.dump(qid2rank, f)

  elif args.passages is not None:  # indexing
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, output_path / 'index.log')
    # load data
    passages = src.util.load_passages(args.passages)
    shard_size = len(passages) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards - 1:
      end_idx = len(passages)
    passages = passages[start_idx:end_idx]
    logger.info(f'embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}')
    # encode
    results: Dict[str, np.ndarray] = encode_context(opt, passages, model, tokenizer)
    # output
    emb_file = output_path / f'embedding_{args.shard_id:02d}.npz'
    logger.info(f'saving {len(results["ids"])} passage embeddings to {emb_file}')
    with open(emb_file, mode='wb') as f:
      np.savez(f, **results)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--queries', type=str, default=None, help='path to queries (.json file)')
  parser.add_argument('--passages', type=str, default=None, help='path to passages (.tsv file)')
  parser.add_argument('--output_path', type=str, default=None, help='prefix path to save embeddings')
  parser.add_argument('--shard_id', type=int, default=0, help='id of the current shard')
  parser.add_argument('--num_shards', type=int, default=1, help='total number of shards')
  parser.add_argument('--per_gpu_batch_size', type=int, default=32)
  parser.add_argument('--passage_maxlength', type=int, default=200, help='maximum number of tokens in a passage')
  parser.add_argument('--query_maxlength', type=int, default=50, help='maximum number of tokens in a query')
  parser.add_argument('--model_path', type=str, help='path to directory containing model weights and config file')
  parser.add_argument('--fp16', action='store_true', help='inference in fp32')
  parser.add_argument('--indexing_dimension', type=int, default=64)
  parser.add_argument('--n_subquantizers', type=int, default=0, help='number of subquantizer used for vector quantization, if 0 flat index is used')
  parser.add_argument('--n_bits', type=int, default=8, help='number of bits per subquantizer')
  parser.add_argument('--hnsw_m', type=int, default=8, help='number of bits per subquantizer')
  parser.add_argument('--save_or_load_index', action='store_true', help='if enabled, save index and load index if it exists')
  parser.add_argument('--topk', type=int, help='return top-k retrieved results')
  parser.add_argument('--head_idx', type=int, default=0, help='head idx used in retrieval')
  args = parser.parse_args()

  src.slurm.init_distributed_mode(args)
  main(args)
