# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import faiss
from typing import List, Tuple, Dict
import argparse
from collections import defaultdict
from tqdm import tqdm
import sys
from pathlib import Path
import logging
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, DPRContextEncoderTokenizer, DPRContextEncoder, \
  DPRQuestionEncoderTokenizer, DPRQuestionEncoder

import src.model
import src.data
import src.util
import src.slurm
import src.index

sys.path.insert(0, f'{Path.home()}/exp/ColBERT')
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint

logger = logging.getLogger(__name__)

def flatten_embedding(
     embeddings,  # (bs, n_layer, n_heads, seq_len, emb_size_per_head)
     input_ids,  # (bs, seq_len)
     attention_mask,  # (bs, seq_len)
     ids,  # (bs,)
     results: Dict[str, List],
     head_idx: int = 0):
  bs, _, _, seq_len, emb_size_ph = embeddings.size()
  # TODO: add support for multi-layer and multi-heads
  # (num_tokens, emb_size_per_head)
  results['embeddings'].append(torch.masked_select(embeddings[:, 0, head_idx], attention_mask.unsqueeze(-1)).view(-1, emb_size_ph).cpu())
  results['words'].append(torch.masked_select(input_ids, attention_mask).cpu())  # (num_tokens,)
  for i in range(bs):
    for j in range(attention_mask[i].sum()):
      results['ids'].append(ids[i])

def encode_context(
     opt,
     passages: List[Tuple[str, str, str]],
     model,
     tokenizer) -> Dict[str, np.ndarray]:
  batch_size = opt.per_gpu_batch_size
  collator = src.data.TextCollator(tokenizer, opt.passage_maxlength)
  dataset = src.data.ContextDataset(passages)
  dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=8, collate_fn=collator)
  results: Dict[str, List] = {'ids': [], 'embeddings': [], 'words': []}
  with torch.no_grad():
    for k, (ids, texts, input_ids, attention_mask) in tqdm(enumerate(dataloader)):
      input_ids = input_ids.cuda()
      attention_mask = attention_mask.cuda()
      if opt.model_type == 'fid':
        # (num_docs, n_layer, n_heads, seq_len, emb_size_per_head)
        embeddings = model.encode_context(
          input_ids=input_ids,
          attention_mask=attention_mask,
          max_query_len=opt.query_maxlength if opt.use_position_bias else None)
        flatten_embedding(embeddings, input_ids, attention_mask, ids, results=results, head_idx=opt.head_idx)
      elif opt.model_type == 'colbert':
        # (num_docs, seq_len, emb_size)
        embeddings, input_ids, attention_mask = model.docFromText(
          texts,
          keep_dims=True,
          return_more=True)
        embeddings = embeddings.unsqueeze(1).unsqueeze(1)
        attention_mask = attention_mask.bool()
        flatten_embedding(embeddings, input_ids, attention_mask, ids, results=results, head_idx=0)
      elif opt.model_type == 'dpr':
        # (num_docs, emb_size)
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        results['embeddings'].append(embeddings.cpu())
        results['words'].append(input_ids[:, 0].cpu())  # TODO: store text
        results['ids'].extend(ids)
      else:
        raise NotImplementedError
  results['embeddings']: np.ndarray = torch.cat(results['embeddings'], dim=0).numpy()
  results['words']: np.ndarray = torch.cat(results['words'], dim=0).numpy()
  results['ids']: np.ndarray = np.array(results['ids'], dtype=str)
  return results

def encode_query_and_search(
     opt,
     queries: List[Tuple[str, str]],
     model,
     tokenizer,
     index: src.index.Indexer,
     debug: bool = False) -> Dict[str, List[Tuple[str, float]]]:
  batch_size = opt.per_gpu_batch_size
  collator = src.data.TextCollator(tokenizer, opt.query_maxlength)
  dataset = src.data.QuestionDataset(queries)
  dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=8, collate_fn=collator)
  qid2rank: Dict[str, List[Tuple[str, float]]] = {}
  with torch.no_grad():
    for k, (ids, texts, input_ids, attention_mask) in tqdm(enumerate(dataloader)):
      results: Dict[str, List] = {'ids': [], 'embeddings': [], 'words': []}
      input_ids = input_ids.cuda()
      attention_mask = attention_mask.cuda()
      if opt.model_type == 'fid':
        # (num_queries, n_layer, n_heads, seq_len, emb_size_per_head)
        embeddings = model.encode_query(
          input_ids=input_ids,
          attention_mask=attention_mask,
          max_query_len=opt.query_maxlength if opt.use_position_bias else None)
        flatten_embedding(embeddings, input_ids, attention_mask, ids, results=results, head_idx=opt.head_idx)
      elif opt.model_type == 'dpr':
        # (num_queries, emb_size)
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        results['embeddings'].append(embeddings.cpu())
        results['words'].append(input_ids[:, 0].cpu())  # TODO: store text
        results['ids'].extend(ids)
      elif opt.model_type == 'colbert':
        # (num_docs, seq_len, emb_size)
        embeddings, input_ids, attention_mask = model.queryFromText(texts, return_more=True)
        embeddings = embeddings.unsqueeze(1).unsqueeze(1)
        attention_mask = attention_mask.bool()
        flatten_embedding(embeddings, input_ids, attention_mask, ids, results=results, head_idx=0)
      else:
        raise NotImplementedError
      results['embeddings']: np.ndarray = torch.cat(results['embeddings'], dim=0).numpy()
      results['words']: np.ndarray = torch.cat(results['words'], dim=0).numpy()
      results['ids']: np.ndarray = np.array(results['ids'], dtype=str)

      qid2did2score: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0))
      if opt.model_type in {'fid', 'colbert'}:
        top_ids_and_scores = index.search_knn(results['embeddings'], opt.token_topk)
        qid2tokens2did2score: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for i, (docids, scores, texts) in enumerate(top_ids_and_scores):  # token-level scores
          qid = results['ids'][i]
          qid2tokens2did2score[qid].append(defaultdict(lambda: -1e10))
          for did, score, text in zip(docids, scores, texts):
            if debug:
              qword = tokenizer.convert_ids_to_tokens([results['words'][i]])[0]
              dword = tokenizer.convert_ids_to_tokens([text])[0]
              print(qword, dword, score, did)
              input()
            qid2tokens2did2score[qid][-1][did] = max(score, qid2tokens2did2score[qid][-1][did])
        for qid, tokens in qid2tokens2did2score.items():  # aggregate token-level scores
          for token in tokens:
            for did, score in token.items():
              qid2did2score[qid][did] += score
      elif opt.model_type == 'dpr':
        top_ids_and_scores = index.search_knn(results['embeddings'], opt.doc_topk)
        for i, (docids, scores, texts) in enumerate(top_ids_and_scores):
          qid = results['ids'][i]
          for did, score, text in zip(docids, scores, texts):
            qid2did2score[qid][did] = score
      else:
        raise NotImplementedError
      
      for qid, did2score in qid2did2score.items():  # rank and only keep top docs
        qid2rank[qid] = list(sorted(did2score.items(), key=lambda x: -x[1]))[:opt.doc_topk]

  return qid2rank

def main(opt):
  output_path = Path(args.output_path)
  output_path.mkdir(parents=True, exist_ok=True)
  is_querying = args.queries is not None

  if opt.model_type == 'fid':
    tokenizer = T5Tokenizer.from_pretrained('t5-base', return_dict=False)
    model = src.model.FiDT5.from_pretrained(opt.model_path)
  elif opt.model_type == 'dpr':
    if is_querying:
      tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(opt.model_path, return_dict=False)
      model = DPRQuestionEncoder.from_pretrained(opt.model_path)
    else:
      tokenizer = DPRContextEncoderTokenizer.from_pretrained(opt.model_path, return_dict=False)
      model = DPRContextEncoder.from_pretrained(opt.model_path)
  elif opt.model_type == 'colbert':
    tokenizer = T5Tokenizer.from_pretrained('t5-base', return_dict=False)  # placeholder
    with Run().context(RunConfig(nranks=1, experiment='test')):
      ckpt_config = ColBERTConfig.load_from_checkpoint(opt.model_path)
      config = ColBERTConfig(doc_maxlen=opt.passage_maxlength, nbits=2)
      config = ColBERTConfig.from_existing(ckpt_config, config, Run().config)
      model = Checkpoint(opt.model_path, colbert_config=config)
  else:
    raise NotImplementedError
  model.eval()
  model = model.to(opt.device)
  if opt.fp16:  # this could be dangerous for certain models
    model = model.half()

  if is_querying:  # querying
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
      hnsw_m=opt.hnsw_m,
      cuda_device=0 if opt.use_faiss_gpu else -1)
    index.load_from_npz(args.passages, args.save_or_load_index)
    # query
    qid2rank = encode_query_and_search(opt, queries, model, tokenizer, index)
    if opt.model_type in {'fid', 'colbert'}:
      rank_file = output_path / f'qid2rank_{opt.token_topk}.pkl'
    elif opt.model_type == 'dpr':
      rank_file = output_path / f'qid2rank_{opt.doc_topk}.pkl'
    else:
      raise NotImplementedError
    with open(rank_file, mode='wb') as f:
      pickle.dump(qid2rank, f)

  else:  # indexing
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
  parser.add_argument('--model_type', type=str, default='fid', help='type of models', choices=['fid', 'dpr', 'colbert'])
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
  parser.add_argument('--token_topk', type=int, help='return top-k retrieved tokens')
  parser.add_argument('--doc_topk', type=int, help='return top-k retrieved documents')
  parser.add_argument('--head_idx', type=int, default=0, help='head idx used in retrieval')
  parser.add_argument('--use_faiss_gpu', action='store_true', help='use faiss gpu')
  parser.add_argument('--use_position_bias', action='store_true', help='use position bias')
  args = parser.parse_args()

  src.slurm.init_distributed_mode(args)
  src.util.global_context['opt'] = args
  main(args)
