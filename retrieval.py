# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import faiss
from typing import List, Tuple, Dict, Union
import argparse
from collections import defaultdict
import glob
from tqdm import tqdm
import sys
from pathlib import Path
import logging
from pathlib import Path
import pickle
from multiprocessing import Queue, Process
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, DPRContextEncoderTokenizer, DPRContextEncoder, \
  DPRQuestionEncoderTokenizer, DPRQuestionEncoder
import torch_scatter

import src.model
import src.data
import src.util
import src.slurm
import src.index

sys.path.insert(0, f'{Path.home()}/exp/ColBERT')
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint

class EmbeddingAdapter:
  def __init__(self, opt, config=None, logger = None):
    self.opt = opt
    self.num_heads = config.num_heads if config is not None else None
    self.logger = logger
    self.num_encoded = 0
    self.save_count = 0
    self._reinit_results()
  
  def __len__(self):
    return len(self.results)

  def _reinit_results(self):
    self.results: Dict[str, List] = {
      'ids': [], 
      'embeddings': [[] for _ in range(self.num_heads)] if self.opt.max_over_head else [], 
      'term_weights': [[] for _ in range(self.num_heads)] if self.opt.max_over_head else [], 
      'words': [], 
      'splits': []}
  
  def save(self, flush: bool = False):
    want_save = (flush and len(self)) or (self.opt.save_every_n_doc and self.num_encoded >= self.opt.save_every_n_doc)
    need_save = len(self) > 0
    if want_save and need_save:
      self.results['words'] = torch.cat(self.results['words'], dim=0).numpy()
      self.results['ids'] = np.array(self.results['ids'], dtype=str)
      self.results['splits'] = np.array(self.results['splits'], dtype=int)
      if self.opt.max_over_head:
        for hi, emb in enumerate(self.results['embeddings']):  # save emb of each head in a separate file
          emb = torch.cat(emb, dim=0).numpy()
          out_file = self.opt.output_path / f'embedding_{self.opt.shard_id:02d}_{self.save_count:03d}.{hi:02d}.npz'
          self.logger.info(f'saving {len(self.results["ids"])} embeddings to {out_file}')
          with open(out_file, mode='wb') as f:
            np.savez_compressed(f, embeddings=emb, **{k: self.results[k] for k in self.results if k != 'embeddings'})
      else:
        self.results['embeddings'] = torch.cat(self.results['embeddings'], dim=0).numpy()
        out_file = self.opt.output_path / f'embedding_{self.opt.shard_id:02d}_{self.save_count:03d}.npz'
        self.logger.info(f'saving {len(self.results["ids"])} embeddings to {out_file}')
        with open(out_file, mode='wb') as f:
          np.savez_compressed(f, **self.results)
      self._reinit_results()
    if want_save:
      self.num_encoded = 0
      self.save_count += 1
  
  def add_by_flatten(
      self,
      embeddings,  # (bs, n_layer, n_heads, seq_len, emb_size_per_head)
      input_ids,  # (bs, seq_len)
      attention_mask,  # (bs, seq_len)
      ids,  # (bs,)
      term_weights = None):  # (bs, n_layer, n_heads, seq_len)
    self.num_encoded += input_ids.size(0)
    bs, _, num_heads, seq_len, emb_size_ph = embeddings.size()
    # TODO: add support for multi-layer and multi-heads
    # (num_tokens, emb_size_per_head)
    if self.opt.max_over_head:  # save all heads
      for hi in range(num_heads):
        self.results['embeddings'][hi].append(torch.masked_select(embeddings[:, 0, hi], attention_mask.unsqueeze(-1)).view(-1, emb_size_ph).cpu())
        if term_weights is not None:
          self.results['term_weights'][hi].append(torch.masked_select(term_weights[:, 0, hi], attention_mask).flatten().cpu())
    else:
      self.results['embeddings'].append(torch.masked_select(embeddings[:, 0, self.opt.head_idx], attention_mask.unsqueeze(-1)).view(-1, emb_size_ph).cpu())
      if term_weights is not None:
        self.results['term_weights'].append(torch.masked_select(term_weights[:, 0, self.opt.head_idx], attention_mask).flatten().cpu())
    self.results['words'].append(torch.masked_select(input_ids, attention_mask).cpu())  # (num_tokens,)
    for i in range(bs):
      num_toks = attention_mask[i].sum().item()
      self.results['splits'].append((num_toks + self.results['splits'][-1]) if len(self.results['splits']) else num_toks)
      for _ in range(num_toks):
        self.results['ids'].append(ids[i])
  
  def add_directly(
      self,
      embeddings,  # (bs, emb_size)
      input_ids,  # (bs, seq_len)
      ids):  # (bs,)
    self.num_encoded += input_ids.size(0)
    self.results['embeddings'].append(embeddings.cpu())
    self.results['words'].append(input_ids[:, 0].cpu())  # TODO: store text
    self.results['ids'].extend(ids)

  def add_dummy(self, count: int):
    self.num_encoded += count
  
  def prepare_query(self):
    if self.opt.max_over_head:
      self.results['embeddings'] = torch.stack([torch.cat(embs, dim=0) for embs in self.results['embeddings']], dim=0).numpy()  # (num_heads, num_query_tokens_in_total, emb_size)
      if len(self.results['term_weights'][0]):
        self.results['term_weights'] = torch.stack([torch.cat(tw, dim=0) for tw in self.results['term_weights']], dim=0).numpy()  # (num_heads, num_query_tokens_in_total)
    else:
      self.results['embeddings'] = torch.cat(self.results['embeddings'], dim=0).numpy()  # (num_query_tokens_in_total, emb_size)
      if len(self.results['term_weights']):
        self.results['term_weights'] = torch.cat(self.results['term_weights'], dim=0).numpy()  # (num_query_tokens_in_total)
    self.results['words'] = torch.cat(self.results['words'], dim=0).numpy()

def encode_context(
     opt,
     passages: Union[List[Tuple[str, str, str]], np.ndarray],
     model,
     tokenizer,
     logger):
  device = model.device
  batch_size = opt.per_gpu_batch_size
  collator = src.data.TextCollator(tokenizer, opt.passage_maxlength)
  dataset = src.data.ContextDataset(passages)
  dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=opt.num_workers, collate_fn=collator)
  adapter = EmbeddingAdapter(opt, model.config if opt.model_type == 'fid' else None, logger)
  
  with torch.no_grad():
    for k, (ids, texts, input_ids, attention_mask) in tqdm(enumerate(dataloader)):
      need_compute = adapter.save_count >= opt.start_from
      if need_compute:  # compute embedding
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if opt.model_type == 'fid':
          # (num_docs, n_layer, n_heads, seq_len, emb_size_per_head)
          embeddings = model.encode_context(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_query_len=opt.query_maxlength if opt.use_position_bias else None)
          adapter.add_by_flatten(embeddings, input_ids, attention_mask, ids)
        elif opt.model_type == 'colbert':
          # (num_docs, seq_len, emb_size)
          embeddings, input_ids, attention_mask = model.docFromText(
            texts,
            keep_dims=True,
            return_more=True)
          embeddings = embeddings.unsqueeze(1).unsqueeze(1)
          attention_mask = attention_mask.bool()
          adapter.add_by_flatten(embeddings, input_ids, attention_mask, ids)
        elif opt.model_type == 'dpr':
          # (num_docs, emb_size)
          embeddings = model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
          adapter.add_directly(embeddings, input_ids, ids)
        else:
          raise NotImplementedError
      else:
        adapter.add_dummy(input_ids.size(0))
      adapter.save()
    adapter.save(flush=True)

def colbert_augmentation_postprocess(
    embeddings,  # (bs, seq_len, emb_size)
    input_ids,  # (bs, seq_len)
    attention_mask):  # (bs, seq_len)  
  attention_mask = attention_mask.to(embeddings.device)
  input_ids = input_ids.to(embeddings.device)
  bs, sl, es = embeddings.size()
  sim_mat = embeddings @ embeddings.permute(0, 2, 1)  # (bs, seq_len, seq_len)
  sim_mat = sim_mat - (attention_mask.unsqueeze(-1) | ~attention_mask.unsqueeze(1)) * 1e5
  ind = sim_mat.max(-1)[1]  # (bs, seq_len)
  emb_selected = torch.gather(embeddings.repeat(1, sl, 1).view(bs, sl, sl, es), 2, ind.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, es)).view(bs, sl, es)  # (bs, seq_len, emb_size)
  emb_selected = attention_mask.unsqueeze(-1) * embeddings + ~attention_mask.unsqueeze(-1) * emb_selected
  input_ids_selected = torch.gather(input_ids.repeat(1, sl).view(bs, sl, sl), 2, ind.unsqueeze(-1)).view(bs, sl)  # (bs, seq_len)
  input_ids_selected = attention_mask * input_ids + ~attention_mask * input_ids_selected
  return emb_selected, input_ids_selected

def encode_query_and_search(
     opt,
     queries: List[Tuple[str, str]],
     model,
     tokenizer,
     index: src.index.Indexer,
     debug: bool = False) -> Dict[str, List[Tuple[str, float]]]:
  device = model.device
  batch_size = opt.per_gpu_batch_size
  collator = src.data.TextCollator(tokenizer, opt.query_maxlength, augmentation=opt.augmentation if opt.model_type == 'fid' else None)
  dataset = src.data.QuestionDataset(queries)
  dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=opt.num_workers, collate_fn=collator)
  qid2rank: Dict[str, List[Tuple[str, float]]] = {}

  with torch.no_grad():
    for k, (ids, texts, input_ids, attention_mask) in tqdm(enumerate(dataloader)):
      # generate query embeddings
      adapter = EmbeddingAdapter(opt, model.config if opt.model_type == 'fid' else None)
      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)
      if opt.model_type == 'fid':
        # (num_queries, n_layer, n_heads, seq_len, emb_size_per_head)
        embeddings, term_weights = model.encode_query(
          input_ids=input_ids,
          attention_mask=attention_mask,
          max_query_len=opt.query_maxlength if opt.use_position_bias else None)
        if opt.normalize:
          #norm = (embeddings * embeddings).sum(-1).sqrt()[:, 0, opt.head_idx].var(dim=-1).mean()
          embeddings = F.normalize(embeddings, dim=-1, p=2)
        adapter.add_by_flatten(embeddings, input_ids, attention_mask, ids, term_weights=term_weights)
      elif opt.model_type == 'dpr':
        # (num_queries, emb_size)
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        adapter.add_directly(embeddings, input_ids, ids)
      elif opt.model_type == 'colbert':
        # (num_docs, seq_len, emb_size)
        embeddings, input_ids, attention_mask = model.queryFromText(texts, return_more=True)
        attention_mask = attention_mask.bool()
        if opt.augmentation == 'mask-select':
          embeddings, input_ids = colbert_augmentation_postprocess(embeddings, input_ids, attention_mask)
        if opt.augmentation in {'mask', 'mask-select'}:  # colbert augmentation
          attention_mask = torch.ones_like(attention_mask)
        embeddings = embeddings.unsqueeze(1).unsqueeze(1)
        adapter.add_by_flatten(embeddings, input_ids, attention_mask, ids)
      else:
        raise NotImplementedError
      # search and aggregation
      adapter.prepare_query()
      if opt.model_type in {'fid', 'colbert'}:
        _qid2rank = index.search_knn(
          adapter.results['embeddings'], 
          opt.token_topk, 
          term_weights=adapter.results['term_weights'] if opt.term_weights else None,
          query_ids=adapter.results['ids'],
          query_splits=adapter.results['splits'],
          rank_topk=opt.doc_topk,
          rerank_topk=opt.candidate_doc_topk,
          device=device)
        qid2rank.update(_qid2rank)
      elif opt.model_type == 'dpr':
        top_ids_and_scores = index.search_knn(adapter.results['embeddings'], opt.doc_topk)
        for i, (docids, scores, texts) in enumerate(top_ids_and_scores):
          qid = adapter.results['ids'][i]
          did2score: Dict[str, float] =  defaultdict(lambda: 0)
          for did, score, text in zip(docids, scores, texts):
            did2score[did] = float(score)
          qid2rank[qid] = list(sorted(did2score.items(), key=lambda x: -x[1]))
      else:
        raise NotImplementedError
  return qid2rank

def get_model_tokenizer(opt, is_querying: bool):
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
  if opt.fp16:  # this could be dangerous for certain models
    model = model.half()
  return model, tokenizer

def run_query(input_queue: Queue, opt, cuda_device: Union[int, List[int]]):
  # log
  logger = src.util.init_logger()
  # load model
  main_device = torch.device(f'cuda:{cuda_device[0]}') if type(cuda_device) is list else torch.device(f'cuda:{cuda_device}')
  model, tokenizer = get_model_tokenizer(opt, is_querying=True)
  model = model.to(main_device)
  # activate term weights based on model config
  opt.term_weights = opt.model_type == 'fid' and model.config.term_weight_parameter
  # load query data
  queries = src.data.load_data(opt.queries)
  queries: List[Tuple[str, str]] = [(q['id'], q['question']) for q in queries]
  if opt.debug:
    queries = queries[:512]
  logger.info(f'#queries: {len(queries)}')
  while True:
    batch = input_queue.get()
    if type(batch) is str and batch == 'done':
      break
    batch_index, batch_files = batch
    # load index
    index_params = {
      'vector_sz': opt.indexing_dimension, 
      'n_subquantizers': opt.n_subquantizers, 
      'n_bits': opt.n_bits, 
      'hnsw_m': opt.hnsw_m, 
      'keep_raw_vector': True, 
      'normalize': opt.normalize,
      'cuda_device': cuda_device}
    if opt.max_over_head:
      index = src.index.MultiIndexer(num_indices=model.config.num_heads, **index_params)
    else:
      index = src.index.Indexer(**index_params)
    index_name = 'index' if opt.save_or_load_index else None
    index.load_from_npz(batch_files, index_name=index_name)
    # query
    qid2rank = encode_query_and_search(opt, queries, model, tokenizer, index)
    if opt.model_type in {'fid', 'colbert'}:
      rank_file = opt.output_path / (f'qid2rank_{opt.token_topk}' + \
        (f'_{opt.candidate_doc_topk}' if opt.candidate_doc_topk else '') + \
        (f'_{opt.augmentation}' if opt.augmentation else '') + \
        (f'_norm' if opt.normalize else '') + \
        (f'_termweights' if opt.term_weights else '') + \
        (f'.{batch_index}' if batch_index >= 0 else '') + '.pkl')
    elif opt.model_type == 'dpr':
      rank_file = opt.output_path / f'qid2rank_{opt.doc_topk}.pkl'
    else:
      raise NotImplementedError
    with open(rank_file, mode='wb') as f:
      pickle.dump(qid2rank, f)

def main(opt):
  # append head idx to the output path
  if opt.max_over_head:
    opt.output_path = opt.output_path + '.maxoverhead'
  else:
    opt.output_path = opt.output_path + f'.{opt.head_idx}'
  opt.output_path = Path(opt.output_path)
  opt.output_path.mkdir(parents=True, exist_ok=True)
  is_querying = opt.queries is not None

  if is_querying:  # querying
    emb_file_pattern = str(args.output_path) + '/embedding_*.npz'
    emb_files = glob.glob(emb_file_pattern)
    emb_files = sorted(emb_files)
    if opt.max_over_head:  # TODO: we assume #files == #heads
      opt.files_per_run = len(emb_files)
    if len(emb_files) > opt.files_per_run:
      opt.save_or_load_index = False

    # launch process
    processes: List[Process] = []
    input_queue: Queue = Queue()
    if opt.faiss_gpus == 'all':  # use a single process and all gpu
      p = Process(target=run_query, args=(input_queue, opt, [gpu for gpu in range(torch.cuda.device_count())]))
      processes.append(p)
    elif opt.faiss_gpus == 'separate':  # ues multiple processes with a single gpu assigned to each of them
      for gpu in range(torch.cuda.device_count()):
        p = Process(target=run_query, args=(input_queue, opt, gpu))
        processes.append(p)
    else:  # single process with a single gpu
      p = Process(target=run_query, args=(input_queue, opt, int(opt.faiss_gpus)))
      processes.append(p)
    for p in processes:
      p.daemon = True
      p.start()
    
    if len(emb_files) > opt.files_per_run:  # iterative over emb files
      for batch_index, file_start in enumerate(range(0, len(emb_files), opt.files_per_run)):
        batch_files = emb_files[file_start:file_start + opt.files_per_run]
        input_queue.put((batch_index, batch_files))
    else:  # use all emb files at once
      input_queue.put((-1, emb_files))
    
    # finish
    for _ in range(len(processes)):
      input_queue.put('done')
    for p in processes:
      p.join()  
 
  else:  # indexing
    # load model
    model, tokenizer = get_model_tokenizer(opt, is_querying=is_querying)
    model = model.cuda()
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, opt.output_path / f'index{opt.shard_id}.log')
    # load data
    passages = next(src.util.load_passages(opt.passages))
    shard_size = len(passages) // opt.num_shards
    start_idx = opt.shard_id * shard_size
    end_idx = start_idx + shard_size
    if opt.shard_id == opt.num_shards - 1:
      end_idx = len(passages)
    passages = passages[start_idx:end_idx]
    logger.info(f'embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}')
    # encode and output
    encode_context(opt, passages, model, tokenizer, logger)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_type', type=str, default='fid', help='type of models', choices=['fid', 'dpr', 'colbert'])
  parser.add_argument('--queries', type=str, default=None, help='path to queries (.json file)')
  parser.add_argument('--passages', type=str, default=None, help='path to passages (.tsv file)')
  parser.add_argument('--output_path', type=str, default=None, help='prefix path to save embeddings')
  parser.add_argument('--shard_id', type=int, default=0, help='id of the current shard')
  parser.add_argument('--num_shards', type=int, default=1, help='total number of shards')
  parser.add_argument('--per_gpu_batch_size', type=int, default=32)
  parser.add_argument('--files_per_run', type=int, default=1, help='embedding files to load for each run')
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
  parser.add_argument('--candidate_doc_topk', type=int, default=0, help='top-k retrieved documents for reranking')
  parser.add_argument('--head_idx', type=int, default=0, help='head idx used in retrieval')
  parser.add_argument('--faiss_gpus', type=str, default='-1', help='gpu indices for faiss or all')
  parser.add_argument('--use_position_bias', action='store_true', help='use position bias')
  parser.add_argument('--max_over_head', action='store_true', help='use all head and max to aggregate')
  parser.add_argument('--augmentation', type=str, help='query augmentation', default=None, choices=[None, 'duplicate', 'mask', 'mask-select'])
  parser.add_argument('--save_every_n_doc', type=int, help='#doc to accumulate before saving. 0 means only saving after encoding all docs', default=0)
  parser.add_argument('--start_from', type=int, help='start from index which is used together with save_every_n_doc', default=0)
  parser.add_argument('--num_workers', type=int, help='#workers for dataloader', default=0)
  parser.add_argument('--normalize', action='store_true', help='normalize query and doc embedding')
  parser.add_argument('--debug', action='store_true')
  args = parser.parse_args()

  src.slurm.init_distributed_mode(args)
  src.util.global_context['opt'] = args
  main(args)
