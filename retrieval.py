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

def flatten_embedding(
     embeddings,  # (bs, n_layer, n_heads, seq_len, emb_size_per_head)
     input_ids,  # (bs, seq_len)
     attention_mask,  # (bs, seq_len)
     ids,  # (bs,)
     results: Dict[str, List],
     head_idx: int = 0,
     max_over_head: bool = False):
  bs, _, num_heads, seq_len, emb_size_ph = embeddings.size()
  # TODO: add support for multi-layer and multi-heads
  # (num_tokens, emb_size_per_head)
  if max_over_head:  # save all heads
    for hi in range(num_heads):
      results['embeddings'][hi].append(torch.masked_select(embeddings[:, 0, hi], attention_mask.unsqueeze(-1)).view(-1, emb_size_ph).cpu())
  else:
    results['embeddings'].append(torch.masked_select(embeddings[:, 0, head_idx], attention_mask.unsqueeze(-1)).view(-1, emb_size_ph).cpu())
  results['words'].append(torch.masked_select(input_ids, attention_mask).cpu())  # (num_tokens,)
  for i in range(bs):
    num_toks = attention_mask[i].sum().item()
    results['splits'].append((num_toks + results['splits'][-1]) if len(results['splits']) else num_toks)
    for _ in range(num_toks):
      results['ids'].append(ids[i])

def encode_context(
     opt,
     passages: Union[List[Tuple[str, str, str]], np.ndarray],
     model,
     tokenizer):
  device = model.device
  batch_size = opt.per_gpu_batch_size
  collator = src.data.TextCollator(tokenizer, opt.passage_maxlength)
  dataset = src.data.ContextDataset(passages)
  dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=opt.num_workers, collate_fn=collator)
  
  def _init_results():
    results: Dict[str, List] = {'ids': [], 'embeddings': [[] for _ in range(model.config.num_heads)] if opt.max_over_head else [], 'words': [], 'splits': []}
    return results
  
  results = _init_results()
  num_encoded_doc = 0
  save_count = 0
  
  def _save():
    results['words'] = torch.cat(results['words'], dim=0).numpy()
    results['ids'] = np.array(results['ids'], dtype=str)
    results['splits'] = np.array(results['splits'], dtype=int)
    if opt.max_over_head:
      for hi, emb in enumerate(results['embeddings']):  # save emb of each head in a separate file
        emb = torch.cat(emb, dim=0).numpy()
        out_file = opt.output_path / f'embedding_{opt.shard_id:02d}_{save_count:03d}.{hi:02d}.npz'
        logger.info(f'saving {len(results["ids"])} embeddings to {out_file}')
        with open(out_file, mode='wb') as f:
          np.savez_compressed(f, embeddings=emb, **{k: results[k] for k in results if k != 'embeddings'})
    else:
      results['embeddings'] = torch.cat(results['embeddings'], dim=0).numpy()
      out_file = opt.output_path / f'embedding_{opt.shard_id:02d}_{save_count:03d}.npz'
      logger.info(f'saving {len(results["ids"])} embeddings to {out_file}')
      with open(out_file, mode='wb') as f:
        np.savez_compressed(f, **results)

  with torch.no_grad():
    for k, (ids, texts, input_ids, attention_mask) in tqdm(enumerate(dataloader)):
      num_encoded_doc += input_ids.size(0)
      need_compute = save_count >= opt.start_from
      if need_compute:  # compute embedding
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if opt.model_type == 'fid':
          # (num_docs, n_layer, n_heads, seq_len, emb_size_per_head)
          embeddings = model.encode_context(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_query_len=opt.query_maxlength if opt.use_position_bias else None)
          flatten_embedding(embeddings, input_ids, attention_mask, ids, results=results, head_idx=opt.head_idx, max_over_head=opt.max_over_head)
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
          # TODO: add splits?
        else:
          raise NotImplementedError
      if opt.save_every_n_doc and num_encoded_doc >= opt.save_every_n_doc:  # output
        if need_compute:
          _save()
          results = _init_results()
        save_count += 1
        num_encoded_doc = 0
    if len(results['ids']) > 0:
      _save()
      results = _init_results()
      save_count += 1
      num_encoded_doc = 0

def encode_query_and_search(
     opt,
     queries: List[Tuple[str, str]],
     model,
     tokenizer,
     index: src.index.Indexer,
     debug: bool = False) -> Dict[str, List[Tuple[str, float]]]:
  device = model.device
  batch_size = opt.per_gpu_batch_size
  collator = src.data.TextCollator(tokenizer, opt.query_maxlength, augmentation=opt.augmentation)
  dataset = src.data.QuestionDataset(queries)
  dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=opt.num_workers, collate_fn=collator)
  qid2rank: Dict[str, List[Tuple[str, float]]] = {}
  with torch.no_grad():
    for k, (ids, texts, input_ids, attention_mask) in tqdm(enumerate(dataloader)):
      results: Dict[str, List] = {'ids': [], 'embeddings': [], 'words': [], 'splits': []}
      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)
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
      results['embeddings'] = torch.cat(results['embeddings'], dim=0).numpy()
      results['words'] = torch.cat(results['words'], dim=0).numpy()
      results['ids'] = np.array(results['ids'], dtype=str)
      results['splits'] = np.array(results['splits'], dtype=int)

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
      
      for i, (qid, did2score) in enumerate(qid2did2score.items()):  # rank and only keep top docs
        qid2rank[qid] = list(sorted(did2score.items(), key=lambda x: -x[1]))
        if opt.candidate_doc_topk <= 0:
          qid2rank[qid] = qid2rank[qid][:opt.doc_topk]
          continue
        dids = [did for did, _ in qid2rank[qid]][:opt.candidate_doc_topk]
        flat_embs, flat_dids = index.get_by_ids(dids)  # (num_doc_tok_in_total, [num_heads,] emb_size), (num_doc_tok_in_total)
        qry_embs = results['embeddings'][results['splits'][i - 1] if i else 0:results['splits'][i]]
        qry_embs = torch.tensor(qry_embs).to(device)
        flat_embs = torch.tensor(flat_embs).to(device)
        if opt.max_over_head:
          ndt, nh, es = flat_embs.size()
          sim_mat = (qry_embs @ flat_embs.view(-1, es).T).view(-1, ndt, nh).max(-1)[0]  # (num_qry_tok, num_doc_tok_in_total)
        else:
          sim_mat = (qry_embs @ flat_embs.T)  # (num_qry_tok, num_doc_tok_in_total)
        did2newdid: Dict[str, int] = {}
        newdid2did: Dict[int, str] = {}
        assert len(dids) == len(set(dids))
        for did in dids:
          if did not in did2newdid:
            did2newdid[did] = len(did2newdid)
            newdid2did[did2newdid[did]] = did
        flat_dids = torch.tensor([did2newdid[did] for did in flat_dids]).to(device)
        agg_sim_mat = torch.zeros(sim_mat.size(0), len(did2newdid)).to(sim_mat)
        agg_sim_mat = torch_scatter.scatter_max(sim_mat, flat_dids, out=agg_sim_mat, dim=-1)[0]  # (num_qry_tok, num_doc)
        agg_sim_mat = agg_sim_mat.sum(0)  # (num_doc)
        qid2rank[qid] = sorted([(newdid2did[nd], score.item()) for nd, score in enumerate(agg_sim_mat)], key=lambda x: -x[1])[:opt.doc_topk]

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
  # load query data
  queries = src.data.load_data(opt.queries)
  queries: List[Tuple[str, str]] = [(q['id'], q['question']) for q in queries]
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
        (f'.{batch_index}' if batch_index >= 0 else '') + '.pkl')
    elif opt.model_type == 'dpr':
      rank_file = opt.output_path / f'qid2rank_{opt.doc_topk}.pkl'
    else:
      raise NotImplementedError
    with open(rank_file, mode='wb') as f:
      pickle.dump(qid2rank, f)

def main(opt):
  if args.max_over_head:
    args.output_path = args.output_path + '.maxoverhead'
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
    encode_context(opt, passages, model, tokenizer)

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
  parser.add_argument('--augmentation', type=str, help='query augmentation', default=None, choices=[None, 'duplicate', 'mask'])
  parser.add_argument('--save_every_n_doc', type=int, help='#doc to accumulate before saving. 0 means only saving after encoding all docs', default=0)
  parser.add_argument('--start_from', type=int, help='start from index which is used together with save_every_n_doc', default=0)
  parser.add_argument('--num_workers', type=int, help='#workers for dataloader', default=0)
  args = parser.parse_args()

  src.slurm.init_distributed_mode(args)
  src.util.global_context['opt'] = args
  main(args)
