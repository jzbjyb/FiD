# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import enum
import logging
from operator import index
from re import I
from typing import List, Tuple, Set, Union, Dict
import time
import glob
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import faiss
import numpy as np
import torch
import torch_scatter

from src.util import logger
from src.strided_tensor import StridedTensor

class Indexer(object):
  def __init__(self,
               vector_sz: int,
               n_subquantizers: int = 0,
               n_bits: int = 8,
               hnsw_m: int = 0,
               cuda_device: Union[int, List[int]] = -1,
               keep_raw_vector: bool = False,
               normalize: bool = False):
    self.cuda_device = cuda_device
    self.vector_sz = vector_sz
    self.n_subquantizers = n_subquantizers
    self.n_bits = n_bits
    self.hnsw_m = hnsw_m
    self.keep_raw_vector = keep_raw_vector
    self.shard = False  # shard index across multiple gpus
    self.normalize = normalize
    self.create_index()
    self.ids = np.empty((0), dtype=str)
    self.texts = np.empty((0), dtype=str)
    self.embs = torch.zeros(0, vector_sz)

  def load_from_npz(
      self, 
      emb_files: List[str], 
      index_name: str = None):
    root_dir = Path(emb_files[0]).parent
    if index_name is not None and (root_dir / f'{index_name}.faiss').exists():
      self.deserialize_from(root_dir, index_name=index_name)
    else:
      logger.info(f'indexing passages from files {emb_files}')
      start_time_indexing = time.time()
      ids, embeddings, texts = [], [], []
      for i, emb_file in enumerate(emb_files):
        logger.info(f'loading file {emb_file}')
        with open(emb_file, 'rb') as fin:
          npzfile = np.load(fin)
          _ids, _embeddings, _texts = npzfile['ids'], npzfile['embeddings'], npzfile['words']
          ids.append(_ids)
          embeddings.append(_embeddings)
          texts.append(_texts)
      ids = np.concatenate(ids, axis=0)
      embeddings = np.concatenate(embeddings, axis=0)
      texts = np.concatenate(texts, axis=0)
      self.index_data(ids, embeddings, texts)
      logger.info(f'data indexing completed with time {time.time() - start_time_indexing:.1f} s.')
      if index_name is not None:
        self.serialize(root_dir, index_name=index_name)
    self.build_int_ids()
    self.build_strided()
    
  def serialize(
      self, 
      dir_path: Path, 
      index_name: str = 'index'):
    index_file = dir_path / f'{index_name}.faiss'
    meta_file = dir_path / f'{index_name}.meta'
    logger.info(f'Serializing index to {index_file}, meta data to {meta_file}')
    faiss.write_index(faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index, str(index_file))
    with open(meta_file, mode='wb') as f:
      np.savez(f, ids=self.ids, texts=self.texts, embs=self.embs)

  def deserialize_from(
      self, 
      dir_path: Path,
      index_name: str = 'index'):
    index_file = dir_path / f'{index_name}.faiss'
    meta_file = dir_path / f'{index_name}.meta'
    logger.info(f'Loading index from {index_file}, meta data from {meta_file}')
    self.index = faiss.read_index(str(index_file))
    self.move_to_gpu()
    logger.info('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)
    with open(meta_file, 'rb') as reader:
      npzfile = np.load(reader)
      self.ids, self.texts = npzfile['ids'], npzfile['texts']
      if 'embs' in npzfile:
        self.embs = npzfile['embs']
    assert len(self) == self.index.ntotal, 'Deserialized ids should match faiss index size'
  
  def __len__(self):
    return len(self.ids)
  
  @property
  def use_gpu(self):
    if type(self.cuda_device) is int and self.cuda_device == -1:
      return False
    return True

  @property
  def main_gpu(self):
    if self.use_gpu:
      if type(self.cuda_device) is list:
        return self.cuda_device[0]
      return self.cuda_device
    raise ValueError

  def create_index(self):
    metric = faiss.METRIC_INNER_PRODUCT
    if self.n_subquantizers > 0:
      self.index = faiss.IndexPQ(self.vector_sz, self.n_subquantizers, self.n_bits, metric)
    elif self.hnsw_m > 0:
      self.index = faiss.IndexHNSWFlat(self.vector_sz, self.hnsw_m, metric)
    else:
      self.index = faiss.IndexFlatIP(self.vector_sz)
      #quantizer = faiss.IndexFlatIP(self.vector_sz)
      #self.index = faiss.IndexIVFPQ(quantizer, self.vector_sz, 16384, 32, 8)
      #self.index.nprobe = 8
    self.move_to_gpu()

  def move_to_gpu(self):
    if not self.use_gpu:
      return
    if type(self.cuda_device) is list:  # multiple gpu
      logger.info(f'Move FAISS index to gpu {self.cuda_device}')
      if self.shard:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.shard = True
      else:
        co = None
      self.index = faiss.index_cpu_to_gpus_list(self.index, co=co, gpus=self.cuda_device)
    else:  # single gpu
      logger.info(f'Move FAISS index to gpu {self.cuda_device}')
      res = faiss.StandardGpuResources()
      self.index = faiss.index_cpu_to_gpu(res, self.cuda_device, self.index)

  def index_data(self,
                 ids: np.ndarray,
                 embeddings: np.ndarray,
                 texts: np.ndarray = None,
                 indexing_batch_size: int = None,
                 disable_log: bool = False):
      assert len(ids) == len(embeddings)
      if texts is not None:
        assert len(ids) == len(texts)
      embeddings = embeddings.astype('float32')
      if self.normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
      if not self.index.is_trained:
        self.index.train(embeddings)
      if indexing_batch_size is None:
        self.index.add(embeddings)
      else:
        for b in tqdm(range(0, len(embeddings), indexing_batch_size), desc='indexing', disable=disable_log):
          batch = embeddings[b:b + indexing_batch_size]
          #implicit_ids = np.arange(len(self) + b, len(self) + b + len(batch))
          #self.index.add_with_ids(batch, implicit_ids)
          self.index.add(batch)
      self._update_id_mapping(ids)
      self._update_texts(texts)
      self._update_emb(embeddings)
      if not disable_log:
        logger.info(f'total data indexed {len(self)}')
  
  def build_strided(self):
    self.embs_strided = StridedTensor(self.embs, torch.tensor(self.doclens).to(self.main_gpu))
  
  def build_int_ids(self):
    '''
    # np.unique will sort
    self.unique_ids, self.order, self.ids_int = np.unique(self.ids, return_index=True, return_inverse=True)
    self.order = np.unique(self.order, return_inverse=True)[1]  # used in strided tensor
    '''

    self.unique_ids: List[str] = []
    self.doclens: List[int] = []
    self.ids_int: List[str] = []
    
    id2offset: Dict[int, int] = {}
    id2len: Dict[int, int] = {}
    
    for i, _id in enumerate(self.ids):
      if _id not in id2offset:
        id2offset[_id] = i
        id2len[_id] = 1
        self.unique_ids.append(_id)
        self.doclens.append(1)
        self.ids_int.append(len(self.unique_ids) - 1)
      else:
        assert i == id2offset[_id] + id2len[_id]  # make sure ids are consecutive
        id2len[_id] += 1
        self.doclens[-1] += 1
        self.ids_int.append(len(self.unique_ids) - 1)
    
    assert len(self.unique_ids) == len(self.doclens)
    self.unique_ids = np.array(self.unique_ids)
    self.doclens = np.array(self.doclens)
    self.ids_int = np.array(self.ids_int)

  def remove_data(self, num_unique_id: int):
    unique_ids: Set[str] = set()
    stop = False
    for i, id in enumerate(self.ids):
      if len(unique_ids) == num_unique_id and id not in unique_ids:  # do not need to remove all
        stop = True
        break
      unique_ids.add(id)
    if not stop:  # remove all
      i = len(self)
    try:
      self.index.remove_ids(np.arange(i))
    except:  # remove is not supported for some index
      assert i <= self.index.ntotal
      keep = self.index.reconstruct_n(i, self.index.ntotal - i)
      del self.index
      self.create_index()
      self.index.add(keep)
    self.ids = self.ids[i:]
    self.texts = self.texts[i:]
    self.embs = self.embs[i:]
    assert len(self) == self.index.ntotal
  
  def compose_rank_list(self, dids: torch.LongTensor, scores: torch.FloatTensor):
    dids = self.unique_ids[dids.cpu().numpy()]
    return [(str(did), score.item()) for did, score in zip(dids, scores)]

  def search_knn(
    self,
    query_vectors: np.array,  # (nq_flat, emb_size)
    topk: int,
    batch_size: int = 1024,
    return_external_id: bool = True,
    term_weights: np.array = None,  # (nq_flat,)
    query_ids: List[int] = None,  # (nq_flat,)
    query_splits: List[int] = None,  # (uqq,)
    rank_topk: int = 0,
    rerank_topk: int = 0,
    device: torch.device = None) -> List[Tuple[List[Union[str, int]], List[float], List[str]]]:
    # TODO: directly use torch tensors with "import faiss.contrib.torch_utils"

    if term_weights is not None:
      assert term_weights.shape[0] == query_vectors.shape[0]
    query_vectors = query_vectors.astype('float32')
    
    scores_flat, doc_ids_flat = [], []
    batch_size = batch_size or len(query_vectors)
    nbatch = (len(query_vectors) - 1) // batch_size + 1
    
    for k in range(nbatch):
      start_idx = k * batch_size
      end_idx = min(start_idx + batch_size, len(query_vectors))
      qv = query_vectors[start_idx:end_idx]
      _scores_flat, _doc_ids_flat = self.index.search(qv, topk)
      if term_weights is not None:
        tw = term_weights[start_idx:end_idx]
        _scores_flat = _scores_flat * np.expand_dims(tw, axis=-1)
      scores_flat.append(_scores_flat)
      doc_ids_flat.append(_doc_ids_flat)
    
    scores_flat = np.concatenate(scores_flat, axis=0)  # (nq_flat, topk)
    doc_ids_flat = np.concatenate(doc_ids_flat, axis=0)  # (nq_flat, topk)
    
    if not rank_topk:  # return raw results
      # get text
      texts = self.texts[doc_ids_flat] if len(self.texts) else None
      if return_external_id:  # convert to external ids
        doc_ids_flat = self.ids[doc_ids_flat]
      return doc_ids_flat, scores_flat, texts
    
    # aggregate scores with max-sum on one query at a time (to save mem)
    assert return_external_id
    doc_ids_flat = self.ids_int[doc_ids_flat]

    assert query_ids is not None and query_splits is not None
    assert len(query_ids) == len(scores_flat)
    device = device or torch.device('cpu')
    scores_flat = torch.tensor(scores_flat).to(device)
    doc_ids_flat = torch.tensor(doc_ids_flat).to(device)

    qid2rank: Dict[int, List[str, float]] = {}
    for qi in range(len(query_splits)):
      qstart, qend = query_splits[qi - 1] if qi else 0, query_splits[qi]
      qid = query_ids[qstart]

      # split
      scores = scores_flat[qstart:qend]  # (nq, topk)
      doc_ids = doc_ids_flat[qstart:qend]  # (nq, topk)

      # max
      unique_doc_ids, doc_ids = torch.unique(doc_ids, return_inverse=True)  # (uqd,) (nq, topk) uqd is the number of unique docs
      nq, uqd = scores.size(0), unique_doc_ids.size(0)
      lowest = scores.min()
      agg_scores = torch.zeros(nq, uqd).to(device) + lowest  # (nq, uqd)
      agg_mask = torch.zeros(nq, uqd).to(device)  # (nq, uqd)
      agg_scores = torch_scatter.scatter_max(scores, doc_ids, out=agg_scores, dim=-1)[0]
      agg_mask = torch_scatter.scatter_max(torch.ones_like(scores), doc_ids, out=agg_mask, dim=-1)[0]
      agg_scores = agg_scores * agg_mask  # assume zero score if a qry-doc pair is absent

      # sum
      agg_scores = agg_scores.sum(0)  # (uqd)

      # sort
      if not rerank_topk:  # return the first-stage ranking result
        sort_scores, sort_i = torch.topk(agg_scores, min(rank_topk, uqd))  # (rank_topk)
        sort_dids = unique_doc_ids[sort_i]
        qid2rank[qid] = self.compose_rank_list(sort_dids, sort_scores)
        continue
        
      # rerank
      sort_scores, sort_i = torch.topk(agg_scores, min(rerank_topk, uqd))
      cand_dids = unique_doc_ids[sort_i]  # (cd)
      cd = cand_dids.size(0)

      # collect doc
      cand_emb_flat, cand_dids_flat = self.get_by_ids(cand_dids)  # (cd_flat, emb_size) (cd_flat)
      cand_emb_flat = cand_emb_flat.to(device)
      cand_dids_flat = cand_dids_flat.to(device)

      # collect qry
      qv = query_vectors[qstart:qend]  # (nq, emb_size)
      qv = torch.tensor(qv).to(device)
    
      # dot product    
      full_scores = (qv @ cand_emb_flat.T)  # (nq, cd_flat)
        
      # term weights
      if term_weights is not None:
        tw = term_weights[qstart:qend]  # (nq,)
        tw = torch.tensor(tw).to(device)
        full_scores = full_scores * term_weights.unsqueeze(-1)
      
      # max-sum
      unique_cand_dids_flat, cand_dids_flat = torch.unique(cand_dids_flat, return_inverse=True)  # (cd,) (cd_flat)
      assert len(cand_dids) == len(unique_cand_dids_flat)
      lowest = full_scores.min()
      agg_full_scores = torch.zeros(nq, cd).to(full_scores) + lowest
      agg_full_scores = torch_scatter.scatter_max(full_scores, cand_dids_flat, out=agg_full_scores, dim=-1)[0]  # (nq, cd)
      agg_full_scores = agg_full_scores.sum(0)  # (cd)

      # sort
      sort_scores, sort_i = torch.topk(agg_full_scores, min(rank_topk, cd))  # (rank_topk)
      sort_dids = unique_cand_dids_flat[sort_i]
      qid2rank[qid] = self.compose_rank_list(sort_dids, sort_scores)

    return qid2rank

  def get_by_ids(
    self, 
    ids_int: torch.LongTensor):  # (n)
    assert self.keep_raw_vector, 'require raw vectors'
    # (n, max_seq_len, emb_size), (n, max_seq_len)
    embs, mask = self.embs_strided.lookup(ids_int, output='padded')
    max_seq_len = embs.size(1)
    ids_packed = ids_int.unsqueeze(-1).repeat(1, max_seq_len)[mask]
    embs_packed = embs[mask]
    return embs_packed, ids_packed

  def _update_id_mapping(self, ids):
    ids = np.array(ids, dtype=str)
    self.ids = np.concatenate((self.ids, ids), axis=0)

  def _update_texts(self, texts: np.ndarray = None):
    if texts is None:
      return
    self.texts = np.concatenate((self.texts, texts), axis=0)

  def _update_emb(self, emb):
    if not self.keep_raw_vector:
      return
    if type(emb) is not torch.tensor:
      emb = torch.tensor(emb)
    self.embs = torch.cat([self.embs, emb], axis=0)


class MultiIndexer(object):
  def __init__(self, num_indices: int, *args, **kwargs):
    self.indices = [Indexer(*args, **kwargs) for _ in range(num_indices)]
    self.is_corresponding = True  # vectors in these indices correspond to each other 
  
  def load_from_npz(
      self, 
      emb_files: List[str], 
      index_name: str = None):
    # group files by idx
    ii2files: Dict[int, List[str]] = defaultdict(list)
    for f in sorted(emb_files):
      ii2files[int(f.rsplit('.', 2)[-2])].append(f)
    assert len(ii2files) >= len(self.indices), 'embedding files are not enought'
    # load files for each idx
    for ii, index in enumerate(self.indices):
      _index_name = index_name + f'.{ii:02d}' if index_name is not None else None
      index.load_from_npz(ii2files[ii], index_name=_index_name)
  
  def search_knn(self, query_vectors, *args, term_weights=None, **kwargs) -> List[Tuple[List[Union[str, int]], List[float], List[str]]]:
    result_li = []
    return_external_id = True
    if 'return_external_id' in kwargs:
      return_external_id = kwargs['return_external_id']
      del kwargs['return_external_id']
    assert query_vectors.shape[0] == len(self.indices)
    for ii, index in enumerate(self.indices):
      tw = term_weights[ii] if term_weights is not None else None
      result_li.append(index.search_knn(query_vectors[ii], *args, return_external_id=False, term_weights=tw, **kwargs))  # return raw id
    # perform max aggregation
    merged_result = []
    for i in range(len(result_li[0])):
      id2scoretext: Dict[int, Tuple[float, str]] = {}
      for result in result_li:
        for id, score, text in zip(*result[i]):
          if id not in id2scoretext:
            id2scoretext[id] = (score, text)
          else:
            assert text == id2scoretext[id][1]
            id2scoretext[id] = (max(score, id2scoretext[id][0]), text)
      ids, scores, texts = [], [], []
      for id, (score, text) in id2scoretext.items():
        ids.append(id)
        scores.append(score)
        texts.append(text)
      if return_external_id:
        ids = self.indices[0].ids[ids]
      merged_result.append((ids, scores, texts))
    return merged_result
  
  def get_by_ids(self, ids: List[str]):
    flat_embs = []
    flat_dids = None
    for index in self.indices:
      _flat_embs, _flat_dids = index.get_by_ids(ids)
      flat_embs.append(_flat_embs)
      if flat_dids is not None:
        assert len(flat_dids) == len(_flat_dids)
      flat_dids = _flat_dids
    flat_embs = np.stack(flat_embs, axis=0)  # (num_indices, num_embs, emb_size)
    return flat_embs, flat_dids
