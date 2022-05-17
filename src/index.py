# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from re import I
from typing import List, Tuple, Set, Union, Dict
import time
import glob
from pathlib import Path
from tqdm import tqdm
import faiss
import numpy as np

from src.util import logger

class Indexer(object):
  def __init__(self,
               vector_sz: int,
               n_subquantizers: int = 0,
               n_bits: int = 8,
               hnsw_m: int = 0,
               cuda_device: Union[int, List[int]] = -1,
               keep_raw_vector: bool = False):
    self.cuda_device = cuda_device
    self.vector_sz = vector_sz
    self.n_subquantizers = n_subquantizers
    self.n_bits = n_bits
    self.hnsw_m = hnsw_m
    self.keep_raw_vector = keep_raw_vector
    self.shard = False  # shard index across multiple gpus
    self.create_index()
    self.ids = np.empty((0), dtype=str)
    self.texts = np.empty((0), dtype=str)
    self.embs = np.empty((0, vector_sz), dtype=np.float32)

  def load_from_npz(self, emb_files: List[str], save_or_load_index: bool = True):
    embeddings_dir = Path(emb_files[0]).parent
    index_path = embeddings_dir / 'index.faiss'
    if save_or_load_index and index_path.exists():
      self.deserialize_from(embeddings_dir)
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
      if save_or_load_index:
        self.serialize(embeddings_dir)
    self.build_length_offset()
  
  def __len__(self):
    return len(self.ids)
  
  @property
  def use_gpu(self):
    if type(self.cuda_device) is int and self.cuda_device == -1:
      return False
    return True

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
      print(f'Move FAISS index to gpu {self.cuda_device}', flush=True)
      if self.shard:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.shard = True
      else:
        co = None
      self.index = faiss.index_cpu_to_gpus_list(self.index, co=co, gpus=self.cuda_device)
    else:  # single gpu
      logger.info(f'Move FAISS index to gpu {self.cuda_device}')
      print(f'Move FAISS index to gpu {self.cuda_device}', flush=True)
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
  
  def build_length_offset(self):
    self.id2offset: Dict[str, int] = {}
    self.id2len: Dict[str, int] = {}
    self.id2indices: Dict[str, List[int]] = {}
    for i, _id in enumerate(self.ids):
      if _id not in self.id2offset:
        self.id2offset[_id] = i
        self.id2len[_id] = 1
      else:
        assert i == self.id2offset[_id] + self.id2len[_id]  # make sure ids are consecutive
        self.id2len[_id] += 1
    for _id, offset in self.id2offset.items():
      length = self.id2len[_id]
      self.id2indices[_id] = [offset + i for i in range(length)]
  
  def ids2indices(self, ids: List[str]):
    indices: List[int] = []
    for _id in ids:
      indices.extend(self.id2indices[_id])
    return indices

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

  def search_knn(self,
                 query_vectors: np.array,
                 top_docs: int,
                 index_batch_size: int = 1024) -> List[Tuple[List[object], List[float], List[str]]]:
    query_vectors = query_vectors.astype('float32')
    result = []
    nbatch = (len(query_vectors) - 1) // index_batch_size + 1
    for k in range(nbatch):
      start_idx = k * index_batch_size
      end_idx = min((k + 1) * index_batch_size, len(query_vectors))
      q = query_vectors[start_idx:end_idx]
      scores, indexes = self.index.search(q, top_docs)
      # convert to external ids
      ids = [self.ids[query_top_idxs] for query_top_idxs in indexes]
      # get text
      texts = [(self.texts[query_top_idxs] if len(self.texts) else None) for query_top_idxs in indexes]
      result.extend([(ids[i], scores[i], texts[i]) for i in range(len(ids))])
    return result
  
  def get_by_ids_by_mask(self, ids: List[str]):
    mask = np.isin(self.ids, ids)
    _ids = self.ids[mask]
    assert self.keep_raw_vector, 'require raw vectors'
    _embs = self.embs[mask]
    return _embs, _ids

  def get_by_ids(self, ids: List[str]):
    indices = self.ids2indices(ids)
    _ids = self.ids[indices]
    assert self.keep_raw_vector, 'require raw vectors'
    _embs = self.embs[indices]
    return _embs, _ids

  def serialize(self, dir_path):
    index_file = dir_path / 'index.faiss'
    meta_file = dir_path / 'index.meta'
    logger.info(f'Serializing index to {index_file}, meta data to {meta_file}')
    faiss.write_index(faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index, str(index_file))
    with open(meta_file, mode='wb') as f:
      np.savez(f, ids=self.ids, texts=self.texts, embs=self.embs)

  def deserialize_from(self, dir_path):
    index_file = dir_path / 'index.faiss'
    meta_file = dir_path / 'index.meta'
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
    self.embs = np.concatenate((self.embs, emb), axis=0)
