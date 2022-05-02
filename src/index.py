# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Tuple, Set
import time
import glob
from pathlib import Path
from tqdm import tqdm
import torch
import faiss
import numpy as np

from src.util import logger

class Indexer(object):
  def __init__(self,
               vector_sz: int,
               n_subquantizers: int = 0,
               n_bits: int = 8,
               hnsw_m: int = 0,
               cuda_device: int = -1):
    self.cuda_device = cuda_device
    self.use_gpu = cuda_device >= 0
    if self.use_gpu:
      self.res = faiss.StandardGpuResources()
    if n_subquantizers > 0:
      self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
    elif hnsw_m > 0:
      self.index = faiss.IndexHNSWFlat(vector_sz, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    else:
      self.index = faiss.IndexFlatIP(vector_sz)
    if self.use_gpu:
      logger.info(f'Move FAISS index to gpu {self.cuda_device}')
      self.index = faiss.index_cpu_to_gpu(self.res, self.cuda_device, self.index)
    self.ids = np.empty((0), dtype=str)
    self.texts = np.empty((0), dtype=str)

  def load_from_npz(self, filepath_pattern: str, save_or_load_index: bool = True):
    input_paths = glob.glob(filepath_pattern)
    input_paths = sorted(input_paths)
    embeddings_dir = Path(input_paths[0]).parent
    index_path = embeddings_dir / 'index.faiss'
    if save_or_load_index and index_path.exists():
      self.deserialize_from(embeddings_dir)
    else:
      logger.info(f'indexing passages from files {input_paths}')
      start_time_indexing = time.time()
      for i, input_path in enumerate(input_paths):
        logger.info(f'loading file {input_path}')
        with open(input_path, 'rb') as fin:
          npzfile = np.load(fin)
          ids, embeddings, texts = npzfile['ids'], npzfile['embeddings'], npzfile['words']
          self.index_data(ids, embeddings, texts)
      logger.info(f'data indexing completed with time {time.time() - start_time_indexing:.1f} s.')
      if save_or_load_index:
        self.serialize(embeddings_dir)

  def index_data(self,
                 ids: np.ndarray,
                 embeddings: np.ndarray,
                 texts: np.ndarray = None,
                 indexing_batch_size: int = 50000,
                 disable_log: bool = False):
      self._update_id_mapping(ids)
      self._update_texts(texts)
      embeddings = embeddings.astype('float32')
      if not self.index.is_trained:
          self.index.train(embeddings)
      for b in tqdm(range(0, len(embeddings), indexing_batch_size), desc='indexing', disable=disable_log):
        self.index.add(embeddings[b:b + indexing_batch_size])
      if not disable_log:
        logger.info(f'total data indexed {len(self.ids)}')

  def remove_data(self, num_unique_id: int):
    unique_ids: Set[str] = set()
    stop = False
    for i, id in enumerate(self.ids):
      if len(unique_ids) == num_unique_id and id not in unique_ids:
        stop = True
        break
      unique_ids.add(id)
    if not stop:  # remove all
      i = len(self.ids)
    self.index.remove_ids(np.arange(i))
    self.ids = self.ids[i:]
    self.texts = self.texts[i:]

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

  def serialize(self, dir_path):
    index_file = dir_path / 'index.faiss'
    meta_file = dir_path / 'index.meta'
    logger.info(f'Serializing index to {index_file}, meta data to {meta_file}')
    if self.use_gpu:
      faiss.write_index(faiss.index_gpu_to_cpu(self.index), str(index_file))
    else:
      faiss.write_index(self.index, str(index_file))
    with open(meta_file, mode='wb') as f:
      np.savez(f, ids=self.ids, texts=self.texts)

  def deserialize_from(self, dir_path):
    index_file = dir_path / 'index.faiss'
    meta_file = dir_path / 'index.meta'
    logger.info(f'Loading index from {index_file}, meta data from {meta_file}')

    self.index = faiss.read_index(str(index_file))
    if self.use_gpu:
      logger.info('Move FAISS index to gpu')
      self.index = faiss.index_cpu_to_gpu(self.res, self.cuda_device, self.index)
    logger.info('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)

    with open(meta_file, 'rb') as reader:
      npzfile = np.load(reader)
      self.ids, self.texts = npzfile['ids'], npzfile['texts']
    assert len(self.ids) == self.index.ntotal, 'Deserialized ids should match faiss index size'

  def _update_id_mapping(self, ids):
    ids = np.array(ids, dtype=str)
    self.ids = np.concatenate((self.ids, ids), axis=0)

  def _update_texts(self, texts: np.ndarray = None):
    if texts is None:
      return
    self.texts = np.concatenate((self.texts, texts), axis=0)
