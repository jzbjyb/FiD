from typing import List, Dict, Union
from collections import defaultdict
from indexed import IndexedOrderedDict
import random
import functools
import torch
import torch.distributed as dist
from torch.multiprocessing import Queue, Process
import numpy as np
from src.util import WandbLogger, global_context
from src.dist_utils import get_rank, all_gather_tensors, scatter_tensors, all_gather_objects
from src.index import Indexer

class MemoryBank:
  def __init__(
      self, 
      max_size: int, 
      indexing_dimension: int, 
      use_gpu: Union[bool, int, List[int]] = True, 
      use_head_idx: int = None,
      pad_token_id: int = None,
      bank_topk: int = None,
      use_random: bool = False):
    self.max_size = max_size
    self.indexing_dimension = indexing_dimension
    self.use_head_idx = use_head_idx
    self.pad_token_id = pad_token_id
    self.bank_topk = bank_topk
    self.use_random = use_random
    if type(use_gpu) is bool and use_gpu == False:
      cuda_device = -1
    elif type(use_gpu) is bool and use_gpu == True:  # use current device
      cuda_device = get_rank() if global_context['opt'].is_distributed else 0
    else:
      cuda_device = use_gpu  # use specified device
    self.get_index = lambda: Indexer(indexing_dimension, n_subquantizers=0, n_bits=0, hnsw_m=0, cuda_device=cuda_device)
    self.main_device = cuda_device[0] if type(cuda_device) is list else cuda_device
    if self.main_device >= 0:
      self.main_device = torch.device(self.main_device)
    else:
      self.main_device = torch.device('cpu')
    self.index = self.get_index()
    self.remove_count_when_full: Union[int, float] = 0.5  # absolute count of ratio
    self.dedup = True
    self.raw_ids: Dict[str, None] = IndexedOrderedDict()  # real doc ids for dedup
    self.ids: List[int] = []  # fake doc ids for indexing
    self.keys: List[torch.FloatTensor] = []
    self.queries: List[torch.FloatTensor] = []
    self.masks: List[torch.BoolTensor] = []
    self.token_ids: List[torch.LongTensor] = []

  def __len__(self):
    return len(self.ids)

  @property
  def doc_seq_len(self):
    if len(self.keys) <= 0:
      return None
    return self.keys[0].size(0)

  def add(self,
          key: torch.FloatTensor,  # (bs, seq_len, emb_size)
          query: torch.FloatTensor,  # (bs, seq_len, emb_size)
          mask: torch.BoolTensor,  # (bs, seq_len)
          token_id: torch.LongTensor,  # (bs, seq_len)
          doc_id: np.ndarray = None):  # (bs)
    bs, emb_size = key.size(0), key.size(-1)
    assert key.size(-1) == query.size(-1) == self.indexing_dimension
    assert key.size(0) == query.size(0) == mask.size(0)
    assert key.size(1) == query.size(1) == mask.size(1)

    dup_count = 0
    # add torch data and index data
    key = key.detach().cpu()
    query = query.detach().cpu()
    mask = mask.detach().cpu()
    token_id = token_id.detach().cpu()
    embs_for_index: List[np.ndarray] = []
    ids_for_index: List[int] = []
    for i in range(bs):
      num_token_in_doc = mask[i].sum().item()
      if num_token_in_doc <= 0:  # skip empty doc
        continue
      if self.dedup:  # skip dup doc
        if doc_id[i] in self.raw_ids:
          dup_count += 1
          continue
        else:
          self.raw_ids[doc_id[i]] = None
      _id = self.ids[-1] + 1 if len(self.ids) else 0
      self.ids.append(_id)  # increase by 1
      self.keys.append(key[i])
      self.queries.append(query[i])
      self.masks.append(mask[i])
      self.token_ids.append(token_id[i])
      embs_for_index.append(torch.masked_select(key[i], mask[i].unsqueeze(-1)).view(-1, emb_size).numpy())
      for _ in range(num_token_in_doc):
        ids_for_index.append(str(_id))
    if len(embs_for_index) <= 0:
      return dup_count
    embs_for_index = np.concatenate(embs_for_index, axis=0)
    ids_for_index = np.array(ids_for_index, dtype=str)
    assert len(ids_for_index) == len(embs_for_index)
    self.index.index_data(ids_for_index, embs_for_index, disable_log=True)
    if len(self) > 0:
      assert str(self.index.ids[0]) == str(self.ids[0]) and str(self.index.ids[-1]) == str(self.ids[-1])
      assert len(self.ids) == len(self.keys) == len(self.queries) == len(self.masks) == len(self.token_ids)
      if self.dedup:
        assert len(self.ids) == len(self.raw_ids)
    # remove if needed
    if len(self) > self.max_size:
      rm_count = min(self.remove_count_when_full, len(self)) if type(self.remove_count_when_full) is int else int(len(self) * self.remove_count_when_full)
      self.remove(rm_count)
    return dup_count

  def remove(self, num_remove: int):
    assert num_remove > 0
    to_empty = num_remove == len(self)
    num_remove = min(num_remove, len(self))    
    if to_empty:
      del self.ids, self.keys, self.queries, self.masks, self.token_ids, self.raw_ids
      self.ids, self.keys, self.queries, self.masks, self.token_ids, self.raw_ids = [], [], [], [], [], IndexedOrderedDict()
      del self.index
      self.index = self.get_index()
    else:
      self.ids = self.ids[num_remove:]
      self.keys = self.keys[num_remove:]
      self.queries = self.queries[num_remove:]
      self.masks = self.masks[num_remove:]
      self.token_ids = self.token_ids[num_remove:]
      self.raw_ids = IndexedOrderedDict.fromkeys(list(self.raw_ids)[num_remove:])
      self.index.remove_data(num_remove)
    if len(self) > 0:
      assert str(self.index.ids[0]) == str(self.ids[0]) and str(self.index.ids[-1]) == str(self.ids[-1])
      assert len(self.ids) == len(self.keys) == len(self.queries) == len(self.masks) == len(self.token_ids)
      if self.dedup:
        assert len(self.ids) == len(self.raw_ids)

  def query(self,
            key: torch.FloatTensor,  # (bs, max_qry_len, emb_size)
            query: torch.FloatTensor,  # (bs, max_qry_len, emb_size)
            mask: torch.BoolTensor,  # (bs, max_qry_len)
            token_id: torch.LongTensor,  # (bs, max_qry_len)
            filter_doc_id: np.ndarray = None,  # (bs, num_filter_doc)
            token_topk: int = 1,
            rerank_topk: int = 0,
            doc_topk: int = 1,
            use_random: bool = False,
            debug: bool = False):
    assert doc_topk <= len(self)
    bs, emb_size = key.size(0), key.size(-1)
    assert key.size(-1) == query.size(-1) == self.indexing_dimension
    assert key.size(0) == query.size(0) == mask.size(0)
    assert key.size(1) == query.size(1) == mask.size(1)
    doc_to_keep = doc_topk + (filter_doc_id.shape[1] if self.dedup else 0)

    # token-level retrieval
    embs_for_query = torch.masked_select(query, mask.unsqueeze(-1)).view(-1, emb_size).detach().cpu().numpy()
    qids = [i for i in range(bs) for j in range(mask[i].sum())]
    query_lens = mask.sum(-1)  # (bs)
    query_splits = torch.cumsum(query_lens, dim=0)  # (bs)
    #top_ids_and_scores = self.index.search_knn(embs_for_query, token_topk)
    qid2rank = self.index.search_knn(
      embs_for_query, 
      token_topk,  # 2048
      term_weights=None,
      query_ids=qids,
      query_splits=query_splits.tolist(),
      rank_topk=doc_to_keep,
      rerank_topk=rerank_topk,
      device=self.main_device)
    assert bs == len(qid2rank)

    # concatenate
    concat_key, concat_qry, concat_mask, concat_tokid = [], [], [], []
    retrieved_count: List[int] = []
    filtered_count: List[int] = []
    for qid in range(bs):
      rank = qid2rank[qid]
      query_len = query_lens[qid]
      dids = [int(did) - self.ids[0] for did, _ in rank[:doc_to_keep]]  # use the first id as offset
      retrieved_count.append(len(dids))

      if debug:
        print(tokenizer.decode(tokid[qid][:query_len]))
        for did in dids[:3]:
          print(tokenizer.decode(self.tokids[did]))
        input()

      if use_random:
        dids = list(random.sample(range(len(self)), doc_to_keep))
      elif len(dids) < doc_to_keep:
        dids += list(random.sample(range(len(self)), doc_to_keep - len(dids)))
      assert len(dids) == doc_to_keep

      if self.dedup:
        raw_ids = self.raw_ids.keys()
        to_filter = set(filter_doc_id[qid].tolist())
        filtered_count.append(len([did for did in dids if raw_ids[did] in to_filter]))
        dids = [did for did in dids if raw_ids[did] not in to_filter][:doc_topk]
        assert len(dids) == doc_topk

      query_qry = query[qid][:query_len].unsqueeze(0).repeat(doc_topk, 1, 1)  # (doc_topk, qry_len, emb_size)
      doc_qry = torch.stack([self.queries[did] for did in dids], dim=0).to(query)  # (doc_topk, seq_len, emb_size)
      concat_qry.append(torch.cat([query_qry, doc_qry], dim=1)[:, :self.doc_seq_len])  # (doc_topk, seq_len, emb_size)

      query_key = key[qid][:query_len].unsqueeze(0).repeat(doc_topk, 1, 1)  # (doc_topk, qry_len, emb_size)
      doc_key = torch.stack([self.keys[did] for did in dids], dim=0).to(key)  # (doc_topk, seq_len, emb_size)
      concat_key.append(torch.cat([query_key, doc_key], dim=1)[:, :self.doc_seq_len]) # (doc_topk, seq_len, emb_size)

      query_mask = torch.cat([mask[qid][:query_len], torch.zeros(self.doc_seq_len - query_len).to(mask)])  # (seq_len)
      query_mask = query_mask.unsqueeze(0).unsqueeze(0).repeat(doc_topk, query_len, 1)  # (doc_topk, query_len, seq_len)
      doc_mask = torch.stack([self.masks[did] for did in dids], dim=0).to(mask)  # (doc_topk, seq_len)
      doc_mask = torch.cat([torch.zeros((doc_topk, query_len)).to(mask), doc_mask], dim=1)[:, :self.doc_seq_len]
      doc_mask = doc_mask.unsqueeze(1).repeat(1, self.doc_seq_len - query_len, 1)  # (doc_topk, seq_len - query_len, seq_len)
      concat_mask.append(torch.cat([query_mask, doc_mask], dim=1))  # (doc_topk, seq_len, seq_len)

      query_tokid = token_id[qid][:query_len].unsqueeze(0).repeat(doc_topk, 1)  # (doc_topk, qry_len)
      doc_tokid = torch.stack([self.token_ids[did] for did in dids], dim=0).to(token_id)  # (doc_topk, seq_len)
      concat_tokid.append(torch.cat([query_tokid, doc_tokid], dim=1)[:, :self.doc_seq_len])  # (doc_topk, seq_len, emb_size)

    concat_key = torch.stack(concat_key, dim=0)  # (bs, doc_topk, seq_len, emb_size)
    concat_qry = torch.stack(concat_qry, dim=0)  # (bs, doc_topk, seq_len, emb_size)
    concat_mask = torch.stack(concat_mask, dim=0)  # (bs, doc_topk, seq_len, seq_len)
    concat_tokid = torch.stack(concat_tokid, dim=0)  # (bs, doc_topk, seq_len)
    return concat_key, concat_qry, concat_mask, concat_tokid, np.sum(filtered_count), np.sum(retrieved_count)

  def add_from_t5(
      self,
      query_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
      key_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
      attention_mask: torch.BoolTensor,  # (bs * n_context, seq_len, seq_len)
      input_ids: torch.LongTensor,  # (bs * n_context, seq_len)
      input_doc_ids: np.ndarray,  # (bs * n_context)
      n_context: int):
    assert self.use_head_idx is not None  # TODO: add multi-head?
    # reshape
    n_heads, seq_len, emb_size = key_states.size()[1:]
    # (bs, n_context, seq_len, emb_size_per_head)
    key_states = key_states.view(-1, n_context, n_heads, seq_len, emb_size)[:, :, self.use_head_idx]
    query_states = query_states.view(-1, n_context, n_heads, seq_len, emb_size)[:, :, self.use_head_idx]
    # (bs, n_context, seq_len, seq_len)
    attention_mask = attention_mask.view(-1, n_context, seq_len, seq_len)
    # (bs, n_context, seq_len)
    input_ids = input_ids.view(-1, n_context, seq_len)
    query_len = attention_mask[:, 0, 0].sum(-1)  # (bs)
    input_doc_ids = input_doc_ids.reshape(-1, n_context)  # (bs, n_context)

    dup_count = 0
    for i in range(len(query_len)):
      pad = torch.zeros(n_context, query_len[i], emb_size).to(key_states)  # (n_context, qry_len, emb_size_per_head)
      pad_mask = torch.zeros(n_context, query_len[i]).to(attention_mask)  # (n_context, qry_len)
      pad_input_ids = torch.full((n_context, query_len[i]), self.pad_token_id).to(input_ids)  # (n_context, qry_len)
      # always use the same seq_len
      # (n_context, seq_len, emb_size_per_head)
      key_to_add = torch.cat([key_states[i, :, query_len[i]:], pad], dim=1)
      query_to_add = torch.cat([query_states[i, :, query_len[i]:], pad], dim=1)
      # (n_context, seq_len)
      mask_to_add = torch.cat([attention_mask[i, :, query_len[i], query_len[i]:], pad_mask], dim=1)
      # (n_context, seq_len)
      input_ids_to_add = torch.cat([input_ids[i, :, query_len[i]:], pad_input_ids], dim=1)
      dup_count += self.add(key_to_add, query_to_add, mask_to_add, input_ids_to_add, input_doc_ids[i])
    return dup_count

  def query_from_t5(
      self,
      query_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
      key_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
      attention_mask: torch.BoolTensor,  # (bs * n_context, seq_len, seq_len)
      input_ids: torch.LongTensor,  # (bs * n_context, seq_len)
      input_doc_ids: np.ndarray,  # (bs * n_context)
      n_context: int):
    assert self.use_head_idx is not None  # TODO: add multi-head?
    assert len(self) >= self.bank_topk  # query memory bank when there are enough
    assert self.bank_topk % n_context == 0
    assert self.bank_topk <= 2048, 'faiss does not support topk > 2048'

    # reshape
    n_heads, seq_len, emb_size = key_states.size()[1:]
    # (bs, n_context, seq_len, emb_size_per_head)
    key_states = key_states.view(-1, n_context, n_heads, seq_len, emb_size)[:, :, self.use_head_idx]
    query_states = query_states.view(-1, n_context, n_heads, seq_len, emb_size)[:, :, self.use_head_idx]
    # (bs, n_context, seq_len, seq_len)
    attention_mask = attention_mask.view(-1, n_context, seq_len, seq_len)
    # (bs, n_context, seq_len)
    input_ids = input_ids.view(-1, n_context, seq_len)
    query_len = attention_mask[:, 0, 0].sum(-1)  # (bs)
    input_doc_ids = input_doc_ids.reshape(-1, n_context)  # (bs, n_context)

    # extract query
    max_qry_len = query_len.max().item()
    key_to_query = key_states[:, 0, :max_qry_len]  # (bs, max_qry_len, emb_size_per_head)
    query_to_query = query_states[:, 0, :max_qry_len]  # (bs, max_qry_len, emb_size_per_head)
    mask_to_query = attention_mask[:, 0, 0, :max_qry_len]  # (bs, max_qry_len)
    input_ids_to_query = input_ids[:, 0, :max_qry_len]  # (bs, max_qry_len)

    # retrieve
    # (bs, memory_bank_topk, seq_len, emb_size_per_head) * 2
    # (bs, memory_bank_topk, seq_len, seq_len)
    # (bs, memory_bank_topk, seq_len)
    _key_states, _query_states, _attention_mask, _input_ids, filtered_count, retrieved_count = self.query(
      key_to_query, query_to_query, mask_to_query, input_ids_to_query, filter_doc_id=input_doc_ids,
      token_topk=2048, rerank_topk=0, doc_topk=self.bank_topk, use_random=self.use_random)
    # (bs * memory_bank_topk, seq_len)
    # (bs * memory_bank_topk, seq_len, seq_len)
    return _input_ids.view(-1, seq_len), _attention_mask.view(-1, seq_len, seq_len), filtered_count, retrieved_count
  
  def query_then_add_from_t5(self, *args, **kwargs):
    input_ids, attention_mask, filtered_count, retrieved_count = self.query_from_t5(*args, **kwargs)
    dup_count = self.add_from_t5(*args, **kwargs)
    return input_ids, attention_mask, filtered_count, retrieved_count, dup_count

def memory_bank_process(
    in_queue: Queue, 
    out_queue: Queue, 
    seed: int=2022,
    **init_kwargs):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  bank = MemoryBank(**init_kwargs)

  while True:  # monitor in_queue
    item = in_queue.get()
    if type(item) is str and item == 'done':
      break
    if item['action'] == 'add':
      del item['action']
      dup_count = bank.add_from_t5(**item)
      del item
      out_queue.put({'size': len(bank), 'dup_count': dup_count})
    elif item['action'] == 'query':
      del item['action']
      input_ids, attention_mask, filtered_count, retrieved_count = bank.query_from_t5(**item)
      del item
      out_queue.put({
        'input_ids': input_ids, 'attention_mask': attention_mask, 
        'filtered_count': filtered_count, 'retrieved_count': retrieved_count})
    elif item['action'] == 'query_then_add':
      del item['action']
      input_ids, attention_mask, filtered_count, retrieved_count, dup_count = bank.query_then_add_from_t5(**item)
      del item
      out_queue.put({
        'input_ids': input_ids, 'attention_mask': attention_mask, 
        'filtered_count': filtered_count, 'retrieved_count': retrieved_count, 
        'size': len(bank), 'dup_count': dup_count})
    else:
      raise NotImplementedError

class MemoryBankProcessHelper:
  def __init__(self, **init_kwargs):
    self.is_distributed = global_context['opt'].is_distributed
    self.rank = get_rank() if self.is_distributed else 0
    self.master_rank = 0
    self.bank_topk = init_kwargs['bank_topk']
    self.init_kwargs = init_kwargs
    self._started = False
    self.bank_size = 0
  
  def __len__(self):
    return self.bank_size
  
  def start(self):
    if self.rank != self.master_rank or self._started:  # run the process on the master rank
      return
    if not hasattr(self, 'process'):  # create process
      self.in_queue = Queue()
      self.in_queue_cache = []
      self.out_queue = Queue()
      self.process = Process(target=functools.partial(memory_bank_process, **self.init_kwargs), args=(self.in_queue, self.out_queue))
      self.process.daemon = True
    # start process
    self.process.start()
    self._started = True

  def finish(self):
    if self.rank != self.master_rank:
      return
    assert self._started, 'process not started yet'
    self.out_ueue.put('done')
    self.process.join()
    self._started = False
  
  def _gather(
    self,
    query_states: torch.FloatTensor, 
    key_states: torch.FloatTensor, 
    attention_mask: torch.BoolTensor, 
    input_ids: torch.LongTensor,
    input_doc_ids: np.ndarray):
    if self.is_distributed:
      if self.rank == self.master_rank:
        query_states, key_states, attention_mask, input_ids = all_gather_tensors(query_states, key_states, attention_mask, input_ids)
        query_states = torch.cat(query_states, dim=0)
        key_states = torch.cat(key_states, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        input_ids = torch.cat(input_ids, dim=0)
      else:
        all_gather_tensors(query_states, key_states, attention_mask, input_ids)
      if self.rank == self.master_rank:
        input_doc_ids = all_gather_objects(input_doc_ids)
        input_doc_ids = np.concatenate(input_doc_ids, axis=0)
      else:
        all_gather_objects(input_doc_ids)
    return query_states, key_states, attention_mask, input_ids, input_doc_ids
  
  def _scatter(self, input_ids, attention_mask, bs, seq_len):
    if self.is_distributed:
      if self.rank == self.master_rank:
        input_ids, attention_mask = scatter_tensors(input_ids, attention_mask, src=self.master_rank, by_broadcast=True)
      else:
        input_ids = torch.zeros(bs * self.bank_topk, seq_len).to(input_ids)
        attention_mask = torch.zeros(bs * self.bank_topk, seq_len, seq_len).to(attention_mask)
        input_ids, attention_mask = scatter_tensors(input_ids, attention_mask, src=self.master_rank, by_broadcast=True)
    return input_ids, attention_mask
  
  def _broadcast_bank_size(self):
    if self.is_distributed:  # broadcast the size
      if self.rank == self.master_rank:
        dist.broadcast_object_list([self.bank_size], src=self.master_rank)
      else:
        objs = [None]
        dist.broadcast_object_list(objs, src=self.master_rank)
        self.bank_size = objs[0]

  def add_from_t5(
    self,
    query_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
    key_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
    attention_mask: torch.BoolTensor,  # (bs * n_context, seq_len, seq_len)
    input_ids: torch.LongTensor,  # (bs * n_context, seq_len)
    input_doc_ids: np.ndarray,  # (bs * n_context)
    n_context: int,
    flush: bool = False):
    query_states, key_states, attention_mask, input_ids, input_doc_ids = self._gather(
      query_states, key_states, attention_mask, input_ids, input_doc_ids)
    if not self.is_distributed or self.rank == self.master_rank:
      total_n_doc = query_states.size(0)
      req = {
        'query_states': query_states.cpu(),
        'key_states': key_states.cpu(),
        'attention_mask': attention_mask.cpu(),
        'input_ids': input_ids.cpu(),
        'input_doc_ids': input_doc_ids,
        'n_context': n_context,
        'action': 'add'
      }
      self.in_queue_cache.append(req)
      if flush:
        for req in self.in_queue_cache:
          self.in_queue.put(req)
          rec = self.out_queue.get()  # get the current size
          self.bank_size = rec['size']
          WandbLogger.log_w_step({'memory-band-size': self.bank_size})
          WandbLogger.log_w_step({'memory-band-dup-count': rec['dup_count'] / total_n_doc})
        self.in_queue_cache = []
    if flush:
      self._broadcast_bank_size()
    if self.is_distributed:
      torch.distributed.barrier()
  
  def query_from_t5(
    self,
    query_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
    key_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
    attention_mask: torch.BoolTensor,  # (bs * n_context, seq_len, seq_len)
    input_ids: torch.LongTensor,  # (bs * n_context, seq_len)
    input_doc_ids: np.ndarray,  # (bs * n_context)
    n_context: int):
    bs, seq_len = input_ids.size(0) // n_context, input_ids.size(1)
    query_states, key_states, attention_mask, input_ids, input_doc_ids = self._gather(
      query_states, key_states, attention_mask, input_ids, input_doc_ids)
    if not self.is_distributed or self.rank == self.master_rank:
      total_n_doc = query_states.size(0)
      total_n_query = total_n_doc // n_context
      self.in_queue.put({
        'query_states': query_states.cpu(),
        'key_states': key_states.cpu(),
        'attention_mask': attention_mask.cpu(),
        'input_ids': input_ids.cpu(),
        'input_doc_ids': input_doc_ids,
        'n_context': n_context,
        'action': 'query'
      })
      rec = self.out_queue.get()
      input_ids, attention_mask = rec['input_ids'].to(input_ids), rec['attention_mask'].to(attention_mask)
      WandbLogger.log_w_step({'memory-band-filter-count': rec['filtered_count']  / total_n_doc})
      WandbLogger.log_w_step({'memory-band-retrieve-count': rec['retrieved_count']  / total_n_query})
    input_ids, attention_mask = self._scatter(input_ids, attention_mask, bs, seq_len)
    return input_ids, attention_mask
  
  def query_then_add_from_t5(
    self,
    query_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
    key_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
    attention_mask: torch.BoolTensor,  # (bs * n_context, seq_len, seq_len)
    input_ids: torch.LongTensor,  # (bs * n_context, seq_len)
    input_doc_ids: np.ndarray,  # (bs * n_context)
    n_context: int):
    bs, seq_len = input_ids.size(0) // n_context, input_ids.size(1)
    query_states, key_states, attention_mask, input_ids, input_doc_ids = self._gather(
      query_states, key_states, attention_mask, input_ids, input_doc_ids)
    if not self.is_distributed or self.rank == self.master_rank:
      total_n_doc = query_states.size(0)
      total_n_query = total_n_doc // n_context
      self.in_queue.put({
        'query_states': query_states.cpu(),
        'key_states': key_states.cpu(),
        'attention_mask': attention_mask.cpu(),
        'input_ids': input_ids.cpu(),
        'input_doc_ids': input_doc_ids,
        'n_context': n_context,
        'action': 'query_then_add'
      })
      rec = self.out_queue.get()
      # (bs * num_gpu * bank_topk, seq_len) (bs * num_gpu * bank_topk, seq_len, seq_len)
      input_ids, attention_mask = rec['input_ids'].to(input_ids), rec['attention_mask'].to(attention_mask)
      self.bank_size = rec['size']
      WandbLogger.log_w_step({'memory-band-size': self.bank_size})
      WandbLogger.log_w_step({'memory-band-dup-count': rec['dup_count'] / total_n_doc})
      WandbLogger.log_w_step({'memory-band-filter-count': rec['filtered_count'] / total_n_doc})
      WandbLogger.log_w_step({'memory-band-retrieve-count': rec['retrieved_count'] / total_n_query})
    self._broadcast_bank_size()
    input_ids, attention_mask = self._scatter(input_ids, attention_mask, bs, seq_len)
    return input_ids, attention_mask
