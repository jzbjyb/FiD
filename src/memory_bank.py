from typing import List, Dict, Union
from collections import defaultdict
import random
import functools
import torch
import torch.distributed as dist
from torch.multiprocessing import Queue, Process
import numpy as np
from src.util import WandbLogger, global_context
from src.dist_utils import get_rank, all_gather_tensors, scatter_tensors
from src.index import Indexer

class MemoryBank:
  def __init__(
      self, 
      max_size: int, 
      indexing_dimension: int, 
      use_gpu: Union[bool, int] = True, 
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
    self.index = self.get_index()
    self.ids: List[int] = []
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
          token_id: torch.LongTensor = None,  # (bs, seq_len)
          debug: bool = False):
    bs, emb_size = key.size(0), key.size(-1)
    assert key.size(-1) == query.size(-1) == self.indexing_dimension
    assert key.size(0) == query.size(0) == mask.size(0)
    assert key.size(1) == query.size(1) == mask.size(1)
    prev_id = (self.ids[-1] + 1) if len(self.ids) else 0

    # add torch data
    key = key.detach().cpu()
    query = query.detach().cpu()
    mask = mask.detach().cpu()
    for i in range(bs):
      self.ids.append(prev_id + i)
      self.keys.append(key[i])
      self.queries.append(query[i])
      self.masks.append(mask[i])
      if token_id is not None:
        self.token_ids.append(token_id[i])
    # add index data
    embs_for_index = torch.masked_select(key, mask.unsqueeze(-1)).view(-1, emb_size).numpy()
    ids_for_index = []
    for i in range(bs):
      for j in range(mask[i].sum()):
        ids_for_index.append(str(prev_id + i))
    ids_for_index = np.array(ids_for_index, dtype=str)
    assert len(ids_for_index) == len(embs_for_index)
    self.index.index_data(ids_for_index, embs_for_index, disable_log=True)
    if debug:
      print('add', len(self.ids), len(self.index.ids), self.ids[-3:], self.index.ids[-3:])
    # remove if needed
    if len(self) > self.max_size:  # TODO: add remove or shrink
      self.remove(len(self))
      if debug:
        print('remove', len(self.ids), len(self.index.ids), self.ids[-3:], self.index.ids[-3:])

  def remove(self, num_remove: int):
    assert num_remove > 0
    to_empty = num_remove == len(self)
    num_remove = min(num_remove, len(self))
    # remove torch data
    self.ids = self.ids[num_remove:]
    self.keys = self.keys[num_remove:]
    self.queries = self.queries[num_remove:]
    self.masks = self.masks[num_remove:]
    self.token_ids = self.token_ids[num_remove:]
    # remove index data
    if to_empty:
      self.index = self.get_index()
    else:
      self.index.remove_data(num_remove)
    if len(self) > 0:
      assert str(self.index.ids[0]) == str(self.ids[0])

  def query(self,
            key: torch.FloatTensor,  # (bs, max_qry_len, emb_size)
            query: torch.FloatTensor,  # (bs, max_qry_len, emb_size)
            mask: torch.BoolTensor,  # (bs, max_qry_len)
            token_id: torch.LongTensor = None,  # (bs, max_qry_len)
            token_topk: int = 1,
            doc_topk: int = 1,
            use_random: bool = False,
            debug: bool = False):
    assert doc_topk <= len(self)
    bs, emb_size = key.size(0), key.size(-1)
    assert key.size(-1) == query.size(-1) == self.indexing_dimension
    assert key.size(0) == query.size(0) == mask.size(0)
    assert key.size(1) == query.size(1) == mask.size(1)

    # token-level retrieval
    embs_for_query = torch.masked_select(query, mask.unsqueeze(-1)).view(-1, emb_size).detach().cpu().numpy()
    qids = [i for i in range(bs) for j in range(mask[i].sum())]
    top_ids_and_scores = self.index.search_knn(embs_for_query, token_topk)
    assert len(embs_for_query) == len(qids) == len(top_ids_and_scores)

    # aggregate
    qid2tokens2did2score: Dict[int, List[Dict[int, float]]] = defaultdict(list)
    for i, (dids, scores, texts) in enumerate(top_ids_and_scores):
      qid = qids[i]
      qid2tokens2did2score[qid].append(defaultdict(lambda: -1e10))
      for did, score in zip(dids, scores):
        did = int(did)
        qid2tokens2did2score[qid][-1][did] = max(score, qid2tokens2did2score[qid][-1][did])
    qid2did2score: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(lambda: 0))
    for qid, tokens in qid2tokens2did2score.items():
      for token in tokens:
        for did, score in token.items():
          qid2did2score[qid][did] += score

    # concatenate
    concat_key, concat_qry, concat_mask, concat_tokid = [], [], [], []
    retrieved_count: List[int] = []
    for qid in range(bs):
      did2score = qid2did2score[qid]
      query_len = mask[qid].sum().item()
      dids = list(sorted(did2score.items(), key=lambda x: -x[1]))[:doc_topk]
      retrieved_count.append(len(dids))
      dids = [did - self.ids[0] for did, _ in dids]  # use the first id as offset

      if debug:
        print(tokenizer.decode(tokid[qid][:query_len]))
        for did in dids[:3]:
          print(tokenizer.decode(self.tokids[did]))
        input()

      if use_random:
        dids = list(random.sample(range(len(self)), doc_topk))
      elif len(dids) < doc_topk:
        dids += list(random.sample(range(len(self)), doc_topk - len(dids)))
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

    WandbLogger.log_w_step({'memory-band-retrieved-count': np.mean(retrieved_count)})
    concat_key = torch.stack(concat_key, dim=0)  # (bs, doc_topk, seq_len, emb_size)
    concat_qry = torch.stack(concat_qry, dim=0)  # (bs, doc_topk, seq_len, emb_size)
    concat_mask = torch.stack(concat_mask, dim=0)  # (bs, doc_topk, seq_len, seq_len)
    concat_tokid = torch.stack(concat_tokid, dim=0)  # (bs, doc_topk, seq_len)
    return concat_key, concat_qry, concat_mask, concat_tokid

  def add_from_t5(
      self,
      query_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
      key_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
      attention_mask: torch.BoolTensor,  # (bs * n_context, seq_len, seq_len)
      input_ids: torch.LongTensor,  # (bs * n_context, seq_len)
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
      self.add(key_to_add, query_to_add, mask_to_add, input_ids_to_add)
    
    return len(self)

  def query_from_t5(
      self,
      query_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
      key_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
      attention_mask: torch.BoolTensor,  # (bs * n_context, seq_len, seq_len)
      input_ids: torch.LongTensor,  # (bs * n_context, seq_len)
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
    _key_states, _query_states, _attention_mask, _input_ids = self.query(
      key_to_query, query_to_query, mask_to_query, input_ids_to_query,
      token_topk=2048, doc_topk=self.bank_topk, use_random=self.use_random)
    # (bs * memory_bank_topk, seq_len)
    # (bs * memory_bank_topk, seq_len, seq_len)
    return _input_ids.view(-1, seq_len), _attention_mask.view(-1, seq_len, seq_len)
  
  def query_then_add_from_t5(self, *args, **kwargs):
    input_ids, attention_mask = self.query_from_t5(*args, **kwargs)
    self.add_from_t5(*args, **kwargs)
    return input_ids, attention_mask

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
      bank.add_from_t5(**item)
      del item
      out_queue.put({'size': len(bank)})
    elif item['action'] == 'query':
      del item['action']
      input_ids, attention_mask = bank.query_from_t5(**item)
      del item
      out_queue.put({'input_ids': input_ids, 'attention_mask': attention_mask})
    elif item['action'] == 'query_then_add':
      del item['action']
      input_ids, attention_mask = bank.query_then_add_from_t5(**item)
      del item
      out_queue.put({'input_ids': input_ids, 'attention_mask': attention_mask, 'size': len(bank)})
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
    input_ids: torch.LongTensor):
    if self.is_distributed:
      if self.rank == self.master_rank:
        query_states, key_states, attention_mask, input_ids = all_gather_tensors(query_states, key_states, attention_mask, input_ids)
        query_states = torch.cat(query_states, dim=0)
        key_states = torch.cat(key_states, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        input_ids = torch.cat(input_ids, dim=0)
      else:
        all_gather_tensors(query_states, key_states, attention_mask, input_ids)
    return query_states, key_states, attention_mask, input_ids
  
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
    n_context: int):
    query_states, key_states, attention_mask, input_ids = self._gather(query_states, key_states, attention_mask, input_ids)
    if not self.is_distributed or self.rank == self.master_rank:
      self.in_queue.put({
        'query_states': query_states.cpu(),
        'key_states': key_states.cpu(),
        'attention_mask': attention_mask.cpu(),
        'input_ids': input_ids.cpu(),
        'n_context': n_context,
        'action': 'add'
      })
      rec = self.out_queue.get()  # get the current size
      self.bank_size = rec['size']
    self._broadcast_bank_size()
    if self.is_distributed:
      torch.distributed.barrier()
  
  def query_from_t5(
    self,
    query_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
    key_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
    attention_mask: torch.BoolTensor,  # (bs * n_context, seq_len, seq_len)
    input_ids: torch.LongTensor,  # (bs * n_context, seq_len)
    n_context: int):
    bs, seq_len = input_ids.size(0) // n_context, input_ids.size(1)
    query_states, key_states, attention_mask, input_ids = self._gather(query_states, key_states, attention_mask, input_ids)
    if not self.is_distributed or self.rank == self.master_rank:
      self.in_queue.put({
        'query_states': query_states.cpu(),
        'key_states': key_states.cpu(),
        'attention_mask': attention_mask.cpu(),
        'input_ids': input_ids.cpu(),
        'n_context': n_context,
        'action': 'query'
      })
      rec = self.out_queue.get()
      input_ids, attention_mask = rec['input_ids'].to(input_ids), rec['attention_mask'].to(attention_mask)
    input_ids, attention_mask = self._scatter(input_ids, attention_mask, bs, seq_len)
    return input_ids, attention_mask
  
  def query_then_add_from_t5(
    self,
    query_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
    key_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
    attention_mask: torch.BoolTensor,  # (bs * n_context, seq_len, seq_len)
    input_ids: torch.LongTensor,  # (bs * n_context, seq_len)
    n_context: int):
    bs, seq_len = input_ids.size(0) // n_context, input_ids.size(1)
    query_states, key_states, attention_mask, input_ids = self._gather(query_states, key_states, attention_mask, input_ids)
    if not self.is_distributed or self.rank == self.master_rank:
      self.in_queue.put({
        'query_states': query_states.cpu(),
        'key_states': key_states.cpu(),
        'attention_mask': attention_mask.cpu(),
        'input_ids': input_ids.cpu(),
        'n_context': n_context,
        'action': 'query_then_add'
      })
      rec = self.out_queue.get()
      # (bs * num_gpu * bank_topk, seq_len) (bs * num_gpu * bank_topk, seq_len, seq_len)
      input_ids, attention_mask = rec['input_ids'].to(input_ids), rec['attention_mask'].to(attention_mask)
      self.bank_size = rec['size']
    self._broadcast_bank_size()
    input_ids, attention_mask = self._scatter(input_ids, attention_mask, bs, seq_len)
    return input_ids, attention_mask
