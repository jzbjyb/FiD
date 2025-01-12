#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for distributed model training
"""

from typing import Any
import pickle
import torch
import torch.distributed as dist


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def get_default_group():
    return dist.group.WORLD


def all_reduce(tensor, group=None):
    if group is None:
        group = get_default_group()
    return dist.all_reduce(tensor, group=group)


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.
    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.
    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
    """
    SIZE_STORAGE_BYTES = 4  # int32 to encode the payload size

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + SIZE_STORAGE_BYTES > max_size:
        raise ValueError(
            'encoded data exceeds max_size, this can be fixed by increasing buffer size: {}'.format(enc_size))

    rank = get_rank()
    world_size = get_world_size()
    buffer_size = max_size * world_size

    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()

    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    assert enc_size < 256 ** SIZE_STORAGE_BYTES, 'Encoded object size should be less than {} bytes'.format(
        256 ** SIZE_STORAGE_BYTES)

    size_bytes = enc_size.to_bytes(SIZE_STORAGE_BYTES, byteorder='big')

    cpu_buffer[0:SIZE_STORAGE_BYTES] = torch.ByteTensor(list(size_bytes))
    cpu_buffer[SIZE_STORAGE_BYTES: enc_size + SIZE_STORAGE_BYTES] = torch.ByteTensor(list(enc))

    start = rank * max_size
    size = enc_size + SIZE_STORAGE_BYTES
    buffer[start: start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size: (i + 1) * max_size]
            size = int.from_bytes(out_buffer[0:SIZE_STORAGE_BYTES], byteorder='big')
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[SIZE_STORAGE_BYTES: size + SIZE_STORAGE_BYTES].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )


def _all_gather_tensor(t: torch.Tensor):
  all_tensors = [torch.empty_like(t) for _ in range(get_world_size())]
  dist.all_gather(all_tensors, t)
  all_tensors[get_rank()] = t
  return all_tensors

def all_gather_tensors(*tt: torch.Tensor):
  tt = [_all_gather_tensor(t) for t in tt]
  return tt

def _all_gather_object(t: Any):
  all_objs = [None for _ in range(get_world_size())]
  dist.all_gather_object(all_objs, t)
  all_objs[get_rank()] = t
  return all_objs

def all_gather_objects(*tt: Any):
  tt = [_all_gather_object(t) for t in tt]
  return tt

def _gather_tensor(t: torch.Tensor, dst: int = 0):
  all_tensors = None
  if get_rank() == dst:
    all_tensors = [torch.empty_like(t) for _ in range(get_world_size())]
  dist.gather(t, all_tensors, dst=dst)
  if get_rank() == dst:
    all_tensors[get_rank()] = t
  return all_tensors

def gather_tensors(*tt: torch.Tensor, dst: int = 0):
  tt = [_gather_tensor(t, dst=dst) for t in tt]
  return tt

def _scatter_tensor(t: torch.Tensor, src: int = 0):
  inp_tensors = None
  if get_rank() == src:
    inp_tensors = list(torch.chunk(t, get_world_size(), dim=0))
    out_tensor = torch.zeros_like(inp_tensors[0])
  else:
    out_tensor = t
  dist.scatter(out_tensor, inp_tensors, src=src)
  return out_tensor

def _scatter_tensor_by_broadcast(t: torch.Tensor, src: int = 0):
  if get_rank() != src:
    t = t.repeat(get_world_size(), *([1] * (t.dim() - 1)))
  dist.broadcast(t, src=src)
  rec = list(torch.chunk(t, get_world_size(), dim=0))[get_rank()]
  return rec

def scatter_tensors(*tt: torch.Tensor, src: int = 0, by_broadcast: bool = False):
  tt = [_scatter_tensor_by_broadcast(t, src=src) if by_broadcast else _scatter_tensor(t, src=src) for t in tt]
  return tt
