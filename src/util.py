# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, List
import os
import errno
import torch
import sys
import logging
import json
import contextlib
from pathlib import Path
import torch.distributed as dist
from fairscale.optim.oss import OSS
import csv
import wandb

logger = logging.getLogger(__name__)

clean_text_for_tsv = lambda x: '' if x is None else x.replace('\n', ' ').replace('\t', ' ')

@contextlib.contextmanager
def open_file(path_to_file: str, mode: str = 'r'):
    if path_to_file is None:
      yield None
    else:
      with open(path_to_file, mode) as fin:
        yield fin

class WandbLogger:
  _wandb_logger = None
  _step = None

  @classmethod
  def init(cls, opt):
    if opt.wandb_entity and opt.wandb_name.split('/')[-1] != 'test':
      cls._wandb_logger = wandb.init(entity=opt.wandb_entity, project=opt.wandb_project, name=opt.wandb_name)

  @classmethod
  def enabled(cls):
    return cls._wandb_logger is not None

  @classmethod
  def step(cls, step: int):
    cls._step = step

  @classmethod
  def log_w_step(cls, data):
    if not cls.enabled():
      return
    _data = {}
    for k in data:
      if type(data[k]) is torch.Tensor:
        data[k] = data[k].detach().cpu().numpy().tolist()
      if type(data[k]) is list:
        for i in range(len(data[k])):
          _data[f'{k}-{i}'] = data[k][i]
      else:
        _data[k] = data[k]
    cls._wandb_logger.log(_data, step=cls._step)

def init_logger(is_main=True, is_distributed=False, filename=None):
    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    for h in handlers:
      logger.addHandler(h)
    logger.setLevel(logging.INFO if is_main else logging.WARN)
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
    return logger

def get_checkpoint_path(opt):
    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path, checkpoint_exists

def symlink_force(target, link_name):
    target_last = os.path.basename(os.path.normpath(target))
    try:
        os.symlink(target_last, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target_last, link_name)
        else:
            raise e

def save(model, optimizer, scheduler, step, best_eval_metric, opt, dir_path, name):
    model_to_save = model.module if hasattr(model, "module") else model
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name) #"step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    model_to_save.save_pretrained(epoch_path)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "optimizer.pth.tar")
    checkpoint = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "opt": opt,
        "best_eval_metric": best_eval_metric,
    }
    torch.save(checkpoint, fp)
    symlink_force(epoch_path, cp)


def load(model_class, dir_path, opt, reset_params=False):
    epoch_path = os.path.realpath(dir_path)
    optimizer_path = os.path.join(epoch_path, "optimizer.pth.tar")
    logger.info("Loading %s" % epoch_path)
    model = model_class.from_pretrained(epoch_path)
    model = model.to(opt.device)
    logger.info("loading checkpoint %s" %optimizer_path)
    checkpoint = torch.load(optimizer_path, map_location=opt.device)
    opt_checkpoint = checkpoint["opt"]
    step = checkpoint["step"]
    if "best_eval_metric" in checkpoint:
        best_eval_metric = checkpoint["best_eval_metric"]
    else:
        best_eval_metric = checkpoint["best_dev_em"]
    if not reset_params:
        optimizer, scheduler = set_optim(opt_checkpoint, model, sharding=opt.use_sharding)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler = set_optim(opt, model, sharding=opt.use_sharding)

    return model, optimizer, scheduler, opt_checkpoint, step, best_eval_metric

class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, fixed_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio)*step/float(max(1, self.warmup_steps)) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(0.0,
            1.0 + (self.min_ratio - 1) * (step - self.warmup_steps)/float(max(1.0, self.scheduler_steps - self.warmup_steps)),
        )


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    def lr_lambda(self, step):
        return 1.0


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate


def set_optim(opt, model, sharding: bool = False):
  if opt.optim == 'adam':
    params = {'lr': opt.lr}
    base_optim = torch.optim.Adam
  elif opt.optim == 'adamw':
    params = {'lr': opt.lr, 'weight_decay': opt.weight_decay}
    base_optim = torch.optim.AdamW
  else:
    raise NotImplementedError
  if sharding:
    optimizer = OSS(params=model.parameters(), optim=base_optim, **params)
  else:
    optimizer = base_optim(model.parameters(), **params)

  if opt.scheduler == 'fixed':
    scheduler = FixedScheduler(optimizer)
  elif opt.scheduler == 'linear':
    if opt.scheduler_steps is None:
        scheduler_steps = opt.total_steps
    else:
        scheduler_steps = opt.scheduler_steps
    scheduler = WarmupLinearScheduler(optimizer, warmup_steps=opt.warmup_steps, scheduler_steps=scheduler_steps, min_ratio=0., fixed_lr=opt.fixed_lr)
  else:
    raise NotImplementedError
  return optimizer, scheduler


def average_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
        if opt.is_main:
            x = x / opt.world_size
    return x


def sum_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x


def weighted_average(x, count, opt):
    if not opt.is_distributed:
        return x, count
    t_loss = torch.tensor([x * count], device=opt.device)
    t_total = torch.tensor([count], device=opt.device)
    t_loss = sum_main(t_loss, opt)
    t_total = sum_main(t_total, opt)
    return (t_loss / t_total).item(), t_total.item()


def write_output(glob_path, output_path):
    files = list(glob_path.glob('*.txt'))
    files.sort()
    print(f'merge predictions from {len(files)} files')
    with open(output_path, 'w') as outfile:
        for path in files:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    outfile.write(line)
            path.unlink()
    glob_path.rmdir()


def save_distributed_dataset(data, opt):
    dir_path = Path(opt.checkpoint_dir) / opt.name
    write_path = dir_path / 'tmp_dir'
    write_path.mkdir(exist_ok=True)
    tmp_path = write_path / f'{opt.global_rank}.json'
    with open(tmp_path, 'w') as fw:
        json.dump(data, fw)
    if opt.is_distributed:
        torch.distributed.barrier()
    if opt.is_main:
        final_path = dir_path / 'dataset_wscores.json'
        logger.info(f'Writing dataset with scores at {final_path}')
        glob_path = write_path / '*'
        results_path = write_path.glob('*.json')
        alldata = []
        for path in results_path:
            with open(path, 'r') as f:
                data = json.load(f)
            alldata.extend(data)
            path.unlink()
        with open(final_path, 'w') as fout:
            json.dump(alldata, fout, indent=4)
        write_path.rmdir()

def load_passages(path) -> List[Tuple[str, str, str]]:  # id, text, title
    if not os.path.exists(path):
        logger.info(f'{path} does not exist')
        return
    logger.info(f'Loading passages from: {path}')
    passages = []
    with open(path) as fin:
        reader = csv.reader(fin, delimiter='\t')
        header = next(reader)
        assert len(header) == 3 and header[0] == 'id', 'header format error'
        textfirst = header[1] == 'text'
        for k, row in enumerate(reader):
            try:
                if textfirst:
                    passages.append((row[0], row[1], row[2]))
                else:
                    passages.append((row[0], row[2], row[1]))
            except:
                logger.warning(f'The following input line has not been correctly loaded: {row}')
    return passages

def extract_query_answer(output: List[int], tokenizer, query_in_decoder: bool = False) -> Tuple[str, str, int]:
  pad_id = tokenizer.pad_token_id
  splits: List[List[int]] = []
  splits_start_position: List[int] = []
  for i, t in enumerate(output):
    if t != pad_id:
      splits[-1].append(t)
    elif len(splits) == 0 or len(splits[-1]) > 0:
      splits.append([])
      splits_start_position.append(i)
  if len(splits[-1]) == 0:
    del splits[-1]
    del splits_start_position[-1]
  if query_in_decoder == 'no':  # only answer
    query = ''
    ans = tokenizer.decode(splits[0], skip_special_tokens=True)
    ans_position = splits_start_position[0]
  elif len(splits) == 1:
    # only query
    # this is the case where the decoder fails to generate answers
    # but we still use the first pad as ans_position to collect decoder attn
    query = tokenizer.decode(splits[0], skip_special_tokens=True)
    ans = ''
    ans_position = splits_start_position[0]
  else:  # query & ?? & answer
    query = tokenizer.decode(splits[0], skip_special_tokens=True)
    ans = tokenizer.decode(splits[-1], skip_special_tokens=True)
    ans_position = splits_start_position[-1]
  return query, ans, ans_position


def max_sparsify(
     scores: torch.FloatTensor,  # (any)
     mask: torch.BoolTensor,  # (any)
     dim: int,
     inplace: bool = False):
  value, ind = (scores * mask - (~mask * 1e10)).max(dim)  # make masked position -inf
  valid = value.gt(-1e10)
  value *= valid
  value, ind, valid = value.unsqueeze(dim), ind.unsqueeze(dim), valid.unsqueeze(dim)
  if inplace:
    scores[:] = 0
    scores.scatter_(dim, ind, value)
    mask[:] = False
    mask.scatter_(dim, ind, valid)
    return scores, mask
  else:
    _scores = torch.zeros_like(scores)
    _mask = torch.zeros_like(scores).to(mask)
    _scores.scatter_(dim, ind, value)
    _mask.scatter_(dim, ind, valid)
    return _scores, _mask
