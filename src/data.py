# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import torch
import random
import json
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:',
                 add_eos: bool = False):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.eos = (' </s>' if add_eos else '')
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        # TODO: better way to control this?
        import transformers
        add_eos = transformers.__version__ == '3.0.2'
        if 'target' in example:
            target = example['target']
            return target + (' </s>' if add_eos else '')
        elif 'answers' in example:
            return random.choice(example['answers']) +  (' </s>' if add_eos else '')
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question'] + self.eos
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}" + self.eos
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None


        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

def encode_passages_separate(questions: List[str], passages: List[List[str]], tokenizer, max_length, method: str):
  assert len(questions) == len(passages)
  passage_ids = []  # (bs, num_doc, seq_len)
  passage_masks = []  # (bs, num_doc, seq_len)
  passage_sep_masks = []  # (bs, num_doc, seq_len, seq_len)
  for qind in range(len(questions)):
    passage_ids.append([])
    passage_masks.append([])
    passage_sep_masks.append([])
    question = questions[qind]
    # at least include one token from passage in addition to bos and eos
    qids: List[int] = tokenizer.encode(question, add_special_tokens=False)[:max_length - 3]
    for p in passages[qind]:
      pids: List[int] = tokenizer.encode(p, add_special_tokens=True, max_length=max_length - len(qids))
      qpids = qids + pids + [0] * (max_length - len(qids) - len(pids))
      qpattentin_mask = np.zeros((max_length, max_length), dtype=int)
      if method == 'query-side':
        qpattentin_mask[:len(qids), :len(qids) + len(pids)] = 1  # question tokens can see all
        qpattentin_mask[len(qids):, len(qids):len(qids) + len(pids)] = 1  # passage tokens can only see passage
      elif method == 'separate':
        qpattentin_mask[:len(qids), :len(qids)] = 1  # question tokens can only see question
        qpattentin_mask[len(qids):, len(qids):len(qids) + len(pids)] = 1  # passage tokens can only see passage
      else:
        raise NotImplementedError
      passage_ids[-1].append(qpids)
      passage_masks[-1].append([1] * (len(qids) + len(pids)) + [0] * (max_length - len(qids) - len(pids)))
      passage_sep_masks[-1].append(qpattentin_mask)
  passage_ids = torch.tensor(passage_ids)
  passage_masks = torch.tensor(np.array(passage_masks))
  passage_sep_masks = torch.tensor(np.array(passage_sep_masks))
  return passage_ids, passage_masks.bool(), passage_sep_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20, separate_question_passage: str = None):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        assert separate_question_passage in {None, 'query-side', 'separate'}
        # None: full attention mask
        # query-side: query can attend to doc but doc cannot attend to query
        # separate: both query and doc cannot attent to each other
        self.separate_question_passage = separate_question_passage

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding='longest',
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        if self.separate_question_passage:
            questions = [example['question'] for example in batch]
            passages = [example['passages'] for example in batch]
            passage_ids, passage_masks, passage_sep_masks = encode_passages_separate(
              questions, passages, self.tokenizer, self.text_maxlength, method=self.separate_question_passage)
        else:
            def append_question(example):
                if example['passages'] is None:
                    return [example['question']]
                return [example['question'] + " " + t for t in example['passages']]
            text_passages = [append_question(example) for example in batch]
            passage_ids, passage_masks = encode_passages(text_passages,
                                                         self.tokenizer,
                                                         self.text_maxlength)
            passage_sep_masks = None

        return (index, target_ids, target_mask, passage_ids, passage_masks, passage_sep_masks)

def load_data(data_path=None, global_rank=-1, world_size=-1, n_context=None):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if n_context:
          if len(example['ctxs']) < n_context:
            continue
          example['ctxs'] = example['ctxs'][:n_context]
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples

class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            padding='max_length',
            return_tensors='pt',
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['passages'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['passages'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 title_prefix='title:',
                 passage_prefix='context:',
                 add_eos: bool = False):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.eos = (' </s>' if add_eos else '')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + \
            self.passage_prefix + " " + example[1] + self.eos
        return example[0], text

class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            padding='max_length',
            return_tensors='pt',
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask
