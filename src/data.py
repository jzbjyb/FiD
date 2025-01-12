# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Union
import torch
import random
import json
import numpy as np
from multiprocessing import Manager
from src.evaluation import has_answer, SimpleTokenizer

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:',
                 add_eos: bool = False,
                 in_batch_negative_mimic: int = 0,
                 augmentation: str = None,
                 join_multiple_answer: str = None):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.eos = (' </s>' if add_eos else '')
        self.in_batch_negative_mimic = in_batch_negative_mimic
        assert augmentation in {'duplicate', 'mask', None}
        self.augmentation = augmentation
        self.join_multiple_answer = join_multiple_answer
        if self.join_multiple_answer and len(self.join_multiple_answer) == 1:
            self.join_multiple_answer = self.join_multiple_answer + ' '  # append space
        self.max_num_answer = 10
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        # TODO: better way to control this?
        import transformers
        add_eos = transformers.__version__ == '3.0.2'
        if 'target' in example:
            final_ans = example['target']
        elif 'answers' in example:
            assert type(example['answers']) is list
            if type(example['answers'][0]) is list:  # has alias
                ans_li = [a[0] for a in example['answers']]  # use the first alias as target
            else:
                ans_li = example['answers']
            if self.join_multiple_answer:
                final_ans = self.join_multiple_answer.join(ans_li[:self.max_num_answer])
            else:
                final_ans = random.choice(ans_li)
        else:
            return None
        final_ans = final_ans + (' </s>' if add_eos else '')
        return final_ans
    
    def augment(self, text: str):
        if self.augmentation is None:
            return text
        if self.augmentation == 'duplicate':
            return text + ' ' + text
        if self.augmentation == 'mask':
            return text + ' ' + ' '.join([f'<extra_id_{i}>' for i in range(5)])  # TODO: hyperparam
        raise NotImplementedError

    def __getitem__(self, index):
        example = self.data[index]
        question = self.augment(self.question_prefix + " " + example['question']) + self.eos
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}" + self.eos
            if self.in_batch_negative_mimic:
              sampled_idxs = random.sample(range(len(self)), self.in_batch_negative_mimic)
              sampled_idxs = list(set(sampled_idxs) - {index})[:self.in_batch_negative_mimic - 1]
              contexts = []
              for idx in ([index] + sampled_idxs):
                contexts.extend(self.data[idx]['ctxs'])
              contexts = contexts[:self.n_context]
            else:
              contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            passage_ids = [str(c['id']) for c in contexts]
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
            assert len(contexts) == self.n_context
        else:
            passages, scores, passage_ids = None, None, None

        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores,
            'passage_ids': passage_ids
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
      if method == 'no-query':
        pids: List[int] = tokenizer.encode(p, add_special_tokens=True, max_length=max_length)
        pattn_mask = np.zeros((max_length, max_length), dtype=int)
        pattn_mask[:, :len(pids)] = 1
        passage_ids[-1].append(pids + [0] * (max_length - len(pids)))
        passage_masks[-1].append(pattn_mask[0])
        passage_sep_masks[-1].append(pattn_mask)
      else:
        pids: List[int] = tokenizer.encode(p, add_special_tokens=True, max_length=max_length - len(qids))
        qpids = qids + pids + [0] * (max_length - len(qids) - len(pids))
        qpattn_mask = np.zeros((max_length, max_length), dtype=int)
        if method == 'query-side':
          qpattn_mask[:len(qids), :len(qids) + len(pids)] = 1  # question tokens can see all
          qpattn_mask[len(qids):, len(qids):len(qids) + len(pids)] = 1  # passage tokens can only see passage
        elif method == 'separate':
          qpattn_mask[:len(qids), :len(qids)] = 1  # question tokens can only see question
          qpattn_mask[len(qids):, len(qids):len(qids) + len(pids)] = 1  # passage tokens can only see passage
        else:
          raise NotImplementedError
        passage_ids[-1].append(qpids)
        passage_masks[-1].append([1] * (len(qids) + len(pids)) + [0] * (max_length - len(qids) - len(pids)))
        passage_sep_masks[-1].append(qpattn_mask)
  passage_ids = torch.tensor(passage_ids)
  passage_masks = torch.tensor(np.array(passage_masks))
  passage_sep_masks = torch.tensor(np.array(passage_sep_masks))
  return passage_ids, passage_masks.bool(), passage_sep_masks.bool()

class Collator(object):
    def __init__(self,
                 text_maxlength,
                 tokenizer,
                 answer_maxlength=20,
                 separate_query_passage: str = None,
                 query_in_decoder: str = 'no'):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        assert separate_query_passage in {None, 'query-side', 'separate', 'no-query'}
        # None: full attention mask
        # query-side: query can attend to doc but doc cannot attend to query
        # separate: both query and doc cannot attent to each other
        # no-query: only include doc
        self.separate_query_passage = separate_query_passage
        assert query_in_decoder in {'no', 'all'}
        # no: no query in decoder
        # all: decode query and answer
        self.query_in_decoder = query_in_decoder

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])
        scores = torch.stack([ex['scores'] for ex in batch], dim=0)

        # decoder
        decoder_start_token = self.tokenizer.pad_token  # T5 uses pad as the start token of decoding
        assert batch[0]['target'] != None
        target = ['{} {} {}'.format(ex['question'], decoder_start_token, ex['target'])
                  if self.query_in_decoder != 'no' else ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding='longest',
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target['input_ids']
        target_mask = target['attention_mask'].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        # encoder
        if self.separate_query_passage:
            questions = [example['question'] for example in batch]
            passages = [example['passages'] for example in batch]
            passage_ids, passage_masks, passage_sep_masks = encode_passages_separate(
              questions, passages, self.tokenizer, self.text_maxlength, method=self.separate_query_passage)
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
        
        # passage ids (from memory bank) 
        passage_idx: np.ndarray = np.array([example['passage_ids'] for example in batch], dtype=str)
        
        return index, target_ids, target_mask, passage_ids, passage_masks, passage_sep_masks, passage_idx, scores

def load_data(
    data_path=None, 
    global_rank=-1, 
    world_size=-1, 
    n_context=None, 
    use_gold_doc_dist: bool = False):
    tokenizer = SimpleTokenizer()
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
        if global_rank > -1 and not k % world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = str(k)
        for c in example['ctxs']:
            if use_gold_doc_dist:
                c['score'] = float(has_answer(example['answers'], c['text'], tokenizer))
            elif not 'score' in c:
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

class ContextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data: Union[List[Tuple[str, str, str]], np.ndarray],
                 title_prefix: str = 'title:',
                 passage_prefix: str = 'context:',
                 add_eos: bool = False):
        if type(data) is list and False:  # TODO: avoid copy-on-access
            manager = Manager()
            self.data = manager.list([x for x in data])
        else:
            self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.eos = ' </s>' if add_eos else ''  # TODO: add tokenizer-specific eos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[str, str]:
        example = self.data[index]
        is_byte = type(example[0]) is np.bytes_
        if is_byte:
            example = [x.decode('utf-8') for x in example]
        id = example[0]
        text = f'{self.title_prefix} {example[2]} {self.passage_prefix} {example[1]}{self.eos}'
        return id, text

class QuestionDataset(torch.utils.data.Dataset):
  def __init__(self,
               data: List[Tuple[str, str]],
               question_prefix: str = 'question:',
               add_eos: bool = False):
    self.data = data
    self.question_prefix = question_prefix
    self.eos = ' </s>' if add_eos else ''

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index) -> Tuple[str, str]:
    example = self.data[index]
    id = example[0]
    text = f'{self.question_prefix} {example[1]}{self.eos}'
    return id, text

class TextCollator(object):
    def __init__(self, tokenizer, maxlength: int = 200, augmentation: str = None):
        assert augmentation in {'duplicate', 'mask', None}
        self.tokenizer = tokenizer
        self.maxlength = maxlength
        self.augmentation = augmentation
    
    def augment(self, text: str):
        if self.augmentation is None:
            return text
        if self.augmentation == 'duplicate':
            return text + ' ' + text
        if self.augmentation == 'mask':
            return text + ' ' + ' '.join([f'<extra_id_{i}>' for i in range(5)])  # TODO: hyperparam
        raise NotImplementedError

    def __call__(self, batch):
        index = [x[0] for x in batch]
        texts: List[str] = [self.augment(x[1]) for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            texts,
            padding='max_length',
            return_tensors='pt',
            max_length=self.maxlength,
            truncation=True)
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()
        return index, texts, text_ids, text_mask
