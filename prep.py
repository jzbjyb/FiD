from typing import List, Tuple, Dict, Callable
import argparse
import json
import os
import re
import numpy as np
import time
import random
import logging
import copy
import statistics
from tqdm import tqdm
import scipy.stats
from collections import defaultdict
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from passage_retrieval import validate

logging.basicConfig(level=logging.DEBUG)


class BEIRDataset:
  def __init__(self, root_dir: str, name: str):
    self.name = name
    self.qid2answer: Dict[str, str] = self.load_query(os.path.join(root_dir, 'queries.jsonl'))

  def get_answer_scifact(cls, metadata: Dict):
    labels: List[str] = []
    for did, sents in metadata.items():
      for sent in sents:
        labels.append(sent['label'])
    if len(labels) == 0:
      return 'uncertain'
    return labels[0].lower()

  def get_answer_techqa(cls, metadata: Dict):
    return metadata['answer']

  def get_answer_sciq(cls, metadata: Dict):
    return metadata['answer']

  def load_query(self, filename: str):
    qid2answer: Dict[str, str] = {}
    with open(filename, 'r') as fin:
      for l in fin:
        l = json.loads(l)
        id, text, metadata = l['_id'], l['text'], l['metadata']
        ans = getattr(self, f'get_answer_{self.name}')(metadata)
        qid2answer[id] = ans
    return qid2answer


def aggregate_ctxs(json_files: List[str], out_file: str):
  assert out_file.endswith('.tsv'), 'plz use .tsv as extension'

  id2ctx: Dict[str, Tuple[str, str]] = {}
  for json_file in json_files:
    data = json.load(open(json_file))
    for example in data:
      for ctx in example['ctxs']:
        if ctx['id'] in id2ctx:
          continue
        id2ctx[ctx['id']] = (ctx['title'], ctx['text'])

  with open(out_file, 'w') as fout:
    fout.write(f'id\ttext\ttitle\n')
    for id, (title, text) in id2ctx.items():
      fout.write(f'{id}\t{text}\t{title}\n')


def eval(beir_dir: str, ret_file: str, topks: List[int] = [1, 5, 10, 100]):
  qid2dids: Dict[str, List[str]] = defaultdict(list)
  ret = json.load(open(ret_file, 'r'))
  for example in ret:
    for d in example['ctxs']:
      qid2dids[example['question']].append(d['id'])
  corpus, queries, qrels = GenericDataLoader(data_folder=beir_dir).load(split='test')
  qid2dids_gold: Dict[str, List[str]] = defaultdict(list)
  for qid in qrels:
    query = queries[qid]
    qid2dids_gold[query].extend([did for did in qrels[qid]])
    # print(qid, query, qid2dids_gold[query], corpus[qid2dids_gold[query][0]])
    # input()

  topk2has = defaultdict(list)
  for qid in qid2dids:
    for topk in topks:
      preds = set(qid2dids[qid][:topk])
      gold = set(qid2dids_gold[qid])
      has = len(preds & gold) > 0
      topk2has[topk].append(has)

  for topk in topks:
    print(topk, np.mean(topk2has[topk]))


def convert_beir_to_fid_format(beir_dir: str, out_dir: str, dataset_name: str, splits: List[str], topk: int = 100):
  bert_data = BEIRDataset(beir_dir, name=dataset_name)

  # build index
  hostname = 'localhost'
  number_of_shards = 1  # TODO
  corpus, _, _ = GenericDataLoader(data_folder=beir_dir).load(split=splits[0])
  model = BM25(index_name=dataset_name, hostname=hostname, initialize=True, number_of_shards=number_of_shards)
  model.index(corpus)
  time.sleep(5)

  for split_ind, split in enumerate(splits):
    corpus, queries, qrels = GenericDataLoader(data_folder=beir_dir).load(split=split)

    # retrieve
    model = BM25(index_name=dataset_name, hostname=hostname, initialize=False, number_of_shards=number_of_shards)
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)
    print(f'retriever evaluation for k in: {retriever.k_values}')
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)  # TODO
    print(ndcg, _map, recall, precision)

    # output
    os.makedirs(out_dir, exist_ok=True)
    if split_ind == 0:
      with open(os.path.join(out_dir, 'psgs.tsv'), 'w') as fout:
        fout.write('id\ttext\ttitle\n')
        for did in corpus:
          title = corpus[did].get('title')
          text = corpus[did].get('text')
          fout.write(f'{did}\t{text}\t{title}\n')

    examples: List[Dict] = []
    for qid, scores_dict in results.items():
      answer = bert_data.qid2answer[qid]
      query = queries[qid]
      example = {'question': query, 'answers': [answer], 'ctxs': []}
      scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)[:topk]
      for rank in range(len(scores)):
        did = scores[rank][0]
        title = corpus[did].get('title')
        text = corpus[did].get('text')
        example['ctxs'].append({'id': did, 'title': title, 'text': text})
      examples.append(example)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(examples, open(os.path.join(out_dir, f'{split}.json'), 'w'), indent=2)


def save_beir_format(beir_dir: str,
                     qid2dict: Dict[str, Dict],
                     did2dict: Dict[str, Dict],
                     split2qiddid: Dict[str, List[Tuple[str, str]]]):
  # save
  os.makedirs(beir_dir, exist_ok=True)
  with open(os.path.join(beir_dir, 'queries.jsonl'), 'w') as fout:
    for qid in qid2dict:
      fout.write(json.dumps(qid2dict[qid]) + '\n')
  with open(os.path.join(beir_dir, 'corpus.jsonl'), 'w') as fout:
    for did in did2dict:
      fout.write(json.dumps(did2dict[did]) + '\n')
  os.makedirs(os.path.join(beir_dir, 'qrels'), exist_ok=True)
  for split in split2qiddid:
    with open(os.path.join(beir_dir, 'qrels', f'{split}.tsv'), 'w') as fout:
      fout.write('query-id\tcorpus-id\tscore\n')
      for qid, did in split2qiddid[split]:
        fout.write(f'{qid}\t{did}\t1\n')


def convert_sciq_to_beir_format(sciq_dir: str, beir_dir: str):
  qid2dict: Dict[str, Dict] = {}
  did2dict: Dict[str, Dict] = {}
  unique_text_set: Set[str] = set()
  split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
  for split, nsplit in [('train', 'train'), ('valid', 'dev'), ('test', 'test')]:
    with open(os.path.join(sciq_dir, f'{split}.json'), 'r') as fin:
      data = json.load(fin)
      for ex in data:
        if 'support' not in ex or len(ex['support'].strip()) <= 0:
          continue
        qid = f'{str(len(qid2dict) + len(did2dict))}'
        choices = [ex['correct_answer'], ex['distractor1'], ex['distractor2'], ex['distractor3']]
        qid2dict[qid] = {'_id': qid, 'text': ex['question'], 'metadata': {'choices': choices, 'answer': choices[0]}}
        did = f'{str(len(qid2dict) + len(did2dict))}'
        did2dict[did] = {'_id': did, 'title': '', 'text': ex['support']}
        unique_text_set.add(ex['support'])
        split2qiddid[nsplit].append((qid, did))
  save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)


def convert_techqa_to_beir_format(techqa_dir: str, beir_dir: str, question_field: str):
  pattern = re.compile(r'\s+')
  def clean(text):
    return re.sub(pattern, ' ', text).strip()
  qid2dict: Dict[str, Dict] = {}
  did2dict: Dict[str, Dict] = {}
  swgid2did: Dict[str, str] = {}
  split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
  for split, nsplit in [('training', 'train'), ('dev', 'dev')]:
    with open(os.path.join(techqa_dir, 'training_and_dev', f'{split}_Q_A.json'), 'r') as fin:
      data = json.load(fin)
      for ex in data:
        if ex['ANSWERABLE'] == 'N':
          continue
        qid = f'{str(len(qid2dict) + len(did2dict))}'
        if question_field == 'title':
          question = clean(ex['QUESTION_TITLE'])
        elif question_field == 'text':
          question = clean(ex['QUESTION_TEXT'])
        elif question_field == 'all':
          question = clean(ex['QUESTION_TITLE'] + ' ' + ex['QUESTION_TEXT'])
        else:
          raise NotImplementedError
        answer = clean(ex['ANSWER'])
        gold_swgid = ex['DOCUMENT']
        qid2dict[qid] = {
          '_id': qid,
          'text': question,
          'metadata': {'document': gold_swgid, 'answer': answer, 'offset': [ex['START_OFFSET'], ex['END_OFFSET']]}}
        if gold_swgid not in swgid2did:
          did = f'{str(len(qid2dict) + len(did2dict))}'
          did2dict[did] = {}  # placeholder
          swgid2did[gold_swgid] = did
        split2qiddid[nsplit].append((qid, swgid2did[gold_swgid]))
  with open(os.path.join(techqa_dir, 'training_and_dev', 'training_dev_technotes.json'), 'r') as fin:
    data = json.load(fin)
    for swgid, doc in data.items():
      if swgid not in swgid2did:
        did = f'{str(len(qid2dict) + len(did2dict))}'
        did2dict[did] = {}  # placeholder
        swgid2did[swgid] = did
      did = swgid2did[swgid]
      did2dict[did] = {'_id': did, 'title': clean(doc['title']), 'text': clean(doc['text'])}
  save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)


def convert_quasar_to_beir_format(quasar_dir: str, beir_dir: str):
  qid2dict: Dict[str, Dict] = {}
  did2dict: Dict[str, Dict] = {}
  text2did: Dict[str, str] = {}
  split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
  num_docs = 0
  for split, nsplit in [('train', 'train'), ('dev', 'dev'), ('test', 'test')]:
    with open(os.path.join(quasar_dir, 'questions', f'{split}_questions.json'), 'r') as qfin, \
         open(os.path.join(quasar_dir, 'context', 'long', f'{split}_contexts.json'), 'r') as cfin:
      for l in tqdm(qfin):
        question = json.loads(l)
        answer = question['answer']
        docs = json.loads(cfin.readline())
        assert question['uid'] == docs['uid']
        if 'yes-answer-long' not in question['tags']:
          continue
        qid = f'{str(len(qid2dict) + len(did2dict))}'
        qid2dict[qid] = {'_id': qid, 'text': question['question'], 'metadata': {'answer': answer}}
        num_docs += len(docs['contexts'])
        for score, doc in docs['contexts']:
          if doc not in text2did:
            did = f'{str(len(qid2dict) + len(did2dict))}'
            text2did[doc] = did
            did2dict[did] = {}
          did = text2did[doc]
          did2dict[did] = {'_id': did, 'title': '', 'text': doc}
          if answer.lower() in doc.lower():
            split2qiddid[nsplit].append((qid, did))  # TODO: this annotation is noisy and also not complete
  save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)
  print(f'#docs {len(text2did)}')


def eval_answer(ret_file: str, sort: bool = False, shuffle: bool = False, topk: int = 100, key_func: Callable = lambda x: x['score']):
  #score_len_li: List[Tuple[float, int]] = []
  with open(ret_file, 'r') as fin:
    data = json.load(fin)
    # aggregate ctx of the same query
    query2example: Dict[str, Dict] = {}
    for example in data:
      if example['question'] not in query2example:
        query2example[example['question']] = example
        continue
      query2example[example['question']]['ctxs'].extend(example['ctxs'])
    data = list(query2example.values())
    for example in data:
      #for ctx in example['ctxs']:
      #  score_len_li.append((float(key_func(ctx)), len(ctx['text'])))
      if sort:
        example['ctxs'] = sorted(example['ctxs'], key=lambda x: float(key_func(x)), reverse=True)
      if shuffle:
        random.shuffle(example['ctxs'])
      example['ctxs'] = example['ctxs'][:topk]
    #print('score length correlation', scipy.stats.pearsonr(*list(zip(*score_len_li))))
    validate(data, 32)


def eval_variance(ret_file: str, key_func: Callable = lambda x: x['score']):
  with open(ret_file, 'r') as fin:
    data = json.load(fin)
    # aggregate ctx of the same query
    query2example: Dict[str, Dict] = {}
    for example in data:
      if example['question'] not in query2example:
        query2example[example['question']] = example
        continue
      query2example[example['question']]['ctxs'].extend(example['ctxs'])
    data = list(query2example.values())
    vars: List[float] = []
    for example in data:
      scores = [float(key_func(ctx)) for ctx in example['ctxs']]
      vars.append(statistics.variance(scores))
  print(f'avg variance {np.mean(vars)}')


def create_whole_test(data_dir: str, out_num: int = None, topk: int = 100):
  test_file = os.path.join(data_dir, 'test.json')
  psgs_file = os.path.join(data_dir, 'psgs.tsv')
  out_test_file = os.path.join(data_dir, 'test.all.json' if out_num is None else f'test.all.{out_num}.json')
  id2titletext: Dict[str, Tuple[str, str]] = {}
  with open(psgs_file, 'r') as fin:
    fin.readline()
    for l in fin:
      id, text, title = l.rstrip('\n').split('\t')
      id2titletext[id] = (title, text)
  ids = list(id2titletext.keys())
  with open(test_file, 'r') as fin, open(out_test_file, 'w') as fout:
    data = json.load(fin)
    if out_num:
      random.shuffle(data)
      data = data[:out_num]
    new_data = []
    for example in tqdm(data):
      for batch_ids in range(0, len(ids), topk):
        batch_ids = ids[batch_ids:batch_ids + topk]
        _example = {'question': example['question'], 'answers': example['answers']}
        _example['ctxs'] = [{'id': id, 'title': id2titletext[id][0], 'text': id2titletext[id][1]} for id in batch_ids]
        new_data.append(_example)
    json.dump(new_data, fout, indent=2)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='preprocessing')
  parser.add_argument('--task', type=str, choices=[
    'convert_sciq_to_beir_format', 'convert_techqa_to_beir_format', 'convert_quasar_to_beir_format',
    'aggregate_ctxs', 'eval', 'eval_variance', 'convert_beir_to_fid_format', 'eval_answer', 'create_whole_test'])
  parser.add_argument('--inp', type=str, help='input file', nargs='+')
  parser.add_argument('--out', type=str, help='output file', nargs='+')
  args = parser.parse_args()

  if args.task == 'aggregate_ctxs':
    json_files: List[str] = args.inp
    out_file = args.out[0]
    aggregate_ctxs(json_files, out_file)

  elif args.task == 'convert_beir_to_fid_format':
    beir_dir = args.inp[0]
    out_dir = args.out[0]
    convert_beir_to_fid_format(beir_dir, out_dir, dataset_name='techqa', splits=['dev', 'train'])

  elif args.task == 'eval':
    beir_dir, ret_file = args.inp
    eval(beir_dir, ret_file)

  elif args.task == 'convert_sciq_to_beir_format':
    sciq_dir = args.inp[0]
    beir_dir = args.out[0]
    convert_sciq_to_beir_format(sciq_dir, beir_dir)

  elif args.task == 'convert_techqa_to_beir_format':
    techqa_dir = args.inp[0]
    beir_dir = args.out[0]
    convert_techqa_to_beir_format(techqa_dir, beir_dir, question_field='all')

  elif args.task == 'convert_quasar_to_beir_format':
    quasar_dir = args.inp[0]
    beir_dir = args.out[0]
    convert_quasar_to_beir_format(quasar_dir, beir_dir)

  elif args.task == 'eval_answer':
    key = 'score'  # score two_tower_attn_score
    method = 'avg'
    n_two_tower_layers = 6
    num_heads = 12

    ret_file = args.inp[0]
    if method == 'default':
      eval_answer(ret_file, sort=False)
      exit()
    elif method == 'shuffle':
      eval_answer(ret_file, shuffle=True)
      exit()
    elif method == 'avg':
      eval_answer(ret_file, sort=True, key_func=lambda x: np.mean(x[key]))
      eval_answer(ret_file, sort=True, key_func=lambda x: np.mean(x[key][-1]))
      exit()
    elif method == 'avg_head':
      for i in range(num_heads):
        print(f'head {i}')
        eval_answer(ret_file, sort=True, key_func=lambda x: np.mean(x[key][11][i]))
      exit()
    elif method == 'specific':
      eval_answer(ret_file, sort=True, key_func=lambda x: np.mean(x[key][0][3]))
      exit()

    for l in range(12 - n_two_tower_layers):
      if method == 'avg_layer':
        eval_answer(ret_file, sort=True, key_func=lambda x: np.mean(x[key][l]))
        continue
      for i in range(num_heads):
        print(f'layer {l}, head {i}')
        eval_answer(ret_file, sort=True, key_func=lambda x: np.mean(x[key][l][i]))

  elif args.task == 'eval_variance':
    ret_file = args.inp[0]
    n_two_tower_layers = 9
    num_heads = 12
    for l in range(12 - n_two_tower_layers):
      for i in range(num_heads):
        print(f'layer {l}, head {i}')
        eval_variance(ret_file, key_func=lambda x: x[f'two_tower_attn_score_{l}'][i])

  elif args.task == 'create_whole_test':
    data_dir = args.inp[0]
    create_whole_test(data_dir, out_num=100)
