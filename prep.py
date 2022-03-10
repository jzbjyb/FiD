from typing import List, Tuple, Dict, Callable, Union, Set, Any
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
from src.util import clean_text_for_tsv
import src.evaluation

logging.basicConfig(level=logging.DEBUG)


class BEIRDataset:
  def __init__(self, root_dir: str, name: str):
    self.name = name
    self.qid2answer: Dict[str, Any] = self.load_query(os.path.join(root_dir, 'queries.jsonl'))

  @classmethod
  def get_answer_scifact(cls, metadata: Dict) -> List[str]:
    labels: List[str] = []
    for did, sents in metadata.items():
      for sent in sents:
        labels.append(sent['label'])
    if len(labels) == 0:
      return ['uncertain']
    return [labels[0].lower()]

  @classmethod
  def get_answer_techqa(cls, metadata: Dict) -> List[str]:
    return [metadata['answer']]

  @classmethod
  def get_answer_sciq(cls, metadata: Dict) -> List[str]:
    return [metadata['answer']]

  @classmethod
  def get_answer_bioasq(cls, metadata: Dict) -> List[List[str]]:
    if metadata['type'] == 'yesno':
      return [[metadata['answer']]]
    elif metadata['type'] in {'list', 'factoid'}:
      return metadata['answer']
    elif metadata['type'] == 'summary':
      return metadata['answer']
    else:
      raise NotImplementedError

  @classmethod
  def get_answer_msmarcoqa(cls, metadata: Dict) -> List[str]:
    return metadata['answer']

  def load_query(self, filename: str):
    qid2answer: Dict[str, Any] = {}
    with open(filename, 'r') as fin:
      for l in fin:
        l = json.loads(l)
        id, text, metadata = l['_id'], l['text'], l['metadata']
        ans = getattr(self, f'get_answer_{self.name}')(metadata)
        if ans is None:
          continue
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
      with open(os.path.join(out_dir, 'psgs.tsv'), 'w') as fout, \
           open(os.path.join(out_dir, 'line2docid.tsv'), 'w') as l2dfout:
        fout.write('id\ttext\ttitle\n')
        for lid, did in enumerate(corpus):
          title = clean_text_for_tsv(corpus[did].get('title'))
          text = clean_text_for_tsv(corpus[did].get('text'))
          fout.write(f'{did}\t{text}\t{title}\n')
          l2dfout.write(f'{lid}\t{did}\n')

    examples: List[Dict] = []
    for qid, scores_dict in results.items():
      answer = bert_data.qid2answer[qid]
      query = clean_text_for_tsv(queries[qid])
      example = {'question': query, 'id': qid, 'answers': answer, 'ctxs': []}
      scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)[:topk]
      for rank in range(len(scores)):
        did = scores[rank][0]
        title = clean_text_for_tsv(corpus[did].get('title'))
        text = clean_text_for_tsv(corpus[did].get('text'))
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


def convert_msmarcoqa_to_beir_fid_format(msmarcoqa_dir: str,
                                         beir_dir: str,
                                         fid_dir: str,
                                         filter_func: Callable = None):
  qid2dict: Dict[str, Dict] = {}
  did2dict: Dict[str, Dict] = {}
  url2count: Dict[str, int] = defaultdict(lambda: 0)
  cum_did = 0
  split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
  os.makedirs(fid_dir, exist_ok=True)
  for split, nsplit in [('train', 'train'), ('dev', 'dev')]:
    with open(os.path.join(msmarcoqa_dir, f'{split}_v2.1.jsonl'), 'r') as fin:
      fid_examples: List[Dict] = []
      for l in tqdm(fin):
        example = json.loads(l)
        if filter_func is not None and filter_func(example):  # skip examples satisfying certain conditions
          continue
        query = example['query']
        answers = example['answers']
        assert type(example['query_id']) is int
        query_id = str(example['query_id'])
        query_type = example['query_type']
        qid2dict[query_id] = {'_id': query_id, 'text': query, 'metadata': {'answer': answers, 'type': query_type}}
        ctxs: List[Dict] = []
        for passage in example['passages']:
          url = passage['url'].strip()
          text = passage['passage_text'].strip()
          select = passage['is_selected']
          url2count[url] += 1
          while str(cum_did) in qid2dict or str(cum_did) in did2dict:
            cum_did += 1
          did = str(cum_did)
          did2dict[did] = {'_id': did, 'title': '', 'text': text, 'select': select, 'url': url}
          ctxs.append({'id': did, 'title': '', 'text': text, 'select': select, 'url': url})
          if select:
            split2qiddid[nsplit].append((query_id, did))  # TODO: this annotation is not complete
        fid_examples.append({
          'question': query,
          'id': query_id,
          'answers': answers,
          'ctxs': ctxs
        })
    with open(os.path.join(fid_dir, f'{nsplit}.json'), 'w') as fout:
      json.dump(fid_examples, fout, indent=2)
    with open(os.path.join(fid_dir, f'{nsplit}.1000.json'), 'w') as fout:
      fid_examples = [fid_examples[i] for i in np.random.choice(len(fid_examples), min(1000, len(fid_examples)), replace=False)]
      json.dump(fid_examples, fout, indent=2)

  with open(os.path.join(fid_dir, 'psgs.tsv'), 'w') as fout, \
       open(os.path.join(fid_dir, 'line2docid.tsv'), 'w') as l2dfout:
    fout.write('id\ttext\ttitle\n')
    for lid, (did, doc) in enumerate(did2dict.items()):
      title = clean_text_for_tsv(doc['title'])
      text = clean_text_for_tsv(doc['text'])
      fout.write(f'{did}\t{text}\t{title}\n')
      l2dfout.write(f'{lid}\t{did}\n')

  save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)
  print(f'#quries {len(qid2dict)}, #docs {len(did2dict)}')


def convert_bioasq_to_beir_format(bioasq_dir: str, beir_dir: str, sub_sample: int = None):
  def extract_answer(question: Dict) -> Union[List[List[str]], str]:
    if question['type'] == 'factoid':
      ans = question['exact_answer']
      if type(ans) is list and type(ans[0]) is str:  # multi alias of a single entity
        ans = [ans]
      elif type(ans) is list and type(ans[0]) is list and type(ans[0][0]) is str:  # multi alias of multi entities
        pass
      else:
        raise ValueError
      return ans
    elif question['type'] == 'list':
      ans = question['exact_answer']
      assert type(ans) is list and type(ans[0]) is list and type(ans[0][0]) is str
      return ans
    elif question['type'] == 'yesno':
      ans = question['exact_answer'].lower()
      assert type(ans) is str and ans in {'yes', 'no'}
      return ans
    elif question['type'] == 'summary':
      ans = question['ideal_answer']
      assert 'exact_answer' not in question
      assert type(ans) is list and type(ans[0]) is str, ans
      return [ans]
    else:
      raise NotImplementedError

  # load all docs
  did2dict: Dict[str, Dict] = {}
  seen_dids: Set[str] = set()
  doc_duplicates = 0
  with open(os.path.join(bioasq_dir, 'allMeSH_2020.json'), 'rb') as fin:
    _ = fin.readline()  # skip the first line which has no article
    for line_id, l in tqdm(enumerate(fin)):
      l = l.decode('ISO-8859-1')
      l = l.rstrip('\n')
      if l.endswith('},'):
        l = l.rstrip(',')
      elif l.endswith('}]}'):
        l = l[:-2]
      else:
        raise ValueError
      l = json.loads(l)
      title = l['title']
      abs = l['abstractText']
      did = l['pmid']
      if did in seen_dids:
        print(f'duplicate doc id {did}')
        doc_duplicates += 1
        continue
      seen_dids.add(did)
      did2dict[did] = {'_id': did, 'title': title, 'text': abs}
  with open(os.path.join(bioasq_dir, 'BioASQ-Task8b-Manual-Fixes.tsv'), 'r') as fin:
    for l in fin:
      did, title, abs = l.rstrip('\n').split('\t')
      if did in seen_dids:
        print(f'duplicate doc id {did}')
        doc_duplicates += 1
        continue
      seen_dids.add(did)
      did2dict[did] = {'_id': did, 'title': title, 'text': abs}
  all_dids = set(did2dict.keys())
  print(f'#docs {len(all_dids)}, with {doc_duplicates} duplicates already removed')

  qid2dict: Dict[str, Dict] = {}
  qid = 0
  split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
  split2stat: Dict[str, Dict] = defaultdict(lambda: {
    '#query': 0, '#rel-doc-per-query': [], 'type2count': defaultdict(lambda: 0)})
  all_rel_dids: Set[str] = set()
  covered_rel_dids: Set[str] = set()
  seen_query_ids: Set[str] = set()
  seen_queries: Set[str] = set()
  query_duplicates = query_without_rel = 0
  for files, nsplit in [(['training8b.json'], 'train'),
                        ([f'Task8BGoldenEnriched/8B{i + 1}_golden.json' for i in range(5)], 'test')]:
    for file in files:
      file = os.path.join(bioasq_dir, file)
      with open(file, 'r') as fin:
        questions = json.load(fin)['questions']
        for question in questions:
          raw_id = question['id']
          assert '_' not in raw_id
          assert raw_id not in seen_query_ids
          seen_query_ids.add(raw_id)
          ans_type = question['type']
          assert ans_type in {'factoid', 'list', 'summary', 'yesno'}
          query = question['body']
          if query.lower() in seen_queries:
            print(f'duplicate queries "{query}" in file: {file}')
            query_duplicates += 1
          else:
            seen_queries.add(query.lower())
          rel_docs: List[str] = []
          for d in question['documents']:
            did = d.rsplit('/', 1)[1]
            all_rel_dids.add(did)
            if did not in did2dict:
              continue
            rel_docs.append(did)
            covered_rel_dids.add(did)
          if len(rel_docs) <= 0:
            query_without_rel += 1
            continue
          answers = extract_answer(question)
          while str(qid) in did2dict or str(qid) in qid2dict:
            qid += 1
          qid2dict[str(qid)] = {'_id': str(qid), 'text': query, 'metadata':
            {'raw_id': raw_id, 'type': ans_type, 'answer': answers}}
          split2stat[nsplit]['#query'] += 1
          split2stat[nsplit]['type2count'][ans_type] += 1
          for did in rel_docs:
            split2qiddid[nsplit].append((str(qid), did))
          split2stat[nsplit]['#rel-doc-per-query'].append(len(rel_docs))
  print(f'#duplcate queris {query_duplicates} #queries without relevant docs {query_without_rel}')
  print(f'#uncovered relevant docs {len(all_rel_dids - all_dids)}')
  for split, stat in split2stat.items():
    print(split, {k: np.mean(v) if type(v) is list else v for k, v in stat.items()})

  if sub_sample:
    assert sub_sample >= len(covered_rel_dids)
    all_unrel_dids = all_dids - covered_rel_dids
    num_keep_unrel_dids = min(sub_sample - len(covered_rel_dids), len(all_unrel_dids))
    subset_unrel_dids = set(np.random.choice(list(all_unrel_dids), num_keep_unrel_dids, replace=False))
    remain_dids = subset_unrel_dids | covered_rel_dids
    print(f'final #docs {len(remain_dids)}')
    did2dict = {k: v for k, v in did2dict.items() if k in remain_dids}

  # save
  save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)


def eval_retrieval(data: List[Dict], beir_dir: str, split: str = 'test', topks: List[int] = [1, 3, 5, 10, 100]):
  use_qid = False
  qid2dids: Dict[str, List[str]] = defaultdict(list)
  for example in data:
    qid = example['id'] if 'id' in example else example['question']
    use_qid = True if 'id' in example else use_qid
    for d in example['ctxs']:
      qid2dids[qid].append(d['id'])
  corpus, queries, qrels = GenericDataLoader(data_folder=beir_dir).load(split=split)
  qid2dict: Dict[str, Dict] = {}
  with open(os.path.join(beir_dir, 'queries.jsonl'), 'r') as fin:
    for l in fin:
      l = json.loads(l)
      qid2dict[l['_id']] = l
  qid2dids_gold: Dict[str, List[str]] = defaultdict(list)
  qid2type: Dict[str, str] = {}
  for qid in qrels:
    qid = qid if use_qid else queries[qid]
    qid2dids_gold[qid].extend([did for did in qrels[qid]])
    qid2type[qid] = qid2dict[qid]['metadata']['type']
  topk2has = defaultdict(list)
  type2topk2has = defaultdict(lambda: defaultdict(list))
  for qid in qid2dids:
    for topk in topks:
      preds = set(qid2dids[qid][:topk])
      gold = set(qid2dids_gold[qid])
      has = len(preds & gold) > 0
      topk2has[topk].append(has)
      type2topk2has[qid2type[qid]][topk].append(has)

  print(f'use qid {use_qid}')
  for topk in topks:
    print(topk, np.mean(topk2has[topk]), sep='\t')
  for qtype in type2topk2has:
    print(qtype)
    for topk in topks:
      print(topk, np.mean(type2topk2has[qtype][topk]), sep='\t')


def eval_answer(ret_file: str,
                beir_dir: str = None,
                split: str = 'test',
                sort: bool = False,
                shuffle: bool = False,
                topk: int = 100,
                key_func: Callable = lambda x: x['score']):
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
      if sort:
        example['ctxs'] = sorted(example['ctxs'], key=lambda x: float(key_func(x)), reverse=True)
      if shuffle:
        random.shuffle(example['ctxs'])
      example['ctxs'] = example['ctxs'][:topk]
    if beir_dir is not None:
      eval_retrieval(data, beir_dir, split=split)
    else:
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


def convert_fid_to_rag_format(fid_dir: str,
                              rag_dir: str,
                              splits: List[str] = ['train', 'dev', 'test'],
                              entity_sep: str = '\t\t',
                              alias_sep: str = '\t',
                              with_context: str = None,
                              beir_dir: str = None):
  assert with_context in {None, 'all_relevant'}
  num_entities: List[int] = []
  num_alias: List[int] = []
  os.makedirs(rag_dir, exist_ok=True)
  for split in splits:
    if beir_dir:
      corpus, queries, qrels = GenericDataLoader(data_folder=beir_dir).load(split=split)
    with open(os.path.join(fid_dir, f'{split}.json'), 'r') as fin, \
         open(os.path.join(rag_dir, f'{split}.id'), 'w') as ifout, \
         open(os.path.join(rag_dir, f'{split}.source'), 'w') as sfout, \
         open(os.path.join(rag_dir, f'{split}.target'), 'w') as tfout:
      data = json.load(fin)
      for example in data:
        qid = example['id']
        question = clean_text_for_tsv(example['question'])
        answers = example['answers']
        num_entities.append(len(answers))
        for e in answers:
          num_alias.append(len(e))
        answers = entity_sep.join([alias_sep.join([clean_text_for_tsv(a) for a in e]) for e in answers])
        if with_context is None:  # only question
          ifout.write(f'{qid}\n')
          sfout.write(f'{question}\n')
          tfout.write(f'{answers}\n')
        elif with_context == 'all_relevant':  # add context of all relevant docs
          for did in qrels[qid]:
            title = clean_text_for_tsv(corpus[did].get('title'))
            text = clean_text_for_tsv(corpus[did].get('text'))
            ifout.write(f'{qid}\n')
            sfout.write(f'{question}\t{title}\t{text}\n')
            tfout.write(f'{answers}\n')

  print(f'avg #entities per answer {np.mean(num_entities)}, avg #alias per entity {np.mean(num_alias)}')


def filter_beir_query(beir_dir: str, out_dir: str, filter_func: Callable, splits: List[str] = ['train', 'dev', 'test']):
  os.makedirs(out_dir, exist_ok=True)
  kept_ids: Set[str] = set()
  with open(os.path.join(beir_dir, 'queries.jsonl'), 'r') as fin, \
       open(os.path.join(out_dir, 'queries.jsonl'), 'w') as fout:
    for l in fin:
      id = json.loads(l)['_id']
      if filter_func(json.loads(l)):
        fout.write(l)
        kept_ids.add(id)
  os.makedirs(os.path.join(out_dir, 'qrels'), exist_ok=True)
  for split in splits:
    with open(os.path.join(beir_dir, 'qrels', f'{split}.tsv'), 'r') as fin, \
         open(os.path.join(out_dir, 'qrels', f'{split}.tsv'), 'w') as fout:
      fout.write(fin.readline())  # tsv head
      for l in fin:
        if l.strip().split('\t', 1)[0] in kept_ids:
          fout.write(l)


def eval_qa(pred_file: str, gold_file_beir: str, metric: str = 'src.evaluation.ems'):
  qid2pred: Dict[str, str] = {}
  with open(pred_file, 'r') as fin:
    for l in fin:
      qid, pred = l.rstrip('\n').split('\t')
      assert qid not in qid2pred, 'duplicate qid'
      qid2pred[qid] = pred
  qid2dict: Dict[str, Dict] = {}
  with open(gold_file_beir, 'r') as fin:
    for l in fin:
      l = json.loads(l)
      qid2dict[l['_id']] = l
  scores: List[Any] = []
  type2scores: Dict[str, List[Any]] = defaultdict(list)
  metric_func: Callable = eval(metric)
  for qid, pred in qid2pred.items():
    gold = qid2dict[qid]['metadata']['answer']
    qtype = qid2dict[qid]['metadata']['type'] if 'type' in qid2dict[qid]['metadata'] else None
    score = metric_func(pred, gold)
    scores.append(score)
    type2scores[qtype].append(score)
  print(f'{np.mean(scores)}')
  for qtype, scores in type2scores.items():
    print(f'{qtype}\t{np.mean(scores)}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='preprocessing')
  parser.add_argument('--task', type=str, choices=[
    'convert_sciq_to_beir_format', 'convert_techqa_to_beir_format', 'convert_quasar_to_beir_format',
    'convert_msmarcoqa_to_beir_fid_format', 'eval_qa',
    'convert_bioasq_to_beir_format', 'filter_beir_query', 'convert_fid_to_rag_format',
    'aggregate_ctxs', 'eval_variance', 'convert_beir_to_fid_format', 'eval_answer', 'create_whole_test'])
  parser.add_argument('--inp', type=str, help='input file', nargs='+')
  parser.add_argument('--out', type=str, help='output file', nargs='+')
  parser.add_argument('--other', type=str, nargs='+', help='additional arguments')
  args = parser.parse_args()

  if args.task == 'aggregate_ctxs':
    json_files: List[str] = args.inp
    out_file = args.out[0]
    aggregate_ctxs(json_files, out_file)

  elif args.task == 'convert_beir_to_fid_format':
    beir_dir = args.inp[0]
    out_dir = args.out[0]
    convert_beir_to_fid_format(beir_dir, out_dir, dataset_name='msmarcoqa', splits=['dev', 'train'])

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

  elif args.task == 'convert_msmarcoqa_to_beir_fid_format':
    msmarcoqa_dir = args.inp[0]
    beir_dir, fid_dir = args.out
    def has_ans_no_desc_with10_50char_filter_func(example: Dict):
      if len(example['answers']) == 0 and example['answers'][0].strip() == 'No Answer Present.':
        return True
      if example['query_type'] == 'DESCRIPTION':
        return True
      if len(example['passages']) != 10:
        return True
      if np.sum([passage['is_selected'] for passage in example['passages']]) == 0:
        return True
      if len(example['answers'][0]) > 50:
        return True
      return False
    convert_msmarcoqa_to_beir_fid_format(
      msmarcoqa_dir, beir_dir, fid_dir, filter_func=has_ans_no_desc_with10_50char_filter_func)

  elif args.task == 'convert_bioasq_to_beir_format':
    bioasq_dir = args.inp[0]
    beir_dir = args.out[0]
    convert_bioasq_to_beir_format(bioasq_dir, beir_dir, sub_sample=500000)

  elif args.task == 'convert_fid_to_rag_format':
    fid_dir, beir_dir = args.inp
    rag_dir = args.out[0]
    convert_fid_to_rag_format(fid_dir, rag_dir, splits=['train', 'test'],
                              with_context='all_relevant', beir_dir=beir_dir)

  elif args.task == 'filter_beir_query':
    # TODO: ln corpus manually
    beir_dir = args.inp[0]
    out_dir = args.out[0]
    bioasq_remove_summary = lambda example: example['metadata']['type'] != 'summary'
    filter_beir_query(beir_dir, out_dir, filter_func=bioasq_remove_summary, splits=['train', 'test'])

  elif args.task == 'eval_answer':
    key, method = args.other[:2]
    index = 3
    if len(args.other) > 2:
      index = int(args.other[2])
    #key = 'score'  # score two_tower_attn_score encoder_score
    #method = 'avg'
    n_two_tower_layers = 6
    num_heads = 12

    ret_file = args.inp[0]
    beir_dir = args.inp[1] if len(args.inp) > 1 else None
    split = args.inp[2] if len(args.inp) > 2 else None

    sort = False
    shuffle = False
    key_func = None
    if method == 'default':
      pass
    elif method == 'shuffle':
      shuffle = True
    elif method == 'raw':
      sort = True
      key_func = lambda x: x[key]
    elif method == 'avg':
      eval_answer(ret_file, beir_dir=beir_dir, split=split, sort=True, key_func=lambda x: np.mean(x[key]))
      eval_answer(ret_file, beir_dir=beir_dir, split=split, sort=True, key_func=lambda x: np.mean(x[key][-1]))
      exit()
    elif method == 'avg_head':
      for i in range(num_heads):
        print(f'head {i}')
        eval_answer(ret_file, beir_dir=beir_dir, split=split, sort=True, key_func=lambda x: np.mean(x[key][11][i]))
      exit()
    elif method == 'specific':
      sort = True
      key_func = lambda x: np.mean(x[key][-1][index])
    elif method == 'flat':
      sort = True
      key_func = lambda x: np.mean([b for a in x[key] for b in a][index])
    else:
      for l in range(12 - n_two_tower_layers):
        if method == 'avg_layer':
          eval_answer(ret_file, beir_dir=beir_dir, split=split, sort=True, key_func=lambda x: np.mean(x[key][l]))
          continue
        for i in range(num_heads):
          print(f'layer {l}, head {i}')
          eval_answer(ret_file, beir_dir=beir_dir, split=split, sort=True, key_func=lambda x: np.mean(x[key][l][i]))
      exit()
    eval_answer(ret_file, beir_dir=beir_dir, split=split, sort=sort, key_func=key_func)

  elif args.task == 'eval_qa':
    pred_file, gold_file_beir = args.inp
    eval_qa(pred_file, gold_file_beir)

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
