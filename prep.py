from typing import List, Tuple, Dict, Callable, Union, Set, Any
import argparse
import json
import os
import re
import numpy as np
import time
import random
import logging
import pickle
import statistics
from tqdm import tqdm
import spacy
from spacy.lang.en import English
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
    self.qid2answer, self.qid2meta = self.load_query(os.path.join(root_dir, 'queries.jsonl'))

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

  @classmethod
  def get_answer_fiqa(cls, metadata: Dict) -> List[str]:
    return ['']  # TODO: use the raw long answer?

  @classmethod
  def get_answer_cqadupstack(cls, metadata: Dict) -> List[str]:
    return ['']  # TODO: use what for answer?

  @classmethod
  def get_answer_pseudo(cls, metadata: Dict) -> List[str]:
    return [metadata['answer']]

  def load_query(self, filename: str):
    qid2meta: Dict[str, Dict] = {}
    qid2answer: Dict[str, Any] = {}
    with open(filename, 'r') as fin:
      for l in fin:
        l = json.loads(l)
        id, text, metadata = l['_id'], l['text'], l['metadata']
        qid2meta[id] = metadata
        ans = getattr(self, f'get_answer_{self.name}')(metadata)
        if ans is None:
          continue
        qid2answer[id] = ans
    return qid2answer, qid2meta


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


def convert_beir_to_fid_format(
     beir_dir: str,
     out_dir: str,
     dataset_name: str,
     splits: List[str],
     topk: int = 100,
     add_self: bool = False):
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
      if add_self:
        if bert_data.qid2meta[qid]['docid'] not in set(x[0] for x in scores):  # self doc not retrieved
          scores.insert(0, (bert_data.qid2meta[qid]['docid'], scores[0][1] + 1.0))  # highest score
          scores = scores[:topk]
      for rank in range(len(scores)):
        did = scores[rank][0]
        title = clean_text_for_tsv(corpus[did].get('title'))
        if add_self and did == bert_data.qid2meta[qid]['docid']:
          text = clean_text_for_tsv(bert_data.qid2meta[qid]['context'])
        else:
          text = clean_text_for_tsv(corpus[did].get('text'))
        example['ctxs'].append({'id': did, 'title': title, 'text': text})
      examples.append(example)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(examples, open(os.path.join(out_dir, f'{split}.json'), 'w'), indent=2)


def save_beir_format(beir_dir: str,
                     qid2dict: Dict[str, Dict] = None,
                     did2dict: Dict[str, Dict] = None,
                     split2qiddid: Dict[str, List[Tuple[str, str]]] = None):
  # save
  os.makedirs(beir_dir, exist_ok=True)
  if qid2dict is not None:
    with open(os.path.join(beir_dir, 'queries.jsonl'), 'w') as fout:
      for qid in qid2dict:
        fout.write(json.dumps(qid2dict[qid]) + '\n')
  if did2dict is not None:
    with open(os.path.join(beir_dir, 'corpus.jsonl'), 'w') as fout:
      for did in did2dict:
        fout.write(json.dumps(did2dict[did]) + '\n')
  if split2qiddid is not None:
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


def load_query2dids_gold(bioasq_raw_dir: str) -> Dict[str, List[str]]:
  query2dids_gold: Dict[str, List[str]] = defaultdict(list)
  for root, _, files in os.walk(bioasq_raw_dir):
    for file in files:
      with open(os.path.join(root, file), 'r') as fin:
        data = json.load(fin)
        for q in data['questions']:
          query2dids_gold[q['body']] = [d.rsplit('/', 1)[1] for d in q['documents']]
  return query2dids_gold


def eval_retrieval(
     data: List[Dict],
     beir_dir: str,
     split: str = 'test',
     topks: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100],
     use_raw_bioasq: bool = False):
  use_qid = False
  qid2dids: Dict[str, List[str]] = defaultdict(list)
  for example in data:
    qid = example['id'] if ('id' in example and not use_raw_bioasq) else example['question']
    use_qid = True if ('id' in example and not use_raw_bioasq) else use_qid
    for d in example['ctxs']:
      qid2dids[qid].append(d['id'])
  if use_raw_bioasq:
    qid2dids_gold: Dict[str, List[str]] = load_query2dids_gold(beir_dir)
    qid2type: Dict[str, str] = defaultdict(lambda: None)
  else:
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
      qid2type[qid] = qid2dict[qid]['metadata']['type'] if 'type' in qid2dict[qid]['metadata'] else None
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
    print(np.mean(topk2has[topk]), sep='\t', end='\t')
  print()
  for qtype in type2topk2has:
    print(qtype)
    for topk in topks:
      print(np.mean(type2topk2has[qtype][topk]), sep='\t', end='\t')
    print()


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


def aggregate_ctx(query_files: List[str], tsv_file: str, topk: int = None):
  seen_ids: Set[str] = set()
  with open(tsv_file, 'w') as fout:
    fout.write('id\ttext\ttitle\n')
    for query_file in query_files:
      with open(query_file, 'r') as fin:
        data = json.load(fin)
        for q in data:
          ctxs = q['ctxs']
          if topk:
            ctxs = ctxs[:topk]
          for ctx in ctxs:
            if ctx['id'] not in seen_ids:
              fout.write(f"{ctx['id']}\t{clean_text_for_tsv(ctx['text'])}\t{clean_text_for_tsv(ctx['title'])}\n")
            seen_ids.add(ctx['id'])
  print(f'total #ctx {len(seen_ids)}')


def rank2json(rank_file: str, query_json_file: str, psge_tsv_file: str, out_json_file: str):
  with open(rank_file, 'rb') as fin:
    qid2rank = pickle.load(fin)
  docs = src.util.load_passages(psge_tsv_file)
  did2doc: Dict[str, Tuple] = {doc[0]: doc for doc in docs}
  with open(query_json_file, 'r') as fin, open(out_json_file, 'w') as fout:
    queries = json.load(fin)
    queries_with_newctx: List[Dict] = []
    for i, query in enumerate(queries):
      qid = query['id'] if 'id' in query else str(i)
      rank = qid2rank[qid]
      query['ctxs'] = [{
        'id': did,
        'text': did2doc[did][1],
        'title': did2doc[did][2],
      } for did, score in rank]
      queries_with_newctx.append(query)
    json.dump(queries_with_newctx, fout, indent=2)


def add_negative(query_file: str, out_file: str, raw_count: int = 100, add_count: int = 100):
  did2dict: Dict[str, Dict] = {}
  with open(query_file, 'r') as fin, open(out_file, 'w') as fout:
    data = json.load(fin)
    for q in tqdm(data):
      for ctx in q['ctxs']:
        did2dict[ctx['id']] = ctx
    alldids: List[str] = list(set(did2dict.keys()))
    for q in tqdm(data):
      returned = [ctx['id'] for ctx in q['ctxs'][:raw_count]]
      sampled_dids = random.sample(range(len(alldids)), raw_count + add_count)
      sampled_dids = set(alldids[i] for i in sampled_dids)
      sampled_dids = list(sampled_dids - set(returned))[:add_count]
      assert len(returned) == raw_count
      assert len(sampled_dids) == add_count, f'{len(returned)} {len(sampled_dids)}'
      q['ctxs'] = []
      for did in returned + sampled_dids:
        q['ctxs'].append(did2dict[did])
    json.dump(data, fout, indent=2)


def add_negative_mimic_inbatch(query_file: str, out_file: str, batch_size: int = 1):
  qidx2ctxs: Dict[str, List] = {}
  with open(query_file, 'r') as fin, open(out_file, 'w') as fout:
    data = json.load(fin)
    for qidx, q in tqdm(enumerate(data)):
      qidx2ctxs[qidx] = q['ctxs']
    for qidx, q in tqdm(enumerate(data)):
      sampled_qidxs = random.sample(range(len(qidx2ctxs)), batch_size)
      sampled_qidxs = list(set(sampled_qidxs) - {qidx})[:batch_size - 1]
      assert len(sampled_qidxs) == batch_size - 1
      q['ctxs'] = []
      for _qidx in ([qidx] + sampled_qidxs):
        q['ctxs'].extend(qidx2ctxs[_qidx])
      assert len(q['ctxs']) == 400
    json.dump(data, fout, indent=2)


def create_pseudo_queries_from_corpus(
     beir_dir: str,
     out_beir_dir: str,
     split: str,
     subsample: int = None,
     num_sent_per_doc: int = 1,
     max_num_mask_ent: int = 5):
  sentencizer = English()
  sentencizer.add_pipe('sentencizer')
  ner = spacy.load('en_core_web_sm')

  corpus, _, _ = GenericDataLoader(data_folder=beir_dir).load(split=split)
  overall_qid = 0
  qid2dict: Dict[str, Dict] = {}
  split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
  num_ents: List[int] = []
  if subsample:
    sampled_dids = random.sample(list(corpus.keys()), min(len(corpus), subsample))
  else:
    sampled_dids = corpus.keys()
  for did in tqdm(sampled_dids):
    text = clean_text_for_tsv(corpus[did].get('text'))  # only use text not title
    sents = list(sentencizer(text).sents)
    if len(sents) <= 1:
      continue
    sent_idxs = random.sample(range(len(sents)), min(len(sents), num_sent_per_doc))
    for sent_idx in sent_idxs:
      query = str(sents[sent_idx])
      context = ' '.join(map(str, sents[:sent_idx] + sents[sent_idx + 1:]))
      # ner
      ents: List[Tuple[int, int, str]] = [(ent.start_char, ent.end_char, ent.label_) for ent in ner(query).ents]
      # remove overlap
      ents = sorted(ents, key=lambda x: (x[0], x[1]))
      ents = [ent for i, ent in enumerate(ents) if (i <= 0 or ent[0] >= ents[i - 1][1])]
      if len(ents) <= 0:
        continue
      # max num
      if len(ents) > max_num_mask_ent:
        ents = [ents[i] for i in sorted(random.sample(range(len(ents)), max_num_mask_ent))]
      num_ents.append(len(ents))
      # mask
      masked_query: List[str] = []
      target: List[str] = []
      prev_idx = 0
      mask_idx = 0
      for i, ent in enumerate(ents):
        masked_query.append(query[prev_idx:ent[0]])
        masked_query.append(f'<extra_id_{mask_idx}>')
        target.append(f'<extra_id_{mask_idx}> {query[ent[0]:ent[1]]}')
        prev_idx = ent[1]
        mask_idx += 1
      masked_query.append(query[prev_idx:])
      masked_query: str = ''.join(masked_query)
      target.append(f'<extra_id_{mask_idx}>')
      target: str = ' '.join(target)

      qid = f'{overall_qid}'
      qid2dict[qid] = {
        '_id': qid,
        'text': masked_query,
        'metadata': {'answer': target, 'docid': did, 'context': context}
      }
      split2qiddid[split].append((qid, did))
      overall_qid += 1
  save_beir_format(out_beir_dir, qid2dict, None, split2qiddid)
  print(f'#queries {len(qid2dict)}, avg #entities {np.mean(num_ents)}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='preprocessing')
  parser.add_argument('--task', type=str, choices=[
    'convert_sciq_to_beir_format', 'convert_techqa_to_beir_format', 'convert_quasar_to_beir_format',
    'convert_msmarcoqa_to_beir_fid_format', 'eval_qa', 'aggregate_ctx', 'rank2json',
    'add_negative', 'add_negative_mimic_inbatch', 'create_pseudo_queries_from_beir',
    'convert_bioasq_to_beir_format', 'filter_beir_query', 'convert_fid_to_rag_format', 'split_fid_file',
    'aggregate_ctxs', 'eval_variance', 'convert_beir_to_fid_format', 'eval_answer', 'create_whole_test'])
  parser.add_argument('--inp', type=str, help='input file', nargs='+')
  parser.add_argument('--out', type=str, help='output file', nargs='+')
  parser.add_argument('--other', type=str, nargs='+', help='additional arguments')
  args = parser.parse_args()

  seed = 2022
  random.seed(seed)
  np.random.seed(seed)

  if args.task == 'aggregate_ctxs':
    json_files: List[str] = args.inp
    out_file = args.out[0]
    aggregate_ctxs(json_files, out_file)

  elif args.task == 'convert_beir_to_fid_format':
    beir_dir = args.inp[0]
    out_dir = args.out[0]
    convert_beir_to_fid_format(beir_dir, out_dir, dataset_name='pseudo', splits=['test'], add_self=True)

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
    layer_index, head_index = -1, 3
    if len(args.other) > 2:
      layer_index, head_index = int(args.other[2]), int(args.other[3])
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
      key_func = lambda x: np.mean(x[key][layer_index][head_index])
    elif method == 'flat':
      sort = True
      key_func = lambda x: np.mean([b for a in x[key] for b in a][head_index])
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

  elif args.task == 'aggregate_ctx':
    query_files = args.inp
    tsv_file = args.out[0]
    aggregate_ctx(query_files, tsv_file, topk=None)

  elif args.task == 'rank2json':
    rank_file, query_json_file, psgs_tsv_file = args.inp
    out_json_file = args.out[0]
    rank2json(rank_file, query_json_file, psgs_tsv_file, out_json_file)

  elif args.task == 'add_negative':
    query_file = args.inp[0]
    out_file = args.out[0]
    add_negative(query_file, out_file, raw_count=50, add_count=50)

  elif args.task == 'add_negative_mimic_inbatch':
    query_file = args.inp[0]
    out_file = args.out[0]
    add_negative_mimic_inbatch(query_file, out_file, batch_size=4)

  elif args.task == 'create_pseudo_queries_from_beir':
    beir_dir = args.inp[0]
    out_beir_dir = args.out[0]
    create_pseudo_queries_from_corpus(beir_dir, out_beir_dir, subsample=100000, split='test', max_num_mask_ent=5)

  elif args.task == 'split_fid_file':
    query_file = args.inp[0]
    split1, split2 = args.out
    count1, count2 = 30000, 5000
    with open(query_file, 'r') as fin, open(split1, 'w') as fout1, open(split2, 'w') as fout2:
      data = json.load(fin)
      assert len(data) >= count1 + count2
      perm = np.random.permutation(len(data))
      data1 = [data[i] for i in perm[:count1]]
      data2 = [data[i] for i in perm[-count2:]]
      json.dump(data1, fout1, indent=2)
      json.dump(data2, fout2, indent=2)
