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
  clean_text = lambda x: '' if x is None else x.replace('\n', ' ').replace('\t', ' ')

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
          title = clean_text(corpus[did].get('title'))
          text = clean_text(corpus[did].get('text'))
          fout.write(f'{did}\t{text}\t{title}\n')
          l2dfout.write(f'{lid}\t{did}\n')

    examples: List[Dict] = []
    for qid, scores_dict in results.items():
      answer = bert_data.qid2answer[qid]
      query = clean_text(queries[qid])
      example = {'question': query, 'id': qid, 'answers': answer, 'ctxs': []}
      scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)[:topk]
      for rank in range(len(scores)):
        did = scores[rank][0]
        title = clean_text(corpus[did].get('title'))
        text = clean_text(corpus[did].get('text'))
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


def convert_fid_to_rag_format(fid_dir: str,
                              rag_dir: str,
                              splits: List[str] = ['train', 'dev', 'test'],
                              entity_sep: str = '\t\t',
                              alias_sep: str = '\t',
                              with_context: str = None,
                              beir_dir: str = None):
  assert with_context in {None, 'all_relevant'}
  clean_text = lambda x: x.replace('\t', ' ').replace('\n', ' ')
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
        question = clean_text(example['question'])
        answers = example['answers']
        num_entities.append(len(answers))
        for e in answers:
          num_alias.append(len(e))
        answers = entity_sep.join([alias_sep.join([clean_text(a) for a in e]) for e in answers])
        if with_context is None:  # only question
          ifout.write(f'{qid}\n')
          sfout.write(f'{question}\n')
          tfout.write(f'{answers}\n')
        elif with_context == 'all_relevant':  # add context of all relevant docs
          for did in qrels[qid]:
            title = clean_text(corpus[did].get('title'))
            text = clean_text(corpus[did].get('text'))
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='preprocessing')
  parser.add_argument('--task', type=str, choices=[
    'convert_sciq_to_beir_format', 'convert_techqa_to_beir_format', 'convert_quasar_to_beir_format',
    'convert_bioasq_to_beir_format', 'filter_beir_query', 'convert_fid_to_rag_format',
    'aggregate_ctxs', 'eval', 'eval_variance', 'convert_beir_to_fid_format', 'eval_answer', 'create_whole_test'])
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
    convert_beir_to_fid_format(beir_dir, out_dir, dataset_name='bioasq', splits=['test', 'train'])

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
    if method == 'default':
      eval_answer(ret_file, sort=False)
      exit()
    elif method == 'shuffle':
      eval_answer(ret_file, shuffle=True)
      exit()
    elif method == 'raw':
      eval_answer(ret_file, sort=True, key_func=lambda x: x[key])
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
      eval_answer(ret_file, sort=True, key_func=lambda x: np.mean(x[key][-1][index]))
      exit()
    elif method == 'flat':
      eval_answer(ret_file, sort=True, key_func=lambda x: np.mean([b for a in x[key] for b in a][index]))
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
