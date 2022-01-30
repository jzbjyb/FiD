from typing import List, Tuple, Dict
import argparse
import json
import os
import numpy as np
from collections import defaultdict
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25


class BEIRDataset:
  def __init__(self, root_dir: str):
    self.qid2answer: Dict[str, str] = self.load_query(os.path.join(root_dir, 'queries.jsonl'))

  @classmethod
  def get_label(cls, metadata: Dict):
    labels: List[str] = []
    for did, sents in metadata.items():
      for sent in sents:
        labels.append(sent['label'])
    if len(labels) == 0:
      return 'uncertain'
    return labels[0].lower()

  @classmethod
  def load_query(cls, filename: str):
    qid2answer: Dict[str, str] = {}
    with open(filename, 'r') as fin:
      for l in fin:
        l = json.loads(l)
        id, text, metadata = l['_id'], l['text'], l['metadata']
        label = cls.get_label(metadata)
        qid2answer[id] = label
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
    #print(qid, query, qid2dids_gold[query], corpus[qid2dids_gold[query][0]])
    #input()

  topk2has = defaultdict(list)
  for qid in qid2dids:
    for topk in topks:
      preds = set(qid2dids[qid][:topk])
      gold = set(qid2dids_gold[qid])
      has = len(preds & gold) > 0
      topk2has[topk].append(has)

  for topk in topks:
    print(topk, np.mean(topk2has[topk]))


def convert_beir_to_fid_format(beir_dir: str, out_dir: str, splits: List[str], topk: int = 100):
  bert_data = BEIRDataset(beir_dir)

  for split_ind, split in enumerate(splits):
    corpus, queries, qrels = GenericDataLoader(data_folder=beir_dir).load(split=split)

    # build index
    hostname = 'localhost'
    index_name = 'scifact'
    initialize = False  # TODO
    number_of_shards = 1  # TODO
    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)
    retriever = EvaluateRetrieval(model)

    # retrieve
    results = retriever.retrieve(corpus, queries)
    print(f'retriever evaluation for k in: {retriever.k_values}')
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)  # TODO
    print(ndcg, _map, recall, precision)

    # output
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='preprocessing')
  parser.add_argument('--task', type=str, choices=['aggregate_ctxs', 'convert_beir_to_fid_format', 'eval'])
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
    convert_beir_to_fid_format(beir_dir, out_dir, splits=['test', 'train'])

  elif args.task == 'eval':
    beir_dir, ret_file = args.inp
    eval(beir_dir, ret_file)
