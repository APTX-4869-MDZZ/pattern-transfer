import json
import random
from tqdm import tqdm
id2entity = dict()
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

with open('entity2id.txt', 'r', encoding='utf-8') as f:
  f.readline()
  for line in f.readlines():
    entity, id_ = line[:-1].split('\t')
    id2entity[id_] = entity

id2relation = dict()
with open('relation2id.txt', 'r', encoding='utf-8') as f:
  f.readline()
  for line in f.readlines():
    relation, id_ = line[:-1].split('\t')
    id2relation[id_] = relation

id2img = dict()
with open('id2img.txt', 'r', encoding='utf-8') as f:
  for line in f.readlines():
    id_, img = line[:-1].split('\t')
    id2img[id_] = img

with open('entity2wikidata.json', 'r', encoding='utf-8') as f:
  entity2wikidata = json.loads(f.read())

dataset_names = ['train', 'valid', 'test']
# dataset_names = ['train', 'valid', 'test']
datasets = []
for dataset_name in dataset_names:
  dataset = []
  with open(f'{dataset_name}2id.txt', 'r', encoding='utf-8') as f:
    f.readline()
    for line in tqdm(list(f.readlines())):
      s, o, p = line[:-1].split('\t')
      o_text = entity2wikidata[id2entity[o]]['label']
      o_word = tokenizer(o_text, add_special_tokens=False, return_tensors='pt')
      if o_word.input_ids.shape[1] > 1:
        continue
      if (o_word.input_ids == tokenizer.unk_token_id).sum().item() != 0:
        continue
      
      dataset.append({
        's': entity2wikidata[id2entity[s]]['label'],
        'p': p,
        'o': o_text,
        'y': 1,
        'img_file': id2img[s]
      })
      # if dataset_name in ['valid', 'train']:
      #   for _ in range(3):
      #     corrupted_o = random.randint(0, len(id2entity)-1)
      #     while str(corrupted_o) == o:
      #       corrupted_o = random.randint(0, len(id2entity)-1)
      #     dataset.append({
      #       's': entity2wikidata[id2entity[s]]['label'],
      #       'p': ' '.join(id2relation[p].split('/')[-1].split('_')),
      #       'o': entity2wikidata[id2entity[str(corrupted_o)]]['label'],
      #       'y': 0,
      #       'img_file': id2img[o]
      #     })
  datasets.append(dataset)
  with open(f'{dataset_name}_singleword.jsonl', 'w', encoding='utf-8') as f:
    for data in dataset:
      f.write(json.dumps(data) + '\n')