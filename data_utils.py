import os
import json
import torch
import numpy as np
from imageFeature_torch import ImageFeature
from sklearn.neighbors import LocalOutlierFactor

def read_patterns(pattern_file, without_name=False):
  patterns = dict()
  with open(pattern_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
      type_name = line.split('\t\t')[0]
      if without_name:
        type_name = type_name.split(' ')[0]
      temp_patterns = line[:-1].split('\t\t')[1:]
      patterns[type_name] = temp_patterns
  return patterns

def read_type2entity(type2entity_file):
  type2entity = dict()
  with open(type2entity_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
      type_id, entity_str = line[:-1].split(' ')
      type2entity[type_id] = entity_str.split(',')
  return type2entity

def pattern2wiki_image_dir(old_pattern_file, new_pattern_file):
  old_patterns = dict()
  with open(old_pattern_file, 'r', encoding='utf-8') as f:
    for segment in f.read().split('\n\n'):
      concept_line = segment.split('\n')[0]
      concept_id = concept_line.split(' ')[0]
      old_patterns[concept_id] = []
      for pattern_line in segment.split('\n')[1:]:
        pattern = pattern_line.split('\t\t')[0]
        old_patterns[concept_id].append(pattern)
  new_patterns = read_patterns(new_pattern_file)
  new_pattern_index = dict()
  for concept in new_patterns:
    new_pattern_index[concept] = []
    for pattern in new_patterns[concept]:
      new_pattern_index[concept].append(old_patterns[concept].index(pattern))
  return new_pattern_index

def getLofDetector(wikipedia_image_dir, patterns, device):
  # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
  imageProcessor = ImageFeature(device, 'VGG16')
  clfs = dict()
  for type_name in patterns.keys():
    type_ = type_name.split(' ')[0]
    clfs[type_] = []
    for i in range(len(patterns[type_name])):
      pattern_dir_path = wikipedia_image_dir+type_name+'/'+str(i)
      imageFeatures = []
      for image_file in os.listdir(pattern_dir_path):
        imageFeatures.append(imageProcessor.get_image_feature(pattern_dir_path+'/'+image_file))
      imageFeatures = np.array(imageFeatures).squeeze()
      clf = LocalOutlierFactor(n_neighbors=5, novelty=True).fit(imageFeatures)
      clfs[type_].append(clf)
  return clfs

def read_jsonl(file_name):
  data = []
  with open(file_name, 'r', encoding='utf-8') as f:
    for line in f.readlines():
      data.append(json.loads(line[:-1]))
  return data