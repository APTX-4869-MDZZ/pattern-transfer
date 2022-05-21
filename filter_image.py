from html import entities
from  scipy.spatial.distance import euclidean
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import math
import torch
import codecs
import os
import json

from tqdm import tqdm

from imageFeature_torch import ImageFeature
import data_utils

data_dir = 'image_patterns_data/'
wikipedia_image_dir = 'filtered_wikipedia_images/'
search_engine_dir = 'new_image_patterns/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

patterns = data_utils.read_patterns('selected_type/patterns.txt')
entity_list = data_utils.read_jsonl(data_dir + 'data8.jsonl')
clfs = data_utils.getLofDetector(wikipedia_image_dir, patterns, device)

imageProcessor = ImageFeature(device, 'VGG16')

patterns = data_utils.read_patterns('selected_type/patterns.txt', without_name=True)
pattern_types = list(patterns.keys())
filtered_image_index = dict()
for entity in tqdm(entity_list):
  type_ = (set(entity['types']) & set(pattern_types)).pop()
  # type_ = entity['type']
  entity_id = entity['entity_id']
  for index, pattern in enumerate(patterns[type_]):
    pattern_image_dir = search_engine_dir + entity_id + '_' + pattern
    if not os.path.exists(pattern_image_dir):
      continue
    imageFeatures = []
    if len(os.listdir(pattern_image_dir)) < 1:
      continue
    for image_file in os.listdir(pattern_image_dir):
      imageFeatures.append(imageProcessor.get_image_feature(pattern_image_dir+'/'+image_file))
    imageFeatures = np.array(imageFeatures).squeeze()
    if len(imageFeatures.shape) == 1:
      imageFeatures = imageFeatures.reshape(1, imageFeatures.shape[0])
    clf = clfs[type_][index]
    preds = clf.predict(imageFeatures)
    filtered_image_index[entity_id + '_' + pattern] = np.argwhere(preds != -1).tolist()

with open(data_dir + 'new_filtered_image_index8.json', 'w', encoding='utf-8') as f:
  f.write(json.dumps(filtered_image_index))