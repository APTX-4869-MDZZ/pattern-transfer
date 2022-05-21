import sys
import os
import time
import random
import json
import shutil
sys.path.append('../multi-engine')
from imageEngine import ImageSearchEngine
from MyMultiProcessing import MyMultiProcessing

image_dir = 'new_image_patterns/'
cache_image_dirs = ['bootea_images/', 'blink_images/']
imageEngine = ImageSearchEngine(15, ['google'], image_dir, True)
def dowloadimage_task(search_query, dir_name):
  if os.path.exists(image_dir + dir_name):
    return
  
  imageEngine.get_topk_from_engine(search_query, dir_name)
  if (not random.randint(0, 59)):
    print('sleeping...')
    time.sleep(random.randint(0, 3) * 5)
    print('continue...')

if __name__=='__main__':
  patterns = dict()
  with open('new_patterns.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      type_name = line.split('\t\t')[0]
      entity_id = type_name.split(' ')[0]
      temp_patterns = line[:-1].split('\t\t')[1:]
      patterns[entity_id] = temp_patterns

  # existed_entities = dict()
  # for index, cache_image_dir in enumerate(cache_image_dirs):
  #   for existed_dir in os.listdir(cache_image_dir):
  #     existed_entities[existed_dir.split('_')[0]]=index
  search_querys = []
  dir_names = []

  # entity2pagename = dict()
  # missing_entity = []
  # with open('2M_entity2pagename.txt', 'r', encoding='utf-8') as f:
  #   for line in f.readlines():
  #     entity_id, name = line.split('\t')[:2]
  #     entity2pagename[entity_id] = name
  # with open('tail_type2entity.txt', 'r', encoding='utf-8') as f:
  #   for line in f.readlines():
  #     type_id, entity_str = line[:-1].split(' ')
  #     for entity_id in entity_str.split(','):
  #       if not entity_id in entity2pagename:
  #         missing_entity.append(entity_id)
  #         continue
  #       name = entity2pagename[entity_id]
  #       for pattern in patterns[type_id]:
  #         search_querys.append('"{}" "{}"'.format(name, pattern))
  #         dir_names.append(entity_id + '_' + pattern)
  # with open('missing_entity.txt', 'w', encoding='utf-8') as f:
  #   f.write('\n'.join(missing_entity))

  # ENT_ID = 'entity_id'
  # with open('fb15k/entity_fb15k.jsonl', 'r', encoding='utf-8') as f:
  #   for line in f.readlines():
  #     entity = json.loads(line[:-1])
  #     # for type_ in entity['types']:
  #     #   if type_ in patterns:
  #     type_ = entity['type']
  #     if entity[ENT_ID] in existed_entities:
  #       for pattern in patterns[type_]:
  #         if not os.path.exists(image_dir + entity[ENT_ID] + '_' + pattern):
  #           shutil.copytree(cache_image_dirs[existed_entities[entity[ENT_ID]]] + entity[ENT_ID] + '_' + pattern, image_dir + entity[ENT_ID] + '_' + pattern)
  #     else:
  #       for pattern in patterns[type_]:
  #         search_querys.append('"{}" "{}"'.format(entity['pagename'], pattern))
  #         dir_names.append(entity[ENT_ID] + '_' + pattern)
  
  for i in range(25):
    with open('type2entity/{}.jsonl'.format(i), 'r') as f:
      for line in f.readlines()[4500:5000]:
        data = json.loads(line[:-1])
        type_ = data['types']
        for pattern in patterns[type_]:
          search_querys.append('"{}" "{}"'.format(data['pagename'], pattern))
          dir_names.append(data['entity_id'] + '_' + pattern)

  multiprocessing = MyMultiProcessing(
    4, dowloadimage_task, list(zip(search_querys, dir_names)),
    total=len(search_querys), ordered=False
  )
  multiprocessing.start()