def replace_by_prev_data():
  prev_head_entities = dict()
  with open('../filtered_type2entity.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      type_id, entity_list = line[:-1].split(' ')
      prev_head_entities[type_id] = ','.join(entity_list.split(',')[:100])

  head_type2entity = open('head_type2entity.txt', 'w', encoding='utf-8')
  with open('head_type2entity.txt.back', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      type_id, entity_list = line[:-1].split(' ')
      if type_id in prev_head_entities:
        entity_list = prev_head_entities[type_id]
      head_type2entity.write(type_id + ' ' + entity_list + '\n')

def add_prev_pagename():
  entity2pagename = dict()
  with open('entity2pagename.txt.back', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      entity_id, name = line[:-1].split('\t')
      entity2pagename[entity_id] = name
  with open('../filtered_entity2pagename.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      entity_id, name, _ = line[:-1].split('\t')
      entity2pagename[entity_id] = name

  entity2pagename_file = open('entity2pagename.txt', 'w', encoding='utf-8')
  with open('head_type2entity.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      _, entity_str = line[:-1].split(' ')
      for entity_id in entity_str.split(','):
        entity2pagename_file.write(entity_id + '\t' + entity2pagename[entity_id] + '\n')

def prepare_tail_entities():
  type_ids = []
  with open('new_patterns.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      type_name = line.split('\t\t')[0]
      entity_id = type_name.split(' ')[0]
      type_ids.append(entity_id)

  tail_file = open('tail_type2entity.txt', 'w', encoding='utf-8')
  with open('../filtered_type2entity.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      type_id, entity_str = line[:-1].split(' ')
      if type_id in type_ids:
        tail_file.write(type_id + ' ' + ','.join(entity_str.split(',')[100:]) + '\n')
