import matplotlib.pyplot as plt
import math
import json
import codecs
import os
from tqdm import tqdm
from MyMultiProcessing import MyMultiProcessing
from scp import SCPClient
import paramiko
from itertools import repeat

instanceof_id = 'P31'
# article_id = 'Q13442814'
# gene_id = 'Q7187'

#################
#####过滤NVC#####
#################
# filtered_type = ['Q13442814', 'Q8054', 'Q7187']
# filtered_entity = []
# with open('entity2type.txt', 'r', encoding='utf-8') as f:
#   for line in f.readlines():
#     entity, type_ = line[:-1].split(' ')
#     if type_ in filtered_type:
#       filtered_entity.append(entity)
# sorted_filtered_entity = codecs.open('sorted_top_filtered_entity.txt', 'w', 'utf-8')
# with open('sorted_top_entity.txt', 'r', encoding='utf-8') as f:
#   for entity_line in f.readlines():
#     entity, _ = entity_line[:-1].split(' ')
#     if entity in filtered_entity:
#       continue
#     sorted_filtered_entity.write(entity_line)


#################
###相关实体分布###
#################
# number_list = []
# count = 0
# top_entity = []
# with open('related_entity_count.txt', 'r', encoding='utf-8') as f:
#   for entity_line in f.readlines():
#     entity, number = entity_line[:-1].split(' ')
#     number = int(number)
#     if number != 0:
#       number_list.append(math.log(number, 10))
#       if number > 10:
#         top_entity.append(entity_line)
#     else:
#       count += 1
# print(count)
# plt.figure()
# plt.hist(number_list, color='y')
# plt.show()

# with open('top_entity.txt', 'w', encoding='utf-8') as f:
#   f.write(''.join(top_entity))


#################
###替换行政区域###
#################
# entity2type_replace_region = codecs.open('entity2type_replace_region.txt', 'w', 'utf-8')
# with open('entity2type.txt', 'r', encoding='utf-8') as f:
#   for line in f.readlines():
#     entity_id, type_id = line[:-1].split(' ')
#     with open('random_wiki_data/{}.json'.format(type_id), 'r', encoding='utf-8') as entity_file:
#       entity_json = json.loads(entity_file.read())
#       try:
#         name = entity_json['entities'][type_id]['labels']['en']['value']
#         if name.find('commune of') != -1 or name.find('municipality of') != -1:
#           type_id = 'region'
#         entity2type_replace_region.write(entity_id + ' ' + type_id + '\n')
#       except:
#         print(type_id)

#################
#头部实体类型统计#
#################
# entity2type = dict()
# with open('entity2type_replace_region.txt', 'r', encoding='utf-8') as f:
#   for line in f.readlines():
#     entity_id, type_id = line[:-1].split(' ')
#     entity2type[entity_id] = type_id
# types = dict()
# with open('related_entity_count.txt', 'r', encoding='utf-8') as f:
#   for entity_line in f.readlines():
#     entity_id, number = entity_line[:-1].split(' ')
#     try:
#       type_id = entity2type[entity_id]
#       number = int(number)
#       if number > 10:
#         types[type_id] = types.get(type_id, 0) + 1
#     except:
#       pass

# sorted_types = sorted(types.items(), key=lambda x:x[1], reverse=True)
# with open('sorted_top_types.txt', 'w', encoding='utf-8') as f:
#   for key, number in sorted_types:
#     f.write(key + ' ' + str(number) + '\n')



# types = []
# with open('entity2type_replace_region.txt', 'r', encoding='utf-8') as f:
#   for line in f.readlines():
#     key, number = line[:-1].split(' ')
#     types.append(key)
    
# types_count = dict()
# with open('related_entity_count.txt', 'r', encoding='utf-8') as f:
#   for entity_line in tqdm(f.readlines()):
#     entity_id = entity_line.split(' ')[0]
#     with open('random_wiki_data/'+entity_id+'.json', 'r', encoding='utf-8') as entity_file:
#       entity = json.loads(entity_file.read())
#       claims = entity['entities'][entity_id]['claims']
#       if instanceof_id in claims.keys():
#         instances_of = claims[instanceof_id]

#         # popular_type_index = len(types)
#         for instance_of in instances_of:
#           datavalue = instance_of['mainsnak']['datavalue']
#           entity_type = datavalue['type']
#           if entity_type == 'wikibase-entityid':
#             instance_id = datavalue['value']['id']
#             types_count[instance_id] = types_count.get(instance_id, 0) + 1
#             # index = types.index(instance_id)
#             # if popular_type_index > index:
#             #   popular_type_index = index
#         # types_count[types[popular_type_index]] = types_count.get(types[popular_type_index], 0) + 1

# types_count = sorted(types_count.items(), key=lambda x: x[1], reverse=True)
# with open('sorted_types.txt', 'w', encoding='utf-8') as f:
#   for key, number in types_count:
#     f.write(key + ' ' + str(number) + '\n')

#################
######画图#######
#################
# key_list = []
# number_list = []
# with open('sorted_top_types.txt', 'r', encoding='utf-8') as f:
#   for line in f.readlines()[:35]:
#     key, number = line[:-1].split(' ')
#     name = 'region'
#     if key != 'region':
#       with open('random_wiki_data/{}.json'.format(key), 'r', encoding='utf-8') as entity_file:
#         entity_json = json.loads(entity_file.read())
#         name = entity_json['entities'][key]['labels']['en']['value']
#     key_list.append(name)
#     print(key, name)
#     number_list.append(int(number))
# print(key_list)
# fig, ax = plt.subplots()
# x = range(len(key_list))
# plt.bar(x, height=number_list, width=0.8)
# ax.set_xticks(x)
# ax.set_xticklabels(key_list, rotation=45, ha='right')
# # plt.xticks(x, key_list, rotation=45)
# plt.tight_layout()
# plt.show()

#################
###entity2type###
#################
def entity2type(return_dict, i):
  if i in return_dict:
    return
  with open('random_wiki_data/{}.json'.format(i), 'r', encoding='utf-8') as entity_file:
    entity = json.loads(entity_file.read())
    claims = entity['entities'][str(i)]['claims']
    type_list = []
    if instanceof_id in claims.keys():
      instances_of = claims[instanceof_id]
      for instance_of in instances_of:
        datavalue = instance_of['mainsnak']['datavalue']
        entity_type = datavalue['type']
        if entity_type == 'wikibase-entityid':
          type_list.append(datavalue['value']['id'])
      return_dict[i] = i + ' ' + ','.join(type_list) + '\n'

def entity2type_callback(return_dict, item_ids):
  with open('2M_entity2type.txt', 'a', encoding='utf-8') as f:
    for item_id in item_ids:
      if item_id in return_dict:
        f.write(return_dict[item_id])

#################
###entity2name###
#################
def entity2name(entity_name_list, entity_id):
  with open('random_wiki_data/'+entity_id+'.json', 'r', encoding='utf-8') as entity_file:
    entity = json.loads(entity_file.read())
    if not 'en' in entity['entities'][entity_id]['labels']:
      entity_name_list.append(entity_id + '\t' + '' + '\t' + '' + '\n')
      return
    name = entity['entities'][entity_id]['labels']['en']['value']
    descriptions = ''
    if 'en' in entity['entities'][entity_id]['descriptions']:
      descriptions = entity['entities'][entity_id]['descriptions']['en']['value']
    entity_name_list.append(entity_id + '\t' + name + '\t' + descriptions + '\n')

def entity2name_callback(entity_name_list):
  with open('1M_entity2name.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(entity_name_list))

#################
#entity2pagename#
#################
def entity2pagename(page_name_dict, entity_id):
  with open('random_wiki_data/'+entity_id+'.json', 'r', encoding='utf-8') as entity_file:
    entity = json.loads(entity_file.read())
    if not 'enwiki' in entity['entities'][entity_id]['sitelinks']:
      page_name_dict[entity_id] = entity_id + '\t\t\n'
      return
    name = entity['entities'][entity_id]['sitelinks']['enwiki']['title']
    descriptions = ''
    if 'en' in entity['entities'][entity_id]['descriptions']:
      descriptions = entity['entities'][entity_id]['descriptions']['en']['value']
      page_name_dict[entity_id] = entity_id + '\t' + name + '\t' + descriptions + '\n'

def entity2pagename_callback(page_name_dict, item_ids):
  with open('2M_entity2pagename.txt', 'w', encoding='utf-8') as f:
    for item_id in item_ids:
      if item_id in page_name_dict:
        f.write(page_name_dict[item_id])

#################
###关联实体计数###
#################
def count_related_entity(related_entity_count, i):
  item_id = 'Q'+str(i)
  with open('random_wiki_data/Q{}.json'.format(i), 'r', encoding='utf-8') as entity_file:
    entity = json.loads(entity_file.read())
    claims = entity['entities'][item_id]['claims']
    for claim in claims.keys():
      claim_entity_count = 0
      for claim_value in claims[claim]:
        if 'datavalue' in claim_value['mainsnak']:
          datavalue = claim_value['mainsnak']['datavalue']
          entity_type = datavalue['type']
          if entity_type == 'wikibase-entityid':
            claim_entity_count += 1
            if claim_entity_count >=3:
              break
      related_entity_count[item_id] = related_entity_count.get(item_id, 0) + claim_entity_count

def count_related_entity_callback(related_entity_count):
  sorted_related_entity = sorted(related_entity_count.items(), key=lambda x: x[1], reverse=True)
  with open('2M_related_entity_count.txt', 'w', encoding='utf-8') as f:
    for key, number in sorted_related_entity:
      f.write(key + ' ' + str(number) + '\n')



#################
##删除不存在的Qid#
#################
def remove_invalid(i):
  if not os.path.exists('random_wiki_data/Q{}.json'.format(i)):
    return
  try:
    with open('random_wiki_data/Q{}.json'.format(i), 'r', encoding='utf-8') as f:
      json.loads(f.read())
  except:
    os.remove('random_wiki_data/Q{}.json'.format(i))

#################
##count合法Qid###
#################

'''Q1000000前合法Qid共910688个'''

def count_valid(args):
  result = args[0]
  i = args[1]
  if os.path.exists('random_wiki_data/Q{}.json'.format(i)):
    result.value += 1

# client = paramiko.SSHClient()
# client.load_system_host_keys()
# client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# client.connect('10.176.40.133', 22, 'jiangxueyao', 'aptx4869mdzz')
# scp_client = SCPClient(client.get_transport())
def scp_file(args):
  i = args[0]
  scp_client.put('random_wiki_data/Q{}.json'.format(i), '/home/jiangxueyao/data122/wikidata/random_wiki_data/')

def get_type_dict():
  type_dict = dict()
  with open('above40_entity2type.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      item_id, type_list = line[:-1].split(' ')
      type_list = type_list.split(',')
      for _type in type_list:
        if _type in type_dict:
          type_dict[_type].append(item_id)
        else:
          type_dict[_type] = [item_id]
  with open('type_dict.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(type_dict))

def sort_type_by_count():
  type2name = dict()
  with open('1M_type2name.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      type_id, name, description = line[:-1].split('\t')
      type2name[type_id] = name + '\t' + description
  type_count_dict = dict()
  with open('1M_type2entity.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      type_id, entity_list = line[:-1].split(' ')
      type_count_dict[type_id] = len(entity_list.split(','))
  type_count_list = sorted(type_count_dict.items(), key=lambda x:x[1], reverse=True)
  with open('1M_sorted_type.txt', 'w', encoding='utf-8') as f:
    for type_id, count in type_count_list:
      try:
        f.write(type_id + '\t' + str(count) + '\t' + type2name[type_id] + '\n')
      except:
        print('key error: ' + type_id)

if __name__=='__main__':
  # entity2pagename_dict = dict()
  # with open('1M_entity2pagename.txt', 'r', encoding='utf-8') as f:
  #   for line in f.readlines():
  #     entity_id = line.split('\t')[0]
  #     entity2pagename_dict[entity_id] = line
  # item_ids = []
  # with open('2M_related_entity_count.txt', 'r', encoding='utf-8') as f:
  #   for line in f.readlines():
  #     item_ids.append(line.split(' ')[0])
  # processing = MyMultiProcessing(
  #       4, entity2pagename, item_ids,
  #       ordered=False, returned=True, result_type='dict', init_result=entity2pagename_dict,
  #       callback_func=entity2pagename_callback, callback_args=[item_ids])
  # processing.start()

  entity2type_dict = dict()
  with open('1M_entity2type.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      entity_id = line.split('\t')[0]
      entity2type_dict[entity_id] = line
  item_ids = []
  with open('2M_related_entity_count.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      item_ids.append(line.split(' ')[0])
  processing = MyMultiProcessing(
        4, entity2type, item_ids,
        ordered=False, returned=True, result_type='dict', init_result=entity2type_dict,
        callback_func=entity2type_callback, callback_args=[item_ids])
  processing.start()
