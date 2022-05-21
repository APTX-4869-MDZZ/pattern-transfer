import os
import json
import requests
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
import matplotlib.font_manager as font_manager
font_manager._rebuild()
import math
import seaborn as sns

top_type_name = []

def download_wikidata_type(item_id, count_):
  dir_path = 'wiki_type_data/'
  item_file_name = '{}{}.json'.format(dir_path, item_id)
  if os.path.exists(item_file_name):
    with open(item_file_name, 'r', encoding='utf-8') as f:
      entity = json.loads(f.read())
  else:
    wikidata_url_template = 'https://www.wikidata.org/wiki/Special:EntityData/{}.json'
    wikidata_url = wikidata_url_template.format(item_id)
    response = requests.get(wikidata_url)
    with open(item_file_name, 'w', encoding='utf-8') as f:
      f.write(response.text)
    entity = json.loads(response.text)

  if not 'enwiki' in entity['entities'][item_id]['sitelinks']:
    # top_type_name.append(item_id + '\t\n')
    return
  name = entity['entities'][item_id]['sitelinks']['enwiki']['title']
  top_type_name.append(item_id + '\t' + name + '\t' + count_ + '\n')

def get_type_name():
  entity2pagename = dict()
  with open('2M_entity2pagename.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      entity_id, pagename, _ = line.split('\t')
      entity2pagename[entity_id] = pagename

  with open('top_type_from_server.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      entity_id, count_ = line[:-1].split('\t')
      if entity_id in entity2pagename:
        if entity2pagename[entity_id] != '':
          top_type_name.append(entity_id + '\t' + entity2pagename[entity_id] + '\t' + count_ + '\n')
      else:
        download_wikidata_type(entity_id, count_)

  with open('top_type_name.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(top_type_name))

def plot_analysis():
  key_list = []
  number_list = []
  with open('top_type_name.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines()[:25]:
      name, number = line[:-1].split('\t')[1:]
      key_list.append(name)
      number_list.append(int(number))
  print(key_list)
  # min_number = number_list[-1]
  # number_list = [math.log(x/min_number)+1 for x in number_list]
  number_list = [math.log(x) for x in number_list]
  fig, ax = plt.subplots(figsize=(10,5))
  # fig, ax = plt.subplots()
  x = range(len(key_list))
  # sns.color_palette("light:b", as_cmap=True)
  # sns.set(palette=sns.color_palette("RdBu", 25))
  sns.set(palette=sns.color_palette("light:black_r", 25))
  sns.barplot(x=key_list, y=number_list)
  # plt.bar(x, height=number_list, width=0.8)
  ax.set_xticks(x)
  ax.set_xticklabels(key_list, rotation=45, ha='right')
  plt.xticks(fontproperties='Times New Roman',fontsize=14)
  plt.yticks(fontproperties='Times New Roman',fontsize=14)
  plt.xlabel('Concept Name', fontproperties='Times New Roman',fontsize=14)
  plt.ylabel('log(#entity)', fontproperties='Times New Roman',fontsize=14)
  # plt.xticks(x, key_list, rotation=45)
  plt.tight_layout()
  # plt.show()
  plt.savefig('4_fig-statistics.svg')

plot_analysis()