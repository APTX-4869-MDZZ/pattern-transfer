import os
import requests
from MyMultiProcessing import MyMultiProcessing

def download_wikidata(item_id):
  wikidata_url_template = 'https://www.wikidata.org/wiki/Special:EntityData/Q{}.json'
  dir_path = 'random_wiki_data/'
  item_file_name = '{}Q{}.json'.format(dir_path, item_id)
  if os.path.exists(item_file_name):
    return
  wikidata_url = wikidata_url_template.format(item_id)
  response = requests.get(wikidata_url)
  with open(item_file_name, 'w', encoding='utf-8') as f:
    f.write(response.text)

def download_wikidata_type(item_id):
  wikidata_url_template = 'https://www.wikidata.org/wiki/Special:EntityData/{}.json'
  dir_path = 'wiki_type_data/'
  item_file_name = '{}{}.json'.format(dir_path, item_id)
  if os.path.exists(item_file_name):
    return
  wikidata_url = wikidata_url_template.format(item_id)
  response = requests.get(wikidata_url)
  with open(item_file_name, 'w', encoding='utf-8') as f:
    f.write(response.text)

if __name__=='__main__':
  # start_index = 1000000
  # end_index = 2000000
  # iterations = range(start_index, end_index)
  type_list = []
  with open('1M_type_list.txt', 'r', encoding='utf-8') as f:
    type_list = f.read().split('\n')
  multiprocessing = MyMultiProcessing(
      4, download_wikidata_type, type_list
  )
  multiprocessing.start()