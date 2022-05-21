from email.mime import image
import requests
import json
import os
from socket import error as SocketError
import errno
import time
from bs4 import BeautifulSoup
from requests.sessions import codes
import codecs
from bs4.element import NavigableString
from MyMultiProcessing import MyMultiProcessing
from tqdm import tqdm
import math
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import copy
stop_words = stopwords.words('english')
for w in [',', '.', '(', ')', "'s", '<', '>', 'text', '[', ']', "''", ':', "``", ';', '-']:
  stop_words.append(w)

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}
proxy = {
  'http': 'http://127.0.0.1:7890',
  'https': 'https://127.0.0.1:7890'
}
PARAMS = {
    "action": "parse",
    "page": "Pet door",
    "format": "json"
}
S = requests.Session()
def safe_request(url, params):
    for _ in range(3):
      try:
        item_response = S.get(url, params=params, headers=headers, proxies=proxy)
        break
      except SocketError as e:
        if e.errno != errno.ECONNRESET:
          raise
        pass
      time.sleep(15)
    return item_response

def download_wikipedia(page_name, item_id):
  if os.path.exists('wikipedia_parse_data/'+item_id+'.txt') and page_name.find('(') == -1:
    return
  URL = "https://en.wikipedia.org/w/api.php"
  PARAMS['page']=page_name
  R = safe_request(URL, PARAMS)
  DATA = R.json()

  with open('wikipedia_parse_data/'+item_id+'.txt', 'w', encoding='utf-8') as f:
    f.write(DATA["parse"]["text"]["*"])

# with open('articles.txt', 'r', encoding='utf-8') as f:
#   for page_name in f.read().split('\n'):
#     download_wikipedia(page_name)

def parse_text(entity_id):
  if not os.path.exists('wikipedia_parse_data/'+entity_id+'.txt'):
    return []
  texts = []
  with open('wikipedia_parse_data/'+entity_id+'.txt', 'r', encoding='utf-8') as f:
    parse_data = BeautifulSoup(f.read(), 'lxml')
    for div in parse_data.find_all('div', {'role': 'navigation'}):
      div.decompose()
    for div in parse_data.find_all('div', {'class': 'plainlist'}):
      div.decompose()
    for div in parse_data.find_all('table', {'role': 'presentation'}):
      div.decompose()
    for img_a in parse_data.find_all('a', 'image'):
      img = img_a.img
      src = img['src']
      alt = ''
      sibling = img_a.next_sibling
      while not alt or len(alt) == 0:
        if isinstance(sibling, NavigableString):
          sibling = sibling.next_sibling
          continue
        if sibling:
          alt = sibling.get_text().strip()
        else:
          break
        sibling = sibling.next_sibling
      if not alt or len(alt) == 0:
        alt = img['alt']
      if not alt or len(alt) == 0:
        alt = '<text>'
      texts.append(src + '\t' + alt + '\n')
  return texts

def parse_wikipedia(type2entity_file, entity2pagename_file, output_file):
  entity2type = dict()
  with open(type2entity_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
      type_id, entity_ids = line[:-1].split(' ')
      # 只取top100做pattern抽取
      for entity_id in entity_ids.split(',')[:100]:
        entity2type[entity_id] = type_id

  entity2wikitext = dict()
  with open(entity2pagename_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
      entity_id = line.split('\t')[0]
      if not entity_id in entity2type:
        continue
      type_id = entity2type[entity_id]
      texts = parse_text(entity_id)
      if len(texts) == 0:
        continue
      if not type_id in entity2wikitext:
        entity2wikitext[type_id] = {}
      entity2wikitext[type_id][entity_id] = texts
  with open(output_file, 'w', encoding='utf-8') as f:
    f.write(json.dumps(entity2wikitext))

def rename_file():
  item_dict = dict()
  with open('test_entity2name.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      item_id, item_name, _ = line.split('\t')
      item_dict[item_name] = item_id
  for file in os.listdir('wikipedia_parse_data/'):
    if file.find('Q') != 0:
      try:
        entity_name = file[:-4]
        entity_name = entity_name.replace('_', ' ')
        item_id = item_dict[entity_name]
        if os.path.exists('wikipedia_parse_data/'+item_id+'.txt'):
          os.remove('wikipedia_parse_data/'+file)
          continue
        os.rename('wikipedia_parse_data/'+file, 'wikipedia_parse_data/'+item_id+'.txt')
      except Exception as e:
        os.remove('wikipedia_parse_data/'+file)
      
def common_str(x, y):
  f = []
  for i in range(len(x)+1):
    f.append([])
    for j in range(len(y)+1):
      f[i].append(0)
  for i in range(1, len(x)+1):
    for j in range(1, len(y)+1):
      f[i][j] = max(f[i-1][j-1] + (1 if x[i-1] == y[j-1] else 0), f[i-1][j], f[i][j-1])
  
  i = len(x)
  j = len(y)
  common_words = []
  while i>0 and j>0:
    if f[i][j] == f[i-1][j-1]+1 and x[i-1] == y[j-1]:
      i = i - 1
      j = j - 1
      common_words.append(x[i])
    elif f[i][j] == f[i-1][j]:
      i = i - 1
      common_words = []
    else:
      j = j - 1
      common_words = []
  common_words.reverse()
  return ' '.join(common_words)

def write_common_pattern(result_dict):
  sorted_word_list = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
  common_text = codecs.open('common_pattern.txt', 'w', 'utf-8')
  for text, value in sorted_word_list:
    common_text.write(text + ' ' + str(value) + '\n')
  common_text.close()

def parse_pattern():
  data_dict = None
  with open('filtered_wikitext.json', 'r', encoding='utf-8') as f:
    data_dict = json.loads(f.read())
  for image_text in data_dict.values():
    entity_num = 50
    print(entity_num)
    pattern_dict = dict()
    pbar = tqdm(total=entity_num*(entity_num-1)//2)
    entity_ids = list(image_text.keys())
    for idx, i in enumerate(entity_ids[:entity_num]):
      for j in entity_ids[idx+1:entity_num]:
        for text_i in image_text[i]:
          for text_j in image_text[j]:
            text_i = text_i[:-1].split('\t')[-1].lower()
            text_j = text_j[:-1].split('\t')[-1].lower()
            pattern = common_str(text_i.split(' '), text_j.split(' '))
            if pattern and len(pattern) != 0:
              pattern_dict[pattern] = pattern_dict.get(pattern, 0)+1
        pbar.update(1)
    pbar.close()
    sorted_word_list = sorted(pattern_dict.items(), key=lambda x: x[1], reverse=True)
    print(sorted_word_list)
    common_text = codecs.open('filtered_common_pattern.txt', 'a', 'utf-8')
    for text, value in sorted_word_list:
      if len(text.split(' ')) > 1:
        common_text.write(text + ' ' + str(value) + '\n')
    common_text.write('\n\n')
    common_text.close()

# 获取单词的词性
def get_wordnet_pos(tag):
  if tag.startswith('J'):
    return wordnet.ADJ
  elif tag.startswith('V'):
    return wordnet.VERB
  elif tag.startswith('N'):
    return wordnet.NOUN
  elif tag.startswith('R'):
    return wordnet.ADV
  else:
    return None

def get_ngram(n, word_list):
  ngram_list = []
  i = 0
  while i < len(word_list):
    while i < len(word_list) and word_list[i][0] in stop_words:
      i = i + 1
    j = 0
    temp_ngram = []
    while j < n and i+j < len(word_list):
      if not word_list[i+j][1][0].isalpha():
        break
      temp_ngram.append(word_list[i+j][0])
      if word_list[i+j][1].startswith('N') and not word_list[i+j][0] in stop_words:
        ngram_list.append(' '.join(temp_ngram))
      j = j + 1
    i = i + 1

  return ngram_list

def split_words(sentence):
  word_list = list()
  tags = pos_tag(word_tokenize(sentence))
  wnl = WordNetLemmatizer()
  for tag in tags:
    wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
    word_list.append((wnl.lemmatize(tag[0], pos=wordnet_pos), tag[1]))
  ngram_list = get_ngram(3, word_list)
  # ngram_list = [' '.join(grams) for grams in ngrams(word_list, 3)]
  return [w[:-4] if len(w) >= 4 and w[-4:] == '.svg' else w for w in ngram_list if not w in stop_words and not w[0]=='.']

def word_count(text_list):
  word_dict = dict()
  word_image = dict()
  for text in text_list:
    image_url = text[:-1].split('\t')[0]
    image_title = '\t'.join(text[:-1].split('\t')[1:])
    image_title = image_title.lower()
    for word in split_words(image_title):
      word_dict[word] = word_dict.get(word, 0) + 1
      if not word in word_image:
        word_image[word] = set()
      word_image[word].add(image_url)
  return word_dict, word_image

def check_overlap(common_words, current_word, tf_count):
  i = 0
  while i < len(common_words) and common_words[i][0].find(current_word) == -1:
    i = i + 1
  if i >= len(common_words):
    common_words.append([current_word, tf_count])

def filter_long_overlap(common_words):
  i = 0
  while i < len(common_words):
    j = 0
    while j < i:
      if common_words[i][0].find(common_words[j][0]) != -1:
        del common_words[i]
        break
      j = j + 1
    if j == i:
      i = i + 1

def tf(wikitext_file, type2name_file, pattern_file, pattern_image_file):
  with open(wikitext_file, 'r', encoding='utf-8') as f:
    data_dict = json.loads(f.read())
    
  type2name = dict()
  with open(type2name_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
      type_id, name, _ = line.split('\t')
      if type_id in data_dict.keys():
        type2name[type_id] = name

  # word_dict = dict()
  # with open('filtered_wikitext.json', 'r', encoding='utf-8') as f:
  #   data_dict = json.loads(f.read())
  #   for type_id in data_dict:
  #     word_set = set()
  #     for entity_id in data_dict[type_id]:
  #       entity_word_dict, word_image = word_count(data_dict[type_id][entity_id])
  #       for word in entity_word_dict:
  #         word_set.add(word)
  #     for word in word_set:
  #       word_dict[word] = word_dict.get(word, 0) + 1

  type_num = len(list(data_dict.keys()))
  common_word_file = codecs.open(pattern_file, 'w', 'utf-8')
  word_image_file = codecs.open(pattern_image_file, 'w', 'utf-8')
  for type_id in data_dict:
    if not type_id in type2name:
      continue
    type_word_dict = dict()
    type_word_image = dict()
    for entity_id in data_dict[type_id]:
      entity_word_dict, word_image = word_count(data_dict[type_id][entity_id])
      for word in entity_word_dict:
        # if entity_word_dict.get(word) > 3:
        #   continue
        type_word_dict[word] = type_word_dict.get(word, 0) + 1
        if not word in type_word_image:
          type_word_image[word] = set()
        type_word_image[word] = type_word_image[word] | word_image[word]
    
    pure_type_name = type2name[type_id].lower().split(' (')[0]
    # type_word_dict[pure_type_name] = 0
    sorted_words = sorted(type_word_dict.items(), key=lambda x:(x[1], len(x[0])), reverse=True)
    common_word_file.write(type_id + ' ' + type2name[type_id] + '\n')
    word_image_file.write(type_id + ' ' + type2name[type_id] + '\n')
    common_words = []
    print('\n'.join([sort_pair[0] + ' ' + str(sort_pair[1]) for sort_pair in sorted_words[:20]]))
    print('\n')
    for word, tf_count in sorted_words:
      # if word_dict.get(word, 0) > type_num*2/3:
      #   continue
      if word.find(pure_type_name) != -1:
        continue
      check_overlap(common_words, word, tf_count)
      # print('\n'.join([word[0] + '\t' + str(word[1]) for word in common_words]))
      if len(common_words) == 30:
        break
    filter_long_overlap(common_words)
    common_word_file.write('\n'.join([word[0] + '\t\t' + str(word[1]) for word in common_words]))
    word_image_file.write('\n'.join([word[0] + '\t\t' + str(word[1]) + '\t\t' + '\t'.join(type_word_image[word[0]]) for word in common_words]))
    common_word_file.write('\n\n')
    word_image_file.write('\n\n')

def check_words():
  with open('filtered_wikitext.json', 'r', encoding='utf-8') as f:
    wikitext = json.loads(f.read())
    with open('temp.json', 'w', encoding='utf-8') as file:
      for entity_id in wikitext['Q5']:
        word_freq = word_count(wikitext['Q5'][entity_id])
        print(word_freq.get('leave', 0))
        break


headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}
def safe_request(url):
  for _ in range(3):
    try:
      item_response = requests.get(url, headers=headers)
      break
    except SocketError as e:
      if e.errno != errno.ECONNRESET:
        raise
      pass
    time.sleep(15)
  return item_response

def download_image(img_url, img_path, file_name):
  if not os.path.exists(img_path):
    os.makedirs(img_path)
  extension = img_url[-4:]
  if extension == 'jpeg':
    extension = '.jpeg'
  img_response = safe_request(img_url)
  with open(img_path+'/'+file_name+extension, 'wb') as file:
    file.write(img_response.content)

def download_wiki_image():
  with open('selected_type/pattern_images.txt', 'r', encoding='utf-8') as f:
    for segment in f.read().split('\n\n'):
      concept_line = segment.split('\n')[0]
      if os.path.exists('wikipedia_images/'+concept_line):
        continue
      for word_index, pattern_line in enumerate(segment.split('\n')[1:]):
        _, _, images = pattern_line.split('\t\t')
        image_url_list = images.split('\t')
        image_path = 'wikipedia_images/'+concept_line+'/'+str(word_index)
        for image_index, image_url in enumerate(image_url_list):
          download_image('https:'+image_url, image_path, str(image_index))
        


if __name__=='__main__':
  # names = []
  # item_ids = []
  # with open('1M_entity2pagename.txt', 'r', encoding='utf-8') as f:
  #   for line in f.readlines():
  #     item_id, name = line.split('\t')[0:2]
  #     item_ids.append(item_id)
  #     names.append(name)

  # filtered_names = []
  # filtered_ids = []
  # with open('2021-12-24 13_58_46 error.log', 'r', encoding='utf-8') as f:
  #   for line in f.readlines():
  #     if line.find('HTTPSConnectionPool') != -1:
  #       index = int(line.split(':')[0][6:])
  #       filtered_names.append(names[index])
  #       filtered_ids.append(item_ids[index])
  # iterations = list(zip(filtered_names, filtered_ids))

  # entity_ids = []
  # page_names = []
  # with open('selected_type/entity2pagename.txt', 'r', encoding='utf-8') as f:
  #   for line in f.readlines():
  #     entity_id, page_name = line.split('\t')[0:2]
  #     entity_ids.append(entity_id)
  #     page_names.append(page_name)
  # iterations = list(zip(page_names, entity_ids))
  # processing = MyMultiProcessing(
  #     4, download_wikipedia, iterations, ordered=False
  # )
  # processing.start()


  # parse_wikipedia('selected_type/head_type2entity.txt', 'selected_type/entity2pagename.txt', 'selected_type/wikitext.json')
  tf('selected_type/wikitext.json', 'selected_type/top_type_name.txt', 'selected_type/patterns.txt', 'selected_type/pattern_images.txt')
  # download_wiki_image()
  # print(parse_text('Q131074'))

  # check_words()
  # with open('filtered_entity2pagename.txt', 'r', encoding='utf-8') as f:
  #   for line in tqdm(f.readlines()):
  #     entity_id, pagename, _ = line.split('\t')
  #     download_wikipedia(pagename, entity_id)