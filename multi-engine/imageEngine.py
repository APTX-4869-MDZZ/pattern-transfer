import requests
from bs4 import BeautifulSoup
import base64
import re
import json
import os
import shutil
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from urllib.parse  import unquote
from urllib.parse import quote
from socket import error as SocketError
import errno
import time

location_driver = 'E:\curiosity2.0\multi-engine\chromedriver.exe'
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}

class ImageSearchEngine(object):
  def __init__(self, k=20, engine='baidu', data_path='data/image/', quiet=False):
    self.proxy_addrs = {
      'http': 'http://127.0.0.1:7890',
      'https': 'https://127.0.0.1:7890',
      # 'http': 'http://211.144.213.145:8080',
      # 'https': 'https://211.144.213.145:8080',
    }
    self.search_patterns = {
      'google': 'https://www.google.com/search?q={}&source=lnms&tbm=isch',
      'baidu': 'https://image.baidu.com/search/index?tn=baiduimage&ie=utf-8&word={}&rn={}',
      'bing': 'https://cn.bing.com/images/search?q={}',
      'yahoo': 'https://images.search.yahoo.com/search/images;?p={}',
      'aol': 'https://search.aol.com/aol/image;?q={}',
      'duckduckgo': 'https://duckduckgo.com/?q={}&iax=images&ia=images'
    }
    
    self.k = k
    self.engine = engine
    self.data_path = data_path
    self.quiet = quiet
    self.driver = self.start_brower()
    self.searchEngine = {
      'baidu': self.get_topk_from_baidu,
      'google': self.get_topk_from_google,
      'bing': self.get_topk_from_bing,
      'yahoo': self.get_topk_from_yahoo,
      'aol': self.get_topk_from_yahoo,
      'duckduckgo': self.get_topk_from_duckduckgo
    }

  def start_brower(self):
    chrome_options = Options()
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument('--proxy-server=127.0.0.1:7890')
    chrome_options.add_argument('--log-level=2')
    if self.quiet:
      chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(executable_path=location_driver, chrome_options=chrome_options)
    driver.maximize_window()  
    return driver
  
   
  def safe_request(self, url):
    for _ in range(3):
      try:
        item_response = requests.get(url, headers=headers)
        break
      except SocketError as e:
        if e.errno != errno.ECONNRESET:
          raise
        pass
      time.sleep(3)
    return item_response

  def get_topk_from_baidu(self, engine, html, s, folder_name):
    image_lis = html.find_all('li', 'imgitem')
    for index, image_li in enumerate(image_lis):
      if index == self.k:
        break
      self.download_image(engine, image_li['data-objurl'], s, index, folder_name)

  def get_topk_from_google(self, engine, html, s, forder_name):
    image_lis = html.find_all('div', class_='isv-r PNCib MSM1fd BUooTd')
    for index, image_li in enumerate(image_lis):
      if index == self.k:
        break
      a = image_li.find('a')
      img = a.find('img')
      if img.get('src', None):
        if re.search(r'data:image/(.*);', img['src']):
          self.decode_image(engine, img['src'], s, index, forder_name)
        else:
          self.download_image(engine, img['src'], s, index, forder_name)
  
  def get_topk_from_bing(self, engine, html, s, folder_name):
    image_as = html.find_all('a', class_='iusc')
    for index, image_a in enumerate(image_as):
      if index == self.k:
        break
      img = image_a.find('img')
      if img.get('src', None):
        if re.search(r'data:image/(.*);', img['src']):
          self.decode_image(engine, img['src'], s, index, folder_name)
        else:
          self.download_image(engine, img['src'], s, index, folder_name)

  def get_topk_from_yahoo(self, engine, html, s, folder_name):
    image_lis = html.find_all('li', class_='ld')
    index = 0
    for image_li in image_lis:
      if index == self.k:
        break
      img = image_li.find('img')
      if img.get('src', None):
        if re.search(r'data:image/(.*);', img['src']):
          success = self.decode_image(engine, img['src'], s, index, folder_name)
          if success == 'yes':
            index += 1
        else:
          success = self.download_image(engine, img['src'], s, index, folder_name)
          if success == 'yes':
            index += 1

  def get_topk_from_aol(self):
    pass

  def get_topk_from_duckduckgo(self, engine, html, s, folder_name):
    image_divs = html.find_all('div', class_='tile tile--img has-detail')
    for index, image_div in enumerate(image_divs):
      if index == self.k:
        break
      img = image_div.find('img')
      if img.get('src', None):
        if re.search(r'data:image/(.*);', img['src']):
          self.decode_image(engine, img['src'], s, index, folder_name)
        else:
          http_position = img['src'].find('https')
          self.download_image(engine, unquote(img['src'][http_position:]), s, index, folder_name)

  def download_image(self, engine, image_url, s, index, forder_name):
    image_folder = '{}{}/'.format(self.data_path, forder_name)
    if not os.path.exists(image_folder):
      os.makedirs(image_folder)
    try:
      response = self.safe_request(image_url)
      content_type = response.headers['content-type']
      slash_postition = content_type.rfind('/')
      img_type = content_type[slash_postition+1:]
      with open(image_folder + '{}.{}'.format(index, img_type), 'wb') as file:
        file.write(response.content)
      return 'yes'
    except Exception as e:
      # print(e)
      # print(image_url)
      return None

  def decode_image(self, engine, image_src, s, index, forder_name):
    image_folder = '{}{}/'.format(self.data_path, forder_name)
    if not os.path.exists(image_folder):
      os.makedirs(image_folder)
    try:
      img_type = re.search(r'data:image/(.*);', image_src).groups()[0]
      image_src = re.sub(r'data:image/(.*);base64,', '', image_src)
      with open(image_folder + '{}.{}'.format(index, img_type), 'wb') as file:
        file.write(base64.b64decode(image_src))
      return 'yes'
    except:
      # print(image_src)
      return None

  def get_topk_from_engine(self, s, folder_name=None):
    if not folder_name:
      folder_name = re.sub(r'[\"|\/|\?|\*|\:|\||\\|\<|\>]', ' ', s)
    if os.path.exists(self.data_path+folder_name) and len(os.listdir(self.data_path+folder_name)) >= self.k:
      return
    if not self.driver:
      self.driver = self.start_brower()
    for engine in self.engine:
      search_pattern = self.search_patterns[engine]
      search_url = search_pattern.format(quote(s, 'utf-8'), self.k)
      while True:
        try:
          self.driver.get(search_url)
          jsCode = "var q=document.documentElement.scrollTop=1000"
          self.driver.execute_script(jsCode)
          jsCode = "var q=document.documentElement.scrollTop=1500"
          self.driver.execute_script(jsCode)
          break
        except TimeoutException as e:
          pass
      html = BeautifulSoup(self.driver.page_source, 'html.parser')
      self.searchEngine[engine](engine, html, s, folder_name)

if __name__ == "__main__":
  imageEngine = ImageSearchEngine(engine=['duckduckgo'])
  imageEngine.get_topk_from_engine('博美犬', '动物')