import json
import math
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname='C:/Windows/Fonts/STZHONGS.TTF')
def plot_analysis():
  type2name = dict()
  with open('top_type_name.txt', 'r', encoding='utf-8') as f:
    for line in f.read().split('\n'):
      type_id, name, number = line.split('\t')
      type2name[type_id] = name
  with open('type_image_count.json', 'r') as f:
    data = json.loads(f.read())
  data = sorted(data.items(), key=lambda x: x[1], reverse=True)
  key_list = []
  number_list = []
  for type_id, number in data:
    key_list.append(type2name[type_id])
    number_list.append(number)
  print(key_list)
  # number_list = [math.log(x/min_number)+1 for x in number_list]
  # fig, ax = plt.subplots(figsize=(10,5))
  fig, ax = plt.subplots()
  x = range(len(key_list))
  # sns.color_palette("Blues_r", as_cmap=True)
  # sns.set(palette=sns.color_palette("RdBu", 25))
  sns.set(palette=sns.color_palette("Blues_r", 25))
  sns.barplot(x=key_list, y=number_list)
  # plt.bar(x, height=number_list, width=0.8)
  ax.set_xticks(x)
  ax.set_xticklabels(key_list, rotation=45, ha='right')
  plt.xticks(fontproperties='Times New Roman', fontsize=10)
  plt.yticks(fontproperties='Times New Roman', fontsize=12)
  plt.xlabel('概念名称', fontproperties=myfont, fontsize=12)
  plt.ylabel('图像数量', fontproperties=myfont, fontsize=12)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  # plt.xticks(x, key_list, rotation=45)
  plt.tight_layout()
  # plt.show()
  plt.savefig('type_image_count.pdf')

plot_analysis()