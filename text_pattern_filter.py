
from  scipy.spatial.distance import euclidean
from pyclustering.cluster.gmeans import gmeans
from sklearn.ensemble import IsolationForest
import numpy as np
import math
import torch
import codecs
import os

from imageFeature_torch import ImageFeature

candidate_pattern_num = 6
gmeans_args = {
  'repeat': 30,
  'k_max': 5
}

patterns = dict()
with open('filtered_common_words_without_stopwords.txt', 'r', encoding='utf-8') as f:
  for segment in f.read().split('\n\n'):
    concept = segment.split('\n')[0]
    patterns[concept] = [line.split('\t\t')[0] for line in segment.split('\n')[1:candidate_pattern_num+1]]

def get_cluster_mean_dist(data, labels, centers):
  dists = np.array([euclidean(data[i], centers[labels[i]]) if labels[i] != -1 else 999999999 for i in range(data.shape[0])])
  avg_dist = []
  std_dist = []
  for i in range(len(centers)):
    avg_dist.append(np.mean(dists[labels == i]))
    std_dist.append(np.sqrt(np.sum(np.square(dists[labels == i]-avg_dist[i]))/np.sum(labels == i)))
  return avg_dist, std_dist

def find_outlier_with_kmeans(data, labels, centers, avg_dists, max_l):
  dists = np.array([euclidean(data[i], centers[labels[i]]) for i in range(data.shape[0])])
  ratio = np.zeros((data.shape[0]))
  for i in range(len(centers)):
    avg_dist = avg_dists[i]
    if math.fabs(avg_dist) < 0.001:
      ratio[labels == i] = 10
    else:
      ratio[labels == i] = dists[labels == i] / avg_dist
  return np.argwhere(ratio > max_l)

def find_outlier_with_std(data, labels, centers, avg_dists, std_dists):
  dists = np.array([euclidean(data[i], centers[labels[i]]) for i in range(data.shape[0])])
  outliers = []
  for i in range(data.shape[0]):
    avg_dist = avg_dists[labels[i]]
    std_dist = std_dists[labels[i]]
    if dists[i] - avg_dist > 3 * std_dist:
      outliers.append(i)
  return outliers


def included_by_previous_cluster(cluster_instances, data, distance_threshold, outlier_threshold):
  for cluster_instance in cluster_instances:
    gmeans_instance = cluster_instance['gmeans_instance']
    mean_dists = cluster_instance['mean_dists']
    std_dists = cluster_instance['std_dists']
    predict_labels = gmeans_instance.predict(data)
    centers = gmeans_instance.get_centers()
    outliers = find_outlier_with_std(data, predict_labels, centers, mean_dists, std_dists)
    print('{} in {} is outliers.'.format(len(outliers), data.shape[0]))
    if len(outliers) / data.shape[0] < outlier_threshold:
      return True
  return False
    

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
imageProcessor = ImageFeature(device, 'VGG16')

def pattern_image_overlap(distance_threshold=1.6, outlier_threshold=0.35):
  overlap_filtered_pattern = codecs.open('pattern_Rdist-{}_Routlier-{}.txt'.format(distance_threshold, outlier_threshold), 'w', 'utf-8')
  for concept in os.listdir('wikipedia_images/'):
    cluster_instances = []
    valid_patterns = []
    print(concept)
    for pattern in range(candidate_pattern_num):
      if len(os.listdir('wikipedia_images/'+concept+'/'+str(pattern))) < 20:
        continue
      print('pattern {}'.format(pattern))
      imageFeatures = []
      for image_file in os.listdir('wikipedia_images/'+concept+'/'+str(pattern)):
        imageFeatures.append(imageProcessor.get_image_feature('wikipedia_images/'+concept+'/'+str(pattern)+'/'+image_file))
      imageFeatures = np.array(imageFeatures).squeeze()

      if included_by_previous_cluster(cluster_instances, imageFeatures, distance_threshold, outlier_threshold):
        continue
      gmeans_instance = gmeans(imageFeatures, **gmeans_args).process()
      clusters = gmeans_instance.get_clusters()
      centers = gmeans_instance.get_centers()
      ### 删去单节点聚类
      cluster_i = 0
      while cluster_i < len(clusters):
        if len(clusters[cluster_i]) == 1:
          del clusters[cluster_i]
          del centers[cluster_i]
        else:
          cluster_i += 1

      ### 得到聚类label
      labels = np.zeros((imageFeatures.shape[0], ), dtype=np.int)
      for index in range(imageFeatures.shape[0]):
        labels[index] = -1
      print('concept {}, pattern {}: {} clusters'.format(concept, pattern, len(clusters)))
      for cluster_i, cluster in enumerate(clusters):
        for index in cluster:
          labels[index] = cluster_i
      ### 计算离群点
      mean_dists, std_dists = get_cluster_mean_dist(imageFeatures, labels, centers)
      # outliers = find_outlier_with_kmeans(imageFeatures, labels, centers, mean_dists, std_dist, distance_threshold)
      outliers = find_outlier_with_std(imageFeatures, labels, centers, mean_dists, std_dists)
      for index in outliers:
        labels[index] = -1
      cluster_instances.append({
        'gmeans_instance':gmeans_instance,
        'mean_dists': mean_dists,
        'std_dists': std_dists
      })
      valid_patterns.append(pattern)
    print(valid_patterns)
    overlap_filtered_pattern.write(concept + '\t\t')
    temp_patterns = []
    for pattern_index in valid_patterns:
      temp_patterns.append(patterns[concept][pattern_index])
    overlap_filtered_pattern.write('\t\t'.join(temp_patterns) + '\n')

distance_threshold = 'std'
outlier_threshold = 0.3
pattern_image_overlap(distance_threshold, outlier_threshold)
# for distance_threshold in np.arange(1.5, 2, 0.1):
#   for outlier_threshold in np.arange(0.3, 0.4, 0.01):
#     pattern_image_overlap(distance_threshold, outlier_threshold)