
from  scipy.spatial.distance import euclidean
from pyclustering.cluster.gmeans import gmeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
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
with open('patterns.txt', 'r', encoding='utf-8') as f:
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


def included_by_previous_cluster(cluster_instances, data, outlier_threshold):
  for cluster_instance in cluster_instances:
    clf = cluster_instance['isolateForest']
    preds = clf.predict(data)
    # print(preds)
    
    print('{} in {} is outliers.'.format(np.sum(preds == -1), data.shape[0]))
    if np.sum(preds == -1) / data.shape[0] < outlier_threshold:
      return True
  return False
    

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
imageProcessor = ImageFeature(device, 'VGG16')

def pattern_image_overlap(outlier_threshold=0.35):
  overlap_filtered_pattern = codecs.open('pattern_lof_Routlier-{}.txt'.format(outlier_threshold), 'w', 'utf-8')
  for concept in os.listdir('wikipedia_images/'):
    if not concept in patterns:
      continue
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
      clf = LocalOutlierFactor(n_neighbors=5, novelty=True).fit(imageFeatures)
      preds = clf.predict(imageFeatures)
      imageFeatures = imageFeatures[preds != -1]

      if included_by_previous_cluster(cluster_instances, imageFeatures, outlier_threshold):
        continue
      cluster_instances.append({
        'isolateForest': clf,
        'imageFeatures': imageFeatures
      })
      valid_patterns.append(pattern)
    print(valid_patterns)
    overlap_filtered_pattern.write(concept + '\t\t')
    temp_patterns = []
    for pattern_index in valid_patterns:
      temp_patterns.append(patterns[concept][pattern_index])
    overlap_filtered_pattern.write('\t\t'.join(temp_patterns) + '\n')

# distance_threshold = 'isolate_forest'
# outlier_threshold = 0.3
# pattern_image_overlap(distance_threshold, outlier_threshold)
# for distance_threshold in np.arange(1.5, 2, 0.1):
for outlier_threshold in np.arange(0.1, 0.4, 0.05):
  pattern_image_overlap(outlier_threshold)