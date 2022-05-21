from turtle import distance
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS 
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import MeanShift, estimate_bandwidth
from pyclustering.cluster.gmeans import gmeans
import numpy as np
import torch
import os
import math
import shutil
import json
import codecs

import matplotlib.pyplot as plt

import plot_utils
from imageFeature_torch import ImageFeature
import scipy
from  scipy.spatial.distance import euclidean

k_means_args_dict = {
    'n_clusters': 0,
    # drastically saves convergence time
    'init': 'k-means++',
    'verbose': False,
    # 'n_jobs':8
}

distance_threshold = 2
outlier_threshold = 0.3

# Gap Statistic for K means
def optimalK(data, nrefs=3, maxClusters=15):
  """
  Calculates KMeans optimal K using Gap Statistic 
  Params:
      data: ndarry of shape (n_samples, n_features)
      nrefs: number of sample reference datasets to create
      maxClusters: Maximum number of clusters to test for
  Returns: (gaps, optimalK)
  """
  gaps = np.zeros((len(range(1, maxClusters)),))
  sks = np.zeros((len(range(1, maxClusters)),))
  for gap_index, k in enumerate(range(1, maxClusters)):
    refDisps = np.zeros(nrefs)
    for i in range(nrefs):
      randomReference = np.random.random_sample(size=data.shape)
      km = KMeans(k)
      km.fit(randomReference)
      refDisps[i] = km.inertia_
    sks[gap_index] = math.sqrt((nrefs+1)/nrefs)*np.std(np.log(refDisps))
    km = KMeans(k)
    km.fit(data)
    origDisp = km.inertia_
    gap = np.log(np.mean(refDisps)) - np.log(origDisp)
    gaps[gap_index] = gap

  print(gaps[:-1]-gaps[1:]+sks[1:])
  k = np.argwhere(gaps[:-1]-gaps[1:]+sks[1:] >= 0)[0][0] + 1
  return k

def get_cluster_mean_dist(data, labels, centers):
  dists = np.array([euclidean(data[i], centers[labels[i]]) for i in range(data.shape[0])])
  avg_dist = []
  for i in range(len(centers)):
    avg_dist.append(np.mean(dists[labels == i]))
  return avg_dist

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

def calculate_inner_score(data):
  inner_score = 0
  data_num = data.shape[0]
  for i in range(data_num):
    for j in range(data_num):
      inner_score += euclidean(data[i], data[j])
  return inner_score / data_num / (data_num-1)

def calculate_image_variance(data, k):
  k_means_args_dict['n_clusters'] = k
  kmeans = KMeans(**k_means_args_dict)
  kmeans.fit(data)
  return kmeans.inertia_ / data.shape[0]

patterns = dict()
with open('filtered_common_words_without_stopwords.txt', 'r', encoding='utf-8') as f:
  for segment in f.read().split('\n\n'):
    concept = segment.split('\n')[0]
    patterns[concept] = [line.split('\t\t')[0] for line in segment.split('\n')[1:11]]

def check_similarity(data_i, data_j):
  distance_sum = 0
  for i in range(data_i.shape[0]):
    for j in range(data_j.shape[0]):
      distance_sum += euclidean(data_i[i], data_j[j])
  return distance_sum
   
def included_by_previous_cluster(cluster_instances, data):
  for cluster_instance in cluster_instances:
    gmeans_instance = cluster_instance['gmeans_instance']
    mean_dists = cluster_instance['mean_dists']
    predict_labels = gmeans_instance.predict(data)
    centers = gmeans_instance.get_centers()
    outliers = find_outlier_with_kmeans(data, predict_labels, centers, mean_dists, distance_threshold)
    print('{} in {} is outliers.'.format(len(outliers), data.shape[0]))
    if len(outliers) / data.shape[0] < outlier_threshold:
      return True
  return False
    

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
imageProcessor = ImageFeature(device, 'VGG16')

def pattern_image_overlap():
  overlap_filtered_pattern = codecs.open('overlap_filtered_pattern.txt', 'w', 'utf-8')
  for concept in os.listdir('wikipedia_images/'):
    cluster_instances = []
    valid_patterns = []
    print(concept)
    for pattern in range(7):
      print('pattern {}'.format(pattern))
      imageFeatures = []
      for image_file in os.listdir('wikipedia_images/'+concept+'/'+str(pattern)):
        imageFeatures.append(imageProcessor.get_image_feature('wikipedia_images/'+concept+'/'+str(pattern)+'/'+image_file))
      imageFeatures = np.array(imageFeatures).squeeze()

      if included_by_previous_cluster(cluster_instances, imageFeatures):
        continue
      gmeans_instance = gmeans(imageFeatures, repeat=30, k_max=5).process()
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
      mean_dists = get_cluster_mean_dist(imageFeatures, labels, centers)
      outliers = find_outlier_with_kmeans(imageFeatures, labels, centers, mean_dists, distance_threshold)
      for index in outliers:
        labels[index] = -1
      cluster_instances.append({
        'gmeans_instance':gmeans_instance,
        'mean_dists': mean_dists
      })
      valid_patterns.append(pattern)
    print(valid_patterns)
    overlap_filtered_pattern.write(concept + '\t\t')
    temp_patterns = []
    for pattern_index in valid_patterns:
      temp_patterns.append(patterns[concept][pattern_index])
    overlap_filtered_pattern.write('\t\t'.join(temp_patterns) + '\n')

def temp():     
  cluster_method = 'gmeans'
  # thresholds = [1.5, 2, 2.5, 3]
  thresholds = [2]
  for threshold in thresholds:
    new_patterns = codecs.open('new_patterns-{}.txt'.format(threshold), 'w', 'utf-8')

    for concept in os.listdir('wikipedia_images/'):
      concept = 'Q5 Human'
      variances = []
      inner_scores = []

      for pattern in range(10):
        pattern = str(pattern)
        imageFeatures = []
        for image_file in os.listdir('wikipedia_images/'+concept+'/'+pattern):
          imageFeatures.append(imageProcessor.get_image_feature('wikipedia_images/'+concept+'/'+pattern+'/'+image_file))
        imageFeatures = np.array(imageFeatures).squeeze()
        if imageFeatures.shape[0] < 16:
          print(concept + ' ' + pattern)
          inner_scores.append(999999999)
          continue
        # imageFeatures_for_clustering = PCA(n_components=16).fit_transform(imageFeatures)
        imageFeatures_for_clustering = imageFeatures

        if cluster_method == 'kmeans':
          imageFeatures_for_kmeans = imageFeatures
          scores = []
          for k in range(2, 7):
            k_means_args_dict['n_clusters'] = k
            k_means_args_dict['n_init'] = 5
            kmeans = KMeans(**k_means_args_dict).fit(imageFeatures_for_kmeans)
            
            score = calinski_harabasz_score(imageFeatures_for_kmeans, kmeans.labels_)
            scores.append(score)
          k = np.argmax(np.array(scores))+2
          # k = optimalK(imageFeatures_for_kmeans, nrefs=5, maxClusters=15)
          print('concept {}, pattern {}: {} clusters'.format(concept, pattern, k))

          k_means_args_dict['n_clusters'] = k
          k_means_args_dict['n_init'] = 5
          clustering = KMeans(**k_means_args_dict).fit(imageFeatures_for_kmeans)
          images = ['wikipedia_images/'+concept+'/'+pattern+'/'+image_file for image_file in os.listdir('wikipedia_images/'+concept+'/'+pattern)]
          cluster_dir = 'pattern'+str(pattern)
          if os.path.exists(cluster_dir):
            shutil.rmtree(cluster_dir)
          os.mkdir(cluster_dir)
          plot_utils.show_cluster_images(images, clustering.labels_, cluster_dir+'/kmeans_')
        elif cluster_method == 'DBSCAN':
          imageFeatures_for_dbscan = StandardScaler().fit_transform(imageFeatures)
          clustering = DBSCAN(eps=30, min_samples=8).fit(imageFeatures_for_dbscan)
          print(clustering.labels_)
          images = ['wikipedia_images/'+concept+'/'+pattern+'/'+image_file for image_file in os.listdir('wikipedia_images/'+concept+'/'+pattern)]
          plot_utils.show_cluster_images(images, clustering.labels_, 'dbscan_')
        elif cluster_method == 'OPTICS':
          clustering = OPTICS(min_samples=8).fit(imageFeatures)
          print(clustering.labels_)
          images = ['wikipedia_images/'+concept+'/'+pattern+'/'+image_file for image_file in os.listdir('wikipedia_images/'+concept+'/'+pattern)]
          plot_utils.show_cluster_images(images, clustering.labels_, 'OPTICS_')
        elif cluster_method == 'meanShift':
          bandwidth = estimate_bandwidth(imageFeatures)
          clustering = MeanShift(bandwidth=bandwidth).fit(imageFeatures)
          print(clustering.labels_)
          images = ['wikipedia_images/'+concept+'/'+pattern+'/'+image_file for image_file in os.listdir('wikipedia_images/'+concept+'/'+pattern)]
          if os.path.exists('pattern'+str(iter)):
            shutil.rmtree('pattern'+str(iter))
          os.mkdir('pattern'+str(iter))
          plot_utils.show_cluster_images(images, clustering.labels_, 'pattern'+str(iter)+'/meanShift_')
        elif cluster_method == 'gmeans':
          ### gmeans
          gmeans_instance = gmeans(imageFeatures_for_clustering, repeat=20).process()
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
          labels = np.zeros((imageFeatures_for_clustering.shape[0], ), dtype=np.int)
          for index in range(imageFeatures_for_clustering.shape[0]):
            labels[index] = -1
          print('concept {}, pattern {}: {} clusters'.format(concept, pattern, len(clusters)))
          for cluster_i, cluster in enumerate(clusters):
            for index in cluster:
              labels[index] = cluster_i
          ### 计算离群点
          outliers = find_outlier_with_kmeans(imageFeatures_for_clustering, labels, centers, threshold)
          for index in outliers:
            labels[index] = -1
          variances.append(len(outliers)/labels.shape[0])

          ### 计算内聚性
          labels = np.array(labels).squeeze()
          inner_score = 0
          label_num = np.max(labels)+1
          for label in range(label_num):
            inner_score += calculate_inner_score(imageFeatures_for_clustering[labels == label])
          inner_score /= label_num
          inner_scores.append(inner_score)
          ### 绘图
          images = ['wikipedia_images/'+concept+'/'+pattern+'/'+image_file for image_file in os.listdir('wikipedia_images/'+concept+'/'+pattern)]
          cluster_dir = 'gmeans_cluster/pattern'+str(pattern)
          if os.path.exists(cluster_dir):
            shutil.rmtree(cluster_dir)
          os.mkdir(cluster_dir)
          plot_utils.show_cluster_images(images, labels, cluster_dir+'/gmeans_')
      
      # variances = np.array(variances).squeeze()
      # print(variances)
      # print()
      print(np.argsort(inner_scores).squeeze())
      concept = concept.split(' ')[0]
      new_patterns.write(concept + '\t\t')
      temp_patterns = patterns[concept][:5]
      for pattern_index in np.argsort(inner_scores).squeeze()[:5]:
        if pattern_index >= 5:
          temp_patterns.append(patterns[concept][pattern_index])
      new_patterns.write('\t\t'.join(temp_patterns) + '\n')
      break
    new_patterns.close()

    pattern_json = dict()
    with open('new_patterns-{}.txt'.format(threshold), 'r', encoding='utf-8') as f:
      for line in f.readlines():
        concept = line[:-1].split('\t\t')[0]
        pattern_json[concept] = line[:-1].split('\t\t')[1:]
    with open('filtered_patterns-{}.json'.format(threshold), 'w', encoding='utf-8') as f:
      f.write(json.dumps(pattern_json))

pattern_image_overlap()