import csv
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode

def save_cluster_statistics(cluster_idx, assigned_label, pct_major_label, seed, file_path="results/severity_code/cluster_labels.csv"):
  append = seed != 0 or (seed == 0 and cluster_idx != 0)
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  file_mode = "a" if append else "w"

  with open(file_path, file_mode, newline="") as f:
    writer = csv.writer(f)
    if not append:
      writer.writerow(["seed", "cluster", "assigned_label", "pct_major_label"])
    writer.writerow([seed, cluster_idx, assigned_label, round(pct_major_label * 100, 2)])

class KMeansClassifier():
  def __init__(self, seed, n_clusters=None):
    self.seed = seed
    self.n_clusters = n_clusters
    self.cluster_labels = None
    self.labels = None

  def fit(self, X_train, y_train):
    self.labels = np.unique(y_train)
    self.n_clusters = len(self.labels) if self.n_clusters is None else self.n_clusters
    self.cluster_labels = np.full(self.n_clusters, -1, dtype=int)

    self.kmeans = KMeans(n_clusters=self.n_clusters)
    self.kmeans.fit(X_train)

    # count labels in each cluster
    for i in range(self.n_clusters):
      labels_count = []
      for label in self.labels:
        label_count = np.sum(y_train[self.kmeans.labels_ == i] == label)
        labels_count.append(label_count)
      # assign the cluster to the label with most samples
      label_index = np.argmax(labels_count)
      self.cluster_labels[i] = self.labels[label_index]

      save_cluster_statistics(i, self.labels[label_index], max(labels_count) / sum(labels_count), self.seed)
  
    return self

  def predict(self, X_test):
    # predict the cluster for each sample by distance
    cluster_indexes = self.kmeans.predict(X_test)
    # assign the label of the cluster to each sample
    return self.cluster_labels[cluster_indexes]

class TreesEnsemble():
  def __init__(self, n_estimators):
    self.n_estimators = n_estimators
    self.trees = list()

  def fit(self, X_train, y_train):
    for i in range(self.n_estimators):
      x_sample, _, y_sample, _ = train_test_split(X_train, y_train, test_size=0.7, random_state=i)
      tree = DecisionTreeClassifier(max_depth=3, random_state=42)
      tree.fit(x_sample, y_sample)
      self.trees.append(tree)

  def predict(self, X_test):
    results = list()
    for tree in self.trees:
      results.append(tree.predict(X_test))

    results = np.array(results)
    most_common = mode(results, axis=0)
    final_predictions = most_common.mode.squeeze()
    return final_predictions