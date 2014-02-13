from sys import argv
from re import match
from collections import defaultdict
import os
import hashlib

import numpy as np
from sklearn.decomposition import PCA
from bokeh.plotting import *

from video_features import VideoFeatures
from plan import good_cells

path = argv[1]
output = argv[2]

TRAINING_LEN = 40 # per class
TESTING_LEN = 5 # per class

dataset = defaultdict(list)

for f in os.listdir(path):
  if f.endswith(".mp4"):
    category = match("[a-z]+", f).group()
    dataset[category].append(os.path.join(path, f))

def calc_feature_vector(path):
  x_cells, y_cells = good_cells(path, 3, 3)
  avg_hists, bin_edges, avg_magnitudes, variances = VideoFeatures(path).calc_window_features(x_cells, y_cells, None)
  fv = np.concatenate((avg_hists.flatten(), avg_magnitudes.flatten(), variances.flatten()))
  #fv = np.concatenate((avg_hists.flatten(),))
  #fv = np.concatenate((avg_magnitudes.flatten(), variances.flatten()))

  return fv

def calc_feature_vectors(dataset):
  feature_vectors = []
  categories = []

  for category, paths in dataset.items():
    for path in paths[:TRAINING_LEN]:
      feature_vectors.append(calc_feature_vector(path))
      categories.append(category)

  return feature_vectors

def get_categories(dataset):
  categories = []

  for category, paths in dataset.items():
    for path in paths[:TRAINING_LEN]:
      categories.append(category)

  return categories

feature_path = "feature_vectors_%s.npy" % (hashlib.md5(path).hexdigest(),)
feature_vectors = np.load(feature_path)
try:
  feature_vectors = np.load(feature_path)
except:
  feature_vectors = calc_feature_vectors(dataset)
  np.save(feature_path, feature_vectors)

category_path = "category_%s.npy" % (hashlib.md5(path).hexdigest(),)
try:
  categories = np.load(category_path)
except:
  categories = get_categories(dataset)
  np.save(category_path, categories)

### This PCA just for graphing purposes ###
pca = PCA(n_components=2)
pca_feature_vectors = pca.fit_transform(feature_vectors)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
"#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# assume same number of examples in each class
category_length = len(pca_feature_vectors)/len(dataset)

output_file(output)

hold()
for i in range(len(dataset)):
  base = i*category_length
  scatter(pca_feature_vectors[base:base+category_length].T[0], pca_feature_vectors[base:base+category_length].T[1], 
    fill_color=colors[i], legend=dataset.keys()[i])
show()

# Classification PCA
pca = PCA(n_components=6)
pca_feature_vectors = pca.fit_transform(feature_vectors)

# Construct SVM
from sklearn import svm
classifier = svm.SVC(probability=True)
categories = np.asarray([[i]*TRAINING_LEN for i in range(len(dataset))]).flatten()
classifier.fit(pca_feature_vectors, categories)

# Predict categories
for category, paths in dataset.items():
    for path in paths[TRAINING_LEN:TRAINING_LEN+TESTING_LEN]:
      fv = calc_feature_vector(path)
      predictions = classifier.predict_proba(pca.transform(fv))[0]
      print [(dataset.keys()[pos], p) for pos,p in enumerate(predictions)]
      #prediction = dataset.keys()[classifier.predict(pca.transform(fv))[0]]
      #print "%s is predicted to be a <%s> sample." % (path, prediction)