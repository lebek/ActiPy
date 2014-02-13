import numpy as np

from sklearn import svm
from sklearn.decomposition import PCA

from video_features import VideoFeatures
from plan import good_cells

class DissertationPredictor:

  def __init__(self, training_feature_path, training_category_path):
    feature_vectors = np.load(training_feature_path)
    categories =  np.load(training_category_path)

    # Calculate PCA features
    self.pca = PCA(n_components=6)
    pca_feature_vectors = self.pca.fit_transform(feature_vectors)

    # Fit SVM
    self.classifier = svm.SVC()
    self.classifier.fit(pca_feature_vectors, categories)

  def predict(self, path):
    x_cells, y_cells = good_cells(path, 3, 3)
    
    avg_hists, bin_edges, avg_magnitudes, variances = VideoFeatures(path).calc_window_features(x_cells, y_cells, None)
    fv = np.concatenate((avg_hists.flatten(), avg_magnitudes.flatten(), variances.flatten()))

    prediction = self.classifier.predict(self.pca.transform(fv))[0]
    return prediction

  def realtime_predict(self, path, window_size):
    x_cells, y_cells = good_cells(path, 3, 3)

    for agg_features in VideoFeatures(path).aggregate_features(x_cells, y_cells, window_size):
      avg_hists, bin_edges, avg_magnitudes, variances = agg_features
      fv = np.concatenate((avg_hists.flatten(), avg_magnitudes.flatten(), variances.flatten()))
      prediction = self.classifier.predict(self.pca.transform(fv))[0]
      print prediction


if __name__ == "__main__":
  from sys import argv
  training_feature_path = argv[1]
  training_category_path = argv[2]
  test_path = argv[3]
  dp = DissertationPredictor(training_feature_path, training_category_path)
  #print dp.predict(test_path)
  dp.realtime_predict(test_path, 60)

