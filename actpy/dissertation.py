import numpy as np
import cv2

from sklearn import svm
from sklearn.decomposition import PCA

from actpy.video_features import VideoFeatures
from actpy.plan import good_cells, vid_dims

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
    
    avg_hists, bin_edges, avg_magnitudes, variances, flow = VideoFeatures(path).calc_window_features(x_cells, y_cells, None)
    fv = np.concatenate((avg_hists.flatten(), avg_magnitudes.flatten(), variances.flatten()))

    prediction = self.classifier.predict(self.pca.transform(fv))[0]
    return prediction

  def realtime_predict(self, path, window_size, start, end, output):
    x_cells, y_cells = good_cells(path, 3, 3)

    v = cv2.VideoWriter()
    dims = vid_dims(path)
    codec = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    v.open(output, codec, 15, dims, 1)
    #v.open("out.mp4", cv.CV_FOURCC('F', 'M', 'P', '4'), 30, vid_dims(path), True)

    for pos, agg_features in enumerate(VideoFeatures(path).aggregate_features(x_cells, y_cells, window_size)):
      if pos < start:
        continue
      if pos > end:
        break

      avg_hists, bin_edges, avg_magnitudes, variances, flow = agg_features
      fv = np.concatenate((avg_hists.flatten(), avg_magnitudes.flatten(), variances.flatten()))
      prediction = self.classifier.predict(self.pca.transform(fv))[0]
      vis = flow.show(flow=False, text=prediction, display=False)
      v.write(vis)

    v.release()

if __name__ == "__main__":
  from sys import argv
  training_feature_path = argv[1]
  training_category_path = argv[2]
  test_path = argv[3]
  window_size = int(argv[4])
  start = int(argv[5])
  end = int(argv[6])
  output = argv[7]
  dp = DissertationPredictor(training_feature_path, training_category_path)
  #print dp.predict(test_path)
  dp.realtime_predict(test_path, window_size, start, end, output)

