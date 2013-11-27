import numpy as np
import cv2

class OpticalFlowFeatures:
  """
  Extracts features from an optical flow.
  """

  def __init__(self, flow):
    self.flow = flow

  def _hoof(self, bins, flow_vectors, density=False):
    """ 
    Histogram of (Oriented) Optical Flow
    (http://www.cis.jhu.edu/~rizwanch/papers/ChaudhryCVPR09.pdf)

    I'm not using the bin layout from the paper, since I don't want directional 
    invariance.
    """

    x,y = np.squeeze(np.split(flow_vectors, 2, 2))
    orientations = np.arctan2(x, y)
    magnitudes = np.sqrt(np.square(x) + np.square(y))
    hist, bin_edges = np.histogram(orientations,
                                  bins=bins, 
                                  range=(-np.pi, np.pi), 
                                  weights=magnitudes, 
                                  density=density)

    return hist, bin_edges

  def hoof(self, bins, denisty=False):
    return self._hoof(bins, self.flow.vectors, density)

  def cell_hoof(self, bins, x_cells, y_cells, density=False):
    fvs = self.flow.vectors
    cells = np.vstack([np.split(i, x_cells, axis=1) for i in np.split(fvs, y_cells, axis=0)])

    hists = []
    edges = []
    for c in cells:
      h, e = self._hoof(bins, c, density)
      hists.append(h)
      edges = e

    hists = np.array(hists).reshape(y_cells, x_cells, bins)
    return hists, edges

  def good_features(self):
    return np.array(cv2.goodFeaturesToTrack(self.flow.curr_frame, 5, 0.4, 0)).squeeze(1)

  def magnitude(self, x_cells, y_cells):
    fvs = self.flow.vectors
    x,y = np.squeeze(np.split(fvs, 2, 2))
    magnitudes = np.sqrt(np.square(x) + np.square(y))
    cell_magnitudes = np.vstack([np.split(i, x_cells, axis=1) for i in np.split(magnitudes, y_cells, axis=0)])
    return np.nanmean(np.nanmean(cell_magnitudes, 2), 1).reshape(y_cells, x_cells)
