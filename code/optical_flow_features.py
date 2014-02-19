import numpy as np
import cv2

class OpticalFlowFeatures:
  """
  Extracts features from an optical flow.

  This code is optimized for understanding rather than performance.
  As a result there is much recalculation across the different
  feature extractors.
  """

  def __init__(self, flow):
    self.flow = flow

  def _hoof(self, x, y, bins, density=False):
    """ 
    Histogram of (Oriented) Optical Flow
    (http://www.cis.jhu.edu/~rizwanch/papers/ChaudhryCVPR09.pdf)

    I'm not using the bin layout from the paper, since I don't want directional 
    invariance.

    x = horizontal components of flow
    y = vertical components of flow
    bins = number of HooF bins
    """

    orientations = np.arctan2(x, y)
    magnitudes = np.sqrt(np.square(x) + np.square(y))
    hist, bin_edges = np.histogram(orientations,
                                  bins=bins, 
                                  range=(-np.pi, np.pi), 
                                  weights=magnitudes, 
                                  density=density)

    return hist, bin_edges


  def iterate_cells(self, x_cells, y_cells):
    fvs = self.flow.vectors

    y_len, x_len, _ = fvs.shape
    x_cl = x_len/x_cells
    y_cl = y_len/y_cells

    for i in xrange(x_cells):
      for j in xrange(y_cells):
        x_base = i*x_cl
        y_base = j*y_cl
        x_components = fvs[y_base:y_base+y_cl, x_base:x_base+x_cl, 0]
        y_components = fvs[y_base:y_base+y_cl, x_base:x_base+x_cl, 1]
        yield x_components, y_components, i, j

  def cell_hoof(self, bins, x_cells, y_cells, density=False):
    """
    Calculate HooF across grid cell defined by x_cells and y_cells.
    """

    hists = []
    edges = []

    for x, y, _, _ in self.iterate_cells(x_cells, y_cells):
        h, e = self._hoof(x, y, bins, density)
        hists.append(h)
        edges = e

    hists = np.array(hists).reshape(x_cells, y_cells, bins)
    return hists, edges

  def good_features(self):
    """
    Good Features to Track
    (http://www.ai.mit.edu/courses/6.891/handouts/shi94good.pdf)
    """

    return np.array(cv2.goodFeaturesToTrack(self.flow.curr_frame, 5, 0.4, 0)).squeeze(1)

  def magnitude(self, x_cells, y_cells):
    """
    Calculate flow magnitude across grid cell defined by x_cells and y_cells.
    """

    cell_magnitudes = np.zeros((x_cells, y_cells))
    for x, y, xi, yi in self.iterate_cells(x_cells, y_cells):
      cell_magnitudes[xi,yi] = np.nanmean(np.sqrt(np.square(x) + np.square(y)))

    return cell_magnitudes
