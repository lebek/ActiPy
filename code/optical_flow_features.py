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

  def cell_hoof(self, bins, x_cells, y_cells, density=False):
    """
    Calculate HooF across grid cell defined by x_cells and y_cells.
    """
    
    fvs = self.flow.vectors
    cells = np.vstack([np.split(i, x_cells, axis=1) for i in np.split(fvs, y_cells, axis=0)])

    cells_x = cells[:,:,:,0]
    cells_y = cells[:,:,:,1]

    hists = []
    edges = []
    for i in xrange(len(cells)):
      h, e = self._hoof(cells_x[i], cells_y[i], bins, density)
      hists.append(h)
      edges = e

    hists = np.array(hists).reshape(y_cells, x_cells, bins)
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

    x = self.flow.vectors[:,:,0]
    y = self.flow.vectors[:,:,1]
    magnitudes = np.sqrt(np.square(x) + np.square(y))
    cell_magnitudes = np.vstack([np.split(i, x_cells, axis=1) for i in np.split(magnitudes, y_cells, axis=0)])
    return np.nanmean(np.nanmean(cell_magnitudes, 2), 1).reshape(y_cells, x_cells)
