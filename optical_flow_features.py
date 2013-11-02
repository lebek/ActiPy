import numpy as np

class OpticalFlowFeatures:
  """
  Extracts features from an optical flow.
  """

  def __init__(self, flow_vectors):
    self.flow_vectors = flow_vectors

  def hoof(self, bins):
    """ 
    Histogram of (Oriented) Optical Flow
    (http://www.cis.jhu.edu/~rizwanch/papers/ChaudhryCVPR09.pdf)

    I'm not using the bin layout from the paper, since I don't want directional 
    invariance.
    """

    x,y = np.squeeze(np.split(self.flow_vectors, 2, 2))
    orientations = np.arctan2(x, y)
    magnitudes = np.sqrt(np.square(x) + np.square(y))
    hist, bin_edges = np.histogram(orientations,
                                  bins=bins, 
                                  range=(-np.pi, np.pi), 
                                  weights=magnitudes, 
                                  density=True)

    return hist, bin_edges