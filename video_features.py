from optical_flow import OpticalFlow 
from optical_flow_features import OpticalFlowFeatures
import numpy as np

class VideoFeatures:
  """
  Given a video file extracts features.
  """

  def __init__(self, path):
    self.path = path


  def calc_features(self, x_cells, y_cells, stop_after=None):
    hists = []
    magnitudes = []
    good_features = []

    for pos, flow in enumerate(OpticalFlow(self.path).farneback()):
      if stop_after and pos > stop_after:
        break

      off = OpticalFlowFeatures(flow)
      hist, bin_edges = off.cell_hoof(8, x_cells, y_cells, True)
      hists.append(hist)

      magnitudes.append(off.magnitude(x_cells, y_cells))
      good_features.append(off.good_features())

    hists = np.swapaxes(np.swapaxes(hists, 0, 1), 1, 2)
    magnitudes = np.swapaxes(np.swapaxes(magnitudes, 0, 1), 1, 2)
    return hists, bin_edges, magnitudes, good_features