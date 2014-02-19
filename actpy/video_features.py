from actpy.optical_flow import OpticalFlow 
from actpy.optical_flow_features import OpticalFlowFeatures
import numpy as np
import actpy.plan as plan
from actpy.progress_bar import ProgressBar

import hashlib

class VideoFeatures:
  """
  Given a video file extracts features.
  """

  def __init__(self, path):
    self.path = path

    # video length minus one because the first 
    # frame doesn't have an optical flow
    self.progress = ProgressBar(plan.vid_length(path) - 1)

  def loadFlows(self):
    flowPath = "models/flow_%s.npy" % (hashlib.md5(self.path).hexdigest(),)
    return np.load(flowPath)

  def saveFlows(self, data):
    flowPath = "models/flow_%s.npy" % (hashlib.md5(self.path).hexdigest(),)
    return np.save(flowPath, data)

  def features(self, x_cells, y_cells):
    """
    Extracts HooF and flow magnitude from every frame of a video.
    """

    try:
      generator = (i for i in self.loadFlows())
    except:
      generator = OpticalFlow(self.path).farneback()

    flows = []
    print "Extracting features for %s..." % (self.path,)
    for pos, flow in enumerate(generator):
      flows.append(flow)

      off = OpticalFlowFeatures(flow)

      hist, bin_edges = off.cell_hoof(8, x_cells, y_cells, True)

      # +1 because this iteration complete
      # +1 because 0-based array indexing
      self.progress.animate(pos+2)

      yield hist, bin_edges, off.magnitude(x_cells, y_cells), flow

    print

    self.saveFlows(flows)


  def aggregate_features(self, x_cells, y_cells, window_size=None):
    hists = []
    magnitudes = []

    for pos, fv in enumerate(self.features(x_cells, y_cells)):
      hist, bin_edges, magnitude, flow = fv
      if window_size and len(hists) == window_size:
        yield self.summarise_features(hists, magnitudes, bin_edges) + (flow,)
        hists.pop(0)
        magnitudes.pop(0)

      hists.append(hist)
      magnitudes.append(magnitude)

    if not window_size:
      yield self.summarise_features(hists, magnitudes, bin_edges) + (flow,)

  def summarise_features(self, hists, magnitudes, bin_edges):
    #hists = np.swapaxes(np.swapaxes(hists, 0, 1), 1, 2)
    #magnitudes = np.swapaxes(np.swapaxes(magnitudes, 0, 1), 1, 2)

    # Now calculate mean of HooF
    #import pdb; pdb.set_trace()
    avg_hists = np.nanmean(hists, 0)

    # Mean of flow magnitudes
    avg_magnitudes = np.nanmean(magnitudes, 0)
    avg_magnitudes = (avg_magnitudes-np.nanmin(avg_magnitudes))/(
      np.nanmax(avg_magnitudes)-np.nanmin(avg_magnitudes))

    # Sum of the variances over time in each bin in each cell
    variances = np.sum(np.nanvar(hists, 0), 2)
    variances = (variances-np.nanmin(variances))/(
      np.nanmax(variances)-np.nanmin(variances))

    return avg_hists, bin_edges, avg_magnitudes, variances

  def calc_window_features(self, x_cells, y_cells, end=None):
    """
    Legacy
    """
    for fv in self.aggregate_features(x_cells, y_cells, end):
      return fv
    