import cv
import cv2
import numpy as np
import actipy.cv_compat as cv_compat

class Flow:
  """
  Represents the optical flow between two frames.
  """

  def __init__(self, vectors, curr_frame, prev_frame, curr_frame_color):
    self.vectors = vectors
    self.curr_frame = curr_frame
    self.prev_frame = prev_frame
    self.curr_frame_color = curr_frame_color

  @staticmethod
  def draw_flow(vis, im, flow, step=16):
    mult = 4

    w, h = cv_compat.get_dims(im)

    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T

    # create line endpoints
    lines = np.vstack([x,y,x+(fx*mult),y+(fy*mult)]).T.reshape(-1,2,2)
    lines = np.int32(lines)

    for pos,((x1,y1),(x2,y2)) in enumerate(lines):
      cv_compat.line(vis,(x1,y1),(x2,y2), (255,255,0), 1, cv.CV_AA)
      cv_compat.circle(vis,(x1,y1),1, (255,255,0), 1, cv.CV_AA)

    return vis

  # I don't belong here
  def draw_good_features(self, vis):
    corners = cv2.goodFeaturesToTrack(self.curr_frame, 5, 0.4, 0)
    for corner in corners:
      x, y = [int(i) for i in corner[0]]
      cv_compat.circle(vis,(x,y),1,(0,0,255), 3, cv.CV_AA)

  ## EXPERIMENTAL ##
  def draw_hands(self, vis):
    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([255,173,127],np.uint8)

    imageYCrCb = cv2.cvtColor(vis,cv2.COLOR_BGR2YCR_CB)

    skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

    contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(contours):
      area = cv2.contourArea(c)
      if area > 400:
        cv2.drawContours(vis, contours, i, (0, 255, 0), 3)


  def show(self, title="Optical Flow", flow=True, good_features=False, text=None,
          display=True):
    if title == "Prediction":
        vis = self.curr_frame_color.copy()
    else:
        vis = self.curr_frame
    if flow:
      self.draw_flow(vis, self.curr_frame, self.vectors)
    if good_features:
      self.draw_good_features(vis)
    if text:
      w, h = cv_compat.get_dims(vis)
      cv_compat.putText(vis, text, (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
    if display:
      cv_compat.show(title, vis)
      if cv.WaitKey(10) == 27:
        return

    return vis

class OpticalFlow:
  """
  Given a video file extracts the optical flow.
  """

  def __init__(self, path):
    self.path = path

  def _iter_frames(self, vid):
    vid_length = cv_compat.get_vid_length(vid)
    prev_frame = cv_compat.get_gray_frame(vid)
    curr_frame_color = cv_compat.get_frame(vid)
    curr_frame = cv_compat.gray_copy(curr_frame_color)

    for i in xrange(vid_length-2):
      yield (prev_frame, curr_frame, curr_frame_color)
      prev_frame = curr_frame
      curr_frame_color = cv_compat.get_frame(vid)
      curr_frame = cv_compat.gray_copy(curr_frame_color)

  def lucas_kanade(self):
    vid = cv.CaptureFromFile(self.path)

    first_frame = cv_compat.get_gray_frame(vid)
    velx = cv.CreateImage(cv.GetSize(first_frame), cv.IPL_DEPTH_32F, 1)
    vely = cv.CreateImage(cv.GetSize(first_frame), cv.IPL_DEPTH_32F, 1)

    for prev_frame, curr_frame, curr_frame_color in self._iter_frames(vid):
      cv.CalcOpticalFlowLK(prev_frame, curr_frame, (15,15), velx, vely)
      flow = np.dstack((np.asarray(cv.GetMat(velx)), np.asarray(cv.GetMat(vely))))
      yield Flow(flow, curr_frame, prev_frame, curr_frame_color)

  def horn_schunck(self):
    vid = cv.CaptureFromFile(self.path)

    term_crit = (cv.CV_TERMCRIT_ITER, 100, 0)

    first_frame = cv_compat.get_gray_frame(vid)
    velx = cv.CreateImage(cv.GetSize(first_frame), cv.IPL_DEPTH_32F, 1)
    vely = cv.CreateImage(cv.GetSize(first_frame), cv.IPL_DEPTH_32F, 1)

    for prev_frame, curr_frame, curr_frame_color in self._iter_frames(vid):
      cv.CalcOpticalFlowHS(prev_frame, curr_frame, False, velx, vely, 0.001, term_crit)
      flow = np.dstack((np.asarray(cv.GetMat(velx)), np.asarray(cv.GetMat(vely))))
      yield Flow(flow, curr_frame, prev_frame, curr_frame_color)

  def farneback(self):
    vid = cv2.VideoCapture(self.path)
    flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN & cv2.OPTFLOW_USE_INITIAL_FLOW
    for pos, (prev_frame, curr_frame, curr_frame_color) in enumerate(self._iter_frames(vid)):
      flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame,
                                          pyr_scale=0.5, levels=3,
                                          winsize=20, iterations=5,
                                          poly_n=7, poly_sigma=1.5,
                                          flags=flags)
      yield Flow(flow, curr_frame, prev_frame, curr_frame_color)


if __name__ == "__main__":
  from sys import argv

  path = argv[1]
  flow_func = argv[2]
  for flow in getattr(OpticalFlow(path), flow_func)():
    flow.show(flow=False)
