import cv
import cv2
import numpy as np
import cv_compat


class Flow:
  """
  Represents the optical flow between two frames.
  """

  def __init__(self, vectors, curr_frame, prev_frame):
    self.vectors = vectors
    self.curr_frame = curr_frame
    self.prev_frame = prev_frame

  @staticmethod
  def draw_flow(im, flow, step=16):
    if cv_compat.is_cv2(im):
      h,w = im.shape[:2]
    else:
      h = im.height
      w = im.width
      
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T

    # create line endpoints
    lines = np.vstack([x,y,x+(fx*4),y+(fy*4)]).T.reshape(-1,2,2)
    lines = np.int32(lines)

    # create image and draw
    vis = cv_compat.color_copy(im)
    for (x1,y1),(x2,y2) in lines:
      cv_compat.line(vis,(x1,y1),(x2,y2),(0,255,0), 1, cv.CV_AA)
      cv_compat.circle(vis,(x1,y1),1,(0,255,0), -1, cv.CV_AA)
    return vis

  def show(self):
    cv_compat.show("Optical Flow", self.draw_flow(self.curr_frame, self.vectors))
    if cv.WaitKey(10) == 27:
      return

class OpticalFlow:
  """
  Given a video file extracts the optical flow.
  """

  def __init__(self, path):
    self.path = path

  def _iter_frames(self, vid):
    prev_frame = cv_compat.get_gray_frame(vid)
    curr_frame = cv_compat.get_gray_frame(vid)
    while curr_frame is not None:
      yield (prev_frame, curr_frame)
      prev_frame = curr_frame
      curr_frame = cv_compat.get_gray_frame(vid)

  def lucas_kanade(self):
    vid = cv.CaptureFromFile(self.path)

    first_frame = cv_compat.get_gray_frame(vid)
    velx = cv.CreateImage(cv.GetSize(first_frame), cv.IPL_DEPTH_32F, 1)
    vely = cv.CreateImage(cv.GetSize(first_frame), cv.IPL_DEPTH_32F, 1)

    for prev_frame, curr_frame in self._iter_frames(vid):
      cv.CalcOpticalFlowLK(prev_frame, curr_frame, (15,15), velx, vely)
      flow = np.dstack((np.asarray(cv.GetMat(velx)), np.asarray(cv.GetMat(vely))))
      yield Flow(flow, curr_frame, prev_frame)

  def horn_schunck(self):
    vid = cv.CaptureFromFile(self.path)

    term_crit = (cv.CV_TERMCRIT_ITER, 100, 0)

    first_frame = cv_compat.get_gray_frame(vid)
    velx = cv.CreateImage(cv.GetSize(first_frame), cv.IPL_DEPTH_32F, 1)
    vely = cv.CreateImage(cv.GetSize(first_frame), cv.IPL_DEPTH_32F, 1)

    for prev_frame, curr_frame in self._iter_frames(vid):
      cv.CalcOpticalFlowHS(prev_frame, curr_frame, False, velx, vely, 0.001, term_crit)
      flow = np.dstack((np.asarray(cv.GetMat(velx)), np.asarray(cv.GetMat(vely))))
      yield Flow(flow, curr_frame, prev_frame)

  def farneback(self):
    vid = cv2.VideoCapture(self.path)

    for prev_frame, curr_frame in self._iter_frames(vid):
      flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, 
                                          pyr_scale=0.5, levels=3, 
                                          winsize=15, iterations=10, 
                                          poly_n=7, poly_sigma=1.5, flags=0)
      yield Flow(flow, curr_frame, prev_frame)


if __name__ == "__main__":
  from sys import argv

  path = argv[1]
  flow_func = argv[2]
  for flow in getattr(OpticalFlow(path), flow_func)():
    flow.show()
