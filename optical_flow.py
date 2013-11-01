import cv
import cv2
import numpy as np


class OpticalFlow:

  def _gray_copy(self, im):
    if self._isCV2(im):
      return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    else:
      gray = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_8U, 1)
      cv.CvtColor(im, gray, cv.CV_BGR2GRAY)
      return gray

  def _color_copy(self, im):
    if self._isCV2(im):
      return cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    else:
      color = cv.CreateImage(cv.GetSize(im), im.depth, 3)
      cv.CvtColor(im, color, cv.CV_GRAY2BGR)
      return color
    
  def _line(self, im, *args, **kwargs):
    if self._isCV2(im):
      return cv2.line(im, *args, **kwargs)
    else:
      return cv.Line(im, *args, **kwargs)
    
  def _circle(self, im, *args, **kwargs):
    if self._isCV2(im):
      return cv2.circle(im, *args, **kwargs)
    else:
      return cv.Circle(im, *args, **kwargs)
    
  def _isCV2(self, im):
    return isinstance(im, np.ndarray)

  def _get_frame(self, vid):
    try:
      return vid.read()[1]
    except AttributeError:
      return cv.QueryFrame(vid)

  def _get_gray_frame(self, vid):
    frame = self._get_frame(vid)
    if frame is not None:
      return self._gray_copy(frame)

  def _show(self, label, im):
    if self._isCV2(im):
      return cv2.imshow(label, im)
    else:
      return cv.ShowImage(label, im)

  def iter_frames(self, vid):
    prev_frame = self._get_gray_frame(vid)
    curr_frame = self._get_gray_frame(vid)
    while curr_frame is not None:
      yield (prev_frame, curr_frame)
      prev_frame = curr_frame
      curr_frame = self._get_gray_frame(vid)

  def draw_flow(self, im, flow, step=16):
    if self._isCV2(im):
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
    vis = self._color_copy(im)
    for (x1,y1),(x2,y2) in lines:
      self._line(vis,(x1,y1),(x2,y2),(0,255,0), 1, cv.CV_AA)
      self._circle(vis,(x1,y1),1,(0,255,0), -1, cv.CV_AA)
    return vis

  def lucas_kanade(self, path):
    vid = cv.CaptureFromFile(path)

    first_frame = self._get_gray_frame(vid)
    velx = cv.CreateImage(cv.GetSize(first_frame), cv.IPL_DEPTH_32F, 1)
    vely = cv.CreateImage(cv.GetSize(first_frame), cv.IPL_DEPTH_32F, 1)

    for prev_frame, curr_frame in self.iter_frames(vid):
      cv.CalcOpticalFlowLK(prev_frame, curr_frame, (15,15), velx, vely)
      flow = np.dstack((np.asarray(cv.GetMat(velx)), np.asarray(cv.GetMat(vely))))
      yield (curr_frame, flow)

  def horn_schunck(self, path):
    vid = cv.CaptureFromFile(path)

    term_crit = (cv.CV_TERMCRIT_ITER, 100, 0)

    first_frame = self._get_gray_frame(vid)
    velx = cv.CreateImage(cv.GetSize(first_frame), cv.IPL_DEPTH_32F, 1)
    vely = cv.CreateImage(cv.GetSize(first_frame), cv.IPL_DEPTH_32F, 1)

    for prev_frame, curr_frame in self.iter_frames(vid):
      cv.CalcOpticalFlowHS(prev_frame, curr_frame, False, velx, vely, 0.001, term_crit)
      flow = np.dstack((np.asarray(cv.GetMat(velx)), np.asarray(cv.GetMat(vely))))
      yield (curr_frame, flow)

  def farneback(self, path):
    vid = cv2.VideoCapture(path)

    for prev_frame, curr_frame in self.iter_frames(vid):
      flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, 
                                          pyr_scale=0.5, levels=3, 
                                          winsize=15, iterations=10, 
                                          poly_n=7, poly_sigma=1.5, flags=0)
      yield (curr_frame, flow)

  def show_flow(self, path, flow_func):
    for frame, flow in flow_func(path):
      self._show("Optical Flow", self.draw_flow(frame, flow))
      if cv.WaitKey(10) == 27:
        break


if __name__ == "__main__":
  from sys import argv

  path = argv[1]
  flow_func = argv[2]

  of = OpticalFlow()
  of.show_flow(argv[1], getattr(of, flow_func))
