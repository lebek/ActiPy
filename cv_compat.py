import cv
import cv2
import numpy as np

def gray_copy(im):
    if is_cv2(im):
      return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    else:
      gray = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_8U, 1)
      cv.CvtColor(im, gray, cv.CV_BGR2GRAY)
      return gray

def color_copy(im):
  if is_cv2(im):
    return cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
  else:
    color = cv.CreateImage(cv.GetSize(im), im.depth, 3)
    cv.CvtColor(im, color, cv.CV_GRAY2BGR)
    return color
  
def line(im, *args, **kwargs):
  if is_cv2(im):
    return cv2.line(im, *args, **kwargs)
  else:
    return cv.Line(im, *args, **kwargs)
  
def circle(im, *args, **kwargs):
  if is_cv2(im):
    return cv2.circle(im, *args, **kwargs)
  else:
    return cv.Circle(im, *args, **kwargs)
  
def is_cv2(im):
  return isinstance(im, np.ndarray)

def get_frame(vid):
  try:
    return vid.read()[1]
  except AttributeError:
    return cv.QueryFrame(vid)

def get_gray_frame(vid):
  frame = get_frame(vid)
  if frame is not None:
    return gray_copy(frame)

def show(label, im):
  if is_cv2(im):
    return cv2.imshow(label, im)
  else:
    return cv.ShowImage(label, im)