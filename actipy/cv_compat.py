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

def putText(im, *args, **kwargs):
  if is_cv2(im):
    return cv2.putText(im, *args, **kwargs)
  else:
    # this doesn't work
    return cv.PutText(im, *args, **kwargs)

def get_dims(im):
  if is_cv2(im):
    h,w = im.shape[:2]
  else:
    h = im.height
    w = im.width

  return w, h
  
def is_cv2(o):
  return (isinstance(o, np.ndarray) or type(o).__name__ == 'VideoCapture')

def get_frame(vid):
  try:
    return vid.read()[1]
  except AttributeError:
    return cv.QueryFrame(vid)

def get_vid_length(vid):
  if is_cv2(vid):
    return int(vid.get(cv.CV_CAP_PROP_FRAME_COUNT))
  else:
    return int(cv.GetCaptureProperty(vid, cv.CV_CAP_PROP_FRAME_COUNT))

def open_vid(path):
  return cv2.VideoCapture(path)

def get_gray_frame(vid):
  frame = get_frame(vid)
  if frame is not None:
    return gray_copy(frame)

def show(label, im):
  if is_cv2(im):
    return cv2.imshow(label, im)
  else:
    return cv.ShowImage(label, im)