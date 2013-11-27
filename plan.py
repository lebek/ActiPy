import cv2
import cv_compat

# Utils to help choose a good set of parameters for feature extraction

def factors(n):    
  return set(reduce(list.__add__, 
    ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def closest_factor(p, q):
  factors_of_q = factors(q)
  return min(factors_of_q, key=lambda x:abs(x-p))

def fit_cells(width, height, x_guess, y_guess):
  return (closest_factor(x_guess, width), closest_factor(y_guess, height))

def vid_dims(path):
  vid = cv2.VideoCapture(path)
  im = cv_compat.get_gray_frame(vid)
  h,w = im.shape[:2]
  return w, h

def good_cells(path, x_guess, y_guess):
  width, height = vid_dims(path)
  return fit_cells(width, height, x_guess, y_guess)