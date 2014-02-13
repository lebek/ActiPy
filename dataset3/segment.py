from subprocess import call
from glob import glob
from os import path

SEGMENT_LENGTH = 4 # seconds

import cv2
def get_vid_length(vid_path):
  vid = cv2.VideoCapture(vid_path)
  frame_count = vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
  fps = vid.get(cv2.cv.CV_CAP_PROP_FPS)
  return int(frame_count/fps)

for unsegmented_path in glob("labelled/unsegmented/*.mp4"):
  file_name, file_ext = path.split(unsegmented_path)[1].split(".")

  vid_length = get_vid_length(unsegmented_path)

  i = 0
  ret = 0
  while (i+1)*SEGMENT_LENGTH < vid_length:
    segmented_path = "segments/%s_%d.%s" % (file_name, i, file_ext)
    
    ret = call(['ffmpeg', '-i', unsegmented_path, '-ss', str(i*SEGMENT_LENGTH), '-t', str(SEGMENT_LENGTH), '-filter:v', 'scale=320:-1', segmented_path])
    i += 1
