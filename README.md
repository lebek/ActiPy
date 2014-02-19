# ActiPy

ActiPy is a lab for developing and testing action recognition systems.


**actipy.optical_flow.OpticalFlow**
wraps optical flow algorithms provided by OpenCV 1 & 2 under a common API so they can be swapped in and out easily for evaluation.

**actipy.optical_flow.Flow**
represents the optical flow between two adjacent frames and contains useful functions for visualizing the flow.

**actipy.optical_flow_features.OpticalFlowFeatures**
extracts features from optical flow e.g. cellular HOOF and magnitude.

**actipy.video_features.VideoFeatures**
extracts features from sequences of optical flow.

**actipy.plan**
contains utility functions that help to find good parameterisations for the feature extractors, for example `good_cells()` which finds a grid size that divises the video dimensions without remainder.

**actipy.train**
trains a model for action recognition and saves it to file.

**actipy.hist_plot**
outputs a novel visualization of HOOF features.

**actipy.dissertation**
trains a model and evaluates it using the setup used in my dissertation.
