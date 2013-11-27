import numpy as np
from bokeh.plotting import *
from bokeh.objects import Range1d
from bokeh.glyphs import ImageRGBA
from video_features import VideoFeatures
from plan import good_cells, vid_dims
import hashlib
from sys import argv

path = argv[1]
stop_after = int(argv[2])
feature_path = "features_%s_%d.npy" % (hashlib.md5(path).hexdigest(), stop_after)

vid_width, vid_height = vid_dims(path)
x_cells, y_cells = good_cells(path, 20, 15)
print "Using a cell grid of: %dx%d" % (x_cells, y_cells)

try:
    hists,bin_edges,magnitudes,good_features = np.load(feature_path)
except:
    hists,bin_edges,magnitudes,good_features = VideoFeatures(path).calc_features(x_cells, y_cells, stop_after)
    np.save(feature_path, [hists,bin_edges,magnitudes,good_features])

avgs = np.nanmean(hists, 2)
avg_magnitudes = np.nanmean(magnitudes, 2)
avg_magnitudes = (avg_magnitudes-np.nanmin(avg_magnitudes))/(np.nanmax(avg_magnitudes)-np.nanmin(avg_magnitudes))
#good_features = np.vstack(good_features)

#variances = np.var(magnitudes, 2)
variances = np.sum(np.nanvar(hists, 2), 2) # Sum of the variances over time in each bin in each cell
#variances = np.nanvar(hists, 2)
variances = (variances-np.nanmin(variances))/(np.nanmax(variances)-np.nanmin(variances))

def make_color(x, y):
    # RGB
    #return "#%02x%02x%02x" % (np.floor(x*250), np.floor(np.logspace(0,np.log10(250),251)[np.floor(y*250)]), 150)
    return "#%02x%02x%02x" % (np.floor(x*250), np.floor(y*250), 150)

bins = len(bin_edges)-1
angle = 2.0 * np.pi / bins
#angles = np.arange(bins)*angle
angles = np.roll(bin_edges[:-1], bins/4)

max_sum = np.max(np.sum(avgs, 2))

vis_width = 1200
vis_height = int(vid_height*(float(vis_width)/vid_width))

padding = 20

vid_space_x = np.linspace(0, vid_width, x_cells)
vid_space_y = np.linspace(0, vid_height, y_cells)

output_file("hist_plot.html", title="HOOF plots")

hold()

line(angles, angles, alpha=0, width=vis_width, height=vis_height, title="", tools="", x_axis_type=None, y_axis_type=None)

plot = curplot()
plot.outline_line_color = None
plot.x_range = Range1d(start=-padding, end=vid_width+padding)
plot.y_range = Range1d(start=-padding, end=vid_height+padding)

# for pos, gfs in enumerate(good_features):
#     loc = float(pos)/len(good_features)
#     scatter(gfs.T[0], vid_height-gfs.T[1], radius=10, fill_color=None, line_color=make_color(loc, 0), alpha=0.05)

for x in np.arange(0,x_cells):
    for y in np.arange(0,y_cells):
        #annular_wedge(np.ones(bins)*x, np.ones(bins)*y, 0., np.random.rand(bins)*0.3, angles, angles+angle)
        #magnitudes = np.random.rand(bins)*0.25
        #import pdb; pdb.set_trace()
        magnitudes = avgs[y_cells-y-1,x]*16
        x_start = np.ones(bins)*vid_space_x[x]
        y_start = np.ones(bins)*vid_space_y[y]
        x_ends = x_start+(magnitudes*np.sin(-angles+(1.5*angle)))
        y_ends = y_start+(magnitudes*np.cos(-angles+(1.5*angle)))
        #x+(magnitudes[0]*np.sin(angle)), y+(magnitudes[0]*np.cos(angle))
        #for i in range(bins):
        #  line([x_start[i], x_ends[i]], [y_start[i], y_ends[i]])
        #arc(x_start, y_start, magnitudes, angles, angles+angle)
        line_colors = make_color(avg_magnitudes[y_cells-y-1,x], variances[y_cells-y-1,x])
        annular_wedge(x_start, y_start, 0., magnitudes, angles, angles+angle, 
            fill_color=None, line_color=line_colors)

xgrid().grid_line_color = None
ygrid().grid_line_color = None
show()