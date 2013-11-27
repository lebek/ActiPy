import numpy as np
from bokeh.plotting import *
from bokeh.objects import Range1d
from bokeh.glyphs import ImageRGBA
from optical_flow import OpticalFlow
from optical_flow_features import OpticalFlowFeatures
import hashlib
from sys import argv

path = argv[1]
stop_after = int(argv[2])
feature_path = "features_%s_%d.npy" % (hashlib.md5(path).hexdigest(), stop_after)

def factors(n):    
  return set(reduce(list.__add__, 
    ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def closest_factor(p, q):
  factors_of_q = factors(q)
  return min(factors_of_q, key=lambda x:abs(x-p))

def good_cells(width, height, x_guess, y_guess):
    return (closest_factor(x_guess, width), closest_factor(y_guess, height))

def calc_features(path, x_guess, y_guess, stop_after=None):
    hists = []
    magnitudes = []
    good_features = []

    x_cells = None
    y_cells = None
    for pos, flow in enumerate(OpticalFlow(path).farneback()):
        if stop_after and pos > stop_after:
            break

        off = OpticalFlowFeatures(flow)
        if x_cells is None:
            height, width, _ = flow.vectors.shape
            x_cells, y_cells = good_cells(width, height, x_guess, y_guess)
        hist, bin_edges = off.cell_hoof(8, x_cells, y_cells, True)
        hists.append(hist)

        magnitudes.append(off.magnitude(x_cells, y_cells))
        good_features.append(off.good_features())

    dims = flow.get_dims()

    hists = np.swapaxes(np.swapaxes(hists, 0, 1), 1, 2)
    magnitudes = np.swapaxes(np.swapaxes(magnitudes, 0, 1), 1, 2)
    return (x_cells, y_cells), dims, hists, bin_edges, magnitudes, good_features

x_guess = 20
y_guess = 15

try:
    cells,dims,hists,bin_edges,magnitudes,good_features = np.load(feature_path)
except:
    cells,dims,hists,bin_edges,magnitudes,good_features = calc_features(path, x_guess, y_guess, stop_after)
    np.save(feature_path, [cells,dims,hists,bin_edges,magnitudes,good_features])


x_cells, y_cells = cells
print "Using a cell grid of: %dx%d" % cells
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

vid_width, vid_height = dims
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