import os
import simulation_parameters
import pylab
import numpy as np
import utils
import matplotlib
import sys
from matplotlib import cm

def plot_scatter_with_histograms(x, y):
#    from matplotlib.ticker import NullFormatter

#    nullfmt   = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    fig2 = pylab.figure(figsize=(8,8))
#    ax = fig.add_subplot(111)
    
    axScatter = fig2.add_axes(rect_scatter)
    axScatter.set_xlabel('v_x')
    axScatter.set_ylabel('v_y')
    axHistx = fig2.add_axes(rect_histx)
    axHisty = fig2.add_axes(rect_histy)

    # no labels
#    axHistx.xaxis.set_major_formatter(nullfmt)
#    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.025
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim = ( int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim( (-lim, lim) )
    axScatter.set_ylim( (-lim, lim) )

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )




# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
pylab.rcParams['lines.markeredgewidth'] = 0


print 'Computing the tuning properties'
try:
    cell_type = sys.argv[1]
except:
    cell_type = 'exc'

fn = params['tuning_prop_means_fn']
d = np.loadtxt(fn)
#d = utils.set_tuning_prop(params, mode='hexgrid', cell_type=cell_type)        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)

n_cells = d[:, 0].size
if cell_type == 'exc':
    n_rf = params['N_RF_X'] * params['N_RF_Y']
    n_units = params['N_RF_X'] * params['N_RF_Y'] * params['N_theta'] * params['N_V']
else:
    n_rf = params['N_RF_X_INH'] * params['N_RF_Y_INH']
    n_units = params['N_RF_X_INH'] * params['N_RF_Y_INH'] * params['N_theta_inh'] * params['N_V_INH']

ms = 5 # markersize for scatterplots

fig = pylab.figure(figsize=(16, 8))
pylab.subplots_adjust(hspace=.6)
pylab.subplots_adjust(wspace=.15)
pylab.subplots_adjust(left=.05)
pylab.subplots_adjust(right=.95)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


scale = 4. # scale of the quivers / arrows
# set the colorscale for directions
o_min = 0.
o_max = 360.
norm = matplotlib.mpl.colors.Normalize(vmin=o_min, vmax=o_max)
m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.hsv)#jet)
m.set_array(np.arange(o_min, o_max, 0.01))
rgba_colors = []

#ax4 = fig.add_subplot(224)

thetas = np.zeros(n_cells)
for i in xrange(n_cells):
    x, y, u, v = d[i, :]
    # calculate the color from tuning angle theta
    thetas[i] = np.arctan2(v, u)
    angle = ((thetas[i] + np.pi) / (2 * np.pi)) * 360. # theta determines h, h must be [0, 360)
    rgba_colors.append(m.to_rgba(angle))
    ax2.plot(u, v, 'o', color='k', markersize=ms)#, edgecolors=None)

q = ax1.quiver(d[:, 0], d[:, 1], d[:, 2], d[:, 3], \
          angles='xy', scale_units='xy', scale=scale, color=rgba_colors, headwidth=4, pivot='tail')
ax1.set_xlabel('$x$', fontsize=16)
ax1.set_ylabel('$y$', fontsize=16)
ax1.set_title('Spatial receptive fields for %s cells\n n_rf=%d, n_units=%d' % (cell_type, n_rf, n_units))
ax1.set_xlim((-.05, 1.15))
ax1.set_ylim((-.05, 1.15))
fig.colorbar(m, ax=ax1)

ax2.set_xlabel('$u$', fontsize=16)
ax2.set_ylabel('$v$', fontsize=16)
ax2.set_ylim((d[:, 3].min() * 1.05, d[:, 3].max() * 1.05))
ax2.set_xlim((d[:, 2].min() * 1.05, d[:, 2].max() * 1.05))
ax2.set_title('Receptive fields for speed')




#ax3.set_xlabel('$x$')
#ax3.set_ylabel('$y$')
#ax3.set_title('Preferred directions')
#yticks = ax3.get_yticks()
#xticks = ax3.get_xticks()
#yticks_rescaled = []
#xticks_rescaled = []
#for i in xrange(len(yticks)):
#    yticks_rescaled.append(yticks[i] / scale)
#for i in xrange(len(xticks)):
#    xticks_rescaled.append(xticks[i] / scale)
#ax3.set_yticklabels(yticks_rescaled)
#ax3.set_xticklabels(xticks_rescaled)

output_fn = params['tuning_prop_fig_%s_fn' % cell_type]
print "Saving to ... ", output_fn
pylab.savefig(output_fn)


plot_scatter_with_histograms(d[:, 2], d[:, 3])
output_fn = params['figures_folder'] + 'v_tuning_histogram.png'
print 'Saving to', output_fn
pylab.savefig(output_fn)

pylab.show()
