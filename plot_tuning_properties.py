import os
import simulation_parameters
import pylab
import numpy as np
import utils
import matplotlib
from matplotlib import cm

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
pylab.rcParams['lines.markeredgewidth'] = 0


#fn = params['tuning_prop_means_fn']
#d = np.loadtxt(fn)
print 'Computing the tuning properties'
d = utils.set_tuning_prop(params, mode='hexgrid', v_max=params['v_max'])        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)

n_cells = d[:, 0].size
n_rf = params['N_RF_X'] * params['N_RF_Y']
n_units = params['N_RF_X'] * params['N_RF_Y'] * params['N_theta'] * params['N_V']
ms = 3 # markersize for scatterplots

fig = pylab.figure()
pylab.subplots_adjust(hspace=.6)
pylab.subplots_adjust(wspace=.15)
pylab.subplots_adjust(left=.05)
pylab.subplots_adjust(right=.95)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

scale = 10. # scale of the quivers / arrows
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
#    h = ((thetas[i] + np.pi) / (2 * np.pi)) * 360. # theta determines h, h must be [0, 360)
#    l = np.sqrt(u**2 + v**2) / np.sqrt(2 * params['v_max']**2) # lightness [0, 1]
#    s = 1. # saturation
#    assert (0 <= h and h < 360)
#    assert (0 <= l and l <= 1)
#    assert (0 <= s and s <= 1)
#    (r, g, b) = utils.convert_hsl_to_rgb(h, s, l)
#    rgba_colors.append((r, g, b, 1.0))
    ax2.plot(u, v, 'o', color='k', markersize=ms)#, edgecolors=None)

q = ax1.quiver(d[:, 0], d[:, 1], d[:, 2], d[:, 3], \
          angles='xy', scale_units='xy', scale=scale, color=rgba_colors, headwidth=4)
ax1.set_xlabel('$x$', fontsize=16)
ax1.set_ylabel('$y$', fontsize=16)
ax1.set_title('Spatial receptive fields\n n_rf=%d, n_units=%d' % (n_rf, n_units))
ax1.set_xlim((-.05, 1.1))
ax1.set_ylim((-.05, 1.1))
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

print "Saving to ... ", params['tuning_prop_fig_fn']
pylab.savefig(params['tuning_prop_fig_fn'])
pylab.show()
