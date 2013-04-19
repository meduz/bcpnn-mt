import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import json
from mpl_toolkits.axes_grid1 import ImageGrid
import utils

fn = sys.argv[1]

if len(sys.argv) > 1:
    param_fn = sys.argv[1]
    if os.path.isdir(param_fn):
        param_fn += '/Parameters/simulation_parameters.json'
    print '\nLoading parameters from %s\n' % (param_fn)
    f = file(param_fn, 'r')
    params = json.load(f)
else:
    print '\nPlotting the default parameters given in simulation_parameters.py\n'
    import simulation_parameters
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

# load tuning properties
tp = np.loadtxt(params['tuning_prop_means_fn'])
fn_exc = params['exc_spiketimes_fn_merged'] + '.ras'
fn_inh = params['inh_spiketimes_fn_merged'] + '.ras'


def plot_output_spikes_sorted_in_space(ax, cell_type, shift=0., m='o', c='g', sort_idx=0, ms=2):
    n_cells = params['n_%s' % cell_type]
    fn = params['%s_spiketimes_fn_merged' % cell_type] + '.ras'
    nspikes, spiketimes = utils.get_nspikes(fn, n_cells, get_spiketrains=True)
    sorted_idx = tp[:, sort_idx].argsort()

    if sort_idx == 0:
        ylim = (0, 1)
    else:
        crop = .8
        ylim = (crop * tp[:, sort_idx].min(), crop * tp[:, sort_idx].max())

    ylen = (abs(ylim[0] - ylim[1]))
    print '\n', 'sort_idx', sort_idx, ylim, 
    for i in xrange(n_cells):
        cell = sorted_idx[i]
        if sort_idx == 0:
            y_pos = (tp[cell, sort_idx] % 1.) / ylen * (abs(ylim[0] - ylim[1]))
        else:
            y_pos = (tp[cell, sort_idx]) / ylen * (abs(ylim[0] - ylim[1]))
        ax.plot(spiketimes[cell], y_pos * np.ones(nspikes[cell]), 'o', color='k', markersize=ms)

fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(2, 1))#, aspect=False)

ax1 = grid[0]
ax2 = grid[1]

plot_output_spikes_sorted_in_space(ax1, 'exc', c='k', sort_idx=0, ms=3) 
plot_output_spikes_sorted_in_space(ax2, 'exc', c='k', sort_idx=2, ms=3) 

#ax1.plot(d[:, 0], d[:, 1], 'o', c='k', ms=1)
#ax1.set_ylim((d[:, 1].min() - 100, d[:, 1].max() + 100))
#n, bins = np.histogram(d[:, 0], bins=50)

#ax2.bar(bins[:-1], n)

plt.show()
