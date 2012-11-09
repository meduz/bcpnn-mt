import pylab
import numpy as np
import simulation_parameters as sp
import matplotlib
from matplotlib import cm

import time

PS = sp.parameter_storage()
params = PS.load_params()
tp = np.loadtxt(params['tuning_prop_means_fn'])
input_params = np.loadtxt(params['parameters_folder'] + 'input_params.txt')

#network_activity_fn = 'AndersWij/activity_test.dat'
#output_folder_fig  = 'AndersWij/test_activity/'

#network_activity_fn = 'AndersWij/activity_training.dat'
#output_folder_fig  = 'AndersWij/training_activity/'

network_activity_fn = 'AndersWij/activity_test_minus_training.dat'
output_folder_fig  = 'AndersWij/test_minus_training_activity/'

#network_activity_fn = 'Abstract/ANNActivity/ann_activity_40iterations_no_rec.dat'
#network_activity_fn = 'Abstract/ANNActivity/ann_activity_40iterations.dat'
#output_folder_fig  = params['figures_folder']

print 'Loading ', network_activity_fn
network_activity = np.loadtxt(network_activity_fn)
network_activity = np.exp(network_activity)

scale = 3
#n_time_steps = d[:, 0].size
n_time_steps_per_iteration = 300
n_cells = params['n_exc']

n_iteration = 40

o_max = 2000.0#n_time_steps_per_iteration
o_min = 0.0#network_activity.min()

for iteration in xrange(n_iteration):
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    norm = matplotlib.mpl.colors.Normalize(vmin=o_min, vmax=o_max)
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.Greys)#jet)
    m.set_array(np.arange(o_min, o_max, 0.01))
    fig.colorbar(m)

    print 'plotting stim', iteration
    t1 = iteration * n_time_steps_per_iteration
    t2 = (iteration + 1) * n_time_steps_per_iteration
    mp = input_params[iteration, :]
    data = np.zeros((n_cells+1, 4), dtype=np.double)
    data[:n_cells,:] = tp
    data[-1,:] = mp

    rgba_colors = []
    summed_activities = np.zeros(n_cells)
    for cell in xrange(n_cells):
        activity = network_activity[t1:t2, cell].sum()
        summed_activities[cell] = activity
#        o_max = max(o_max, activity)
#        o_min = min(o_min, activity)
#        print 'max activity cell %d' % cell, activity.max()
        rgba_colors.append(m.to_rgba(activity))
    print 'max activity', summed_activities.max()
    rgba_colors.append('r')
    ax.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], \
              angles='xy', scale_units='xy', scale=scale, color=rgba_colors, headwidth=4)
    ax.annotate('Stimulus', (mp[0]+.1*mp[2], mp[1]+0.1*mp[1]), fontsize=12, color='r')

    ax.set_xlim((-0.2, 1.2))
    ax.set_ylim((-0.2, 1.2))
    output_fn = output_folder_fig + 'network_activity_full_stimulus_%d.png' % (iteration)
#    print 'output_fig', output_fn
#    print 'o_max o_min', o_max, o_min
    pylab.savefig(output_fn)

print 'Average max activity:', summed_activities.max(), summed_activities.std()
#pylab.show()
