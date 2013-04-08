import pylab
import numpy as np
import simulation_parameters as sp
import matplotlib
from matplotlib import cm
import time
import sys

n_iteration = int(sys.argv[1]) + 1

PS = sp.parameter_storage()
params = PS.load_params()
tp = np.loadtxt(params['tuning_prop_means_fn'])
input_params = np.loadtxt(params['parameters_folder'] + 'input_params.txt')

#network_activity_fn = 'AndersWij/activity_test_minus_training.dat'
#output_folder_fig  = 'AndersWij/test_minus_training_activity/'

#network_activity_fn = 'Abstract/ANNActivity/ann_activity_40iterations_no_rec.dat'
#network_activity_fn = 'Abstract/ANNActivity/ann_activity_40iterations.dat'

#network_activity_fn = 'Abstract/Parameters/all_inputs_scaled.dat'
#output_folder_fig  = 'Abstract/Figures/'

#network_activity_fn = 'Abstract/tmp/output_activity.dat'
#output_folder_fig  = 'Abstract/tmp/'


network_activity_fn = '%sall_inputs_scaled.dat' % params['parameters_folder']
output_folder_fig  = params['figures_folder']

print 'Loading ', network_activity_fn
network_activity = np.loadtxt(network_activity_fn)
#network_activity = np.exp(network_activity)

scale = 1
#n_time_steps = d[:, 0].size
n_time_steps_per_iteration = params['t_sim'] / params['dt_rate']
n_cells = params['n_exc']


o_max = 1.0#n_time_steps_per_iteration
o_min = 0.0#network_activity.min()
#o_max = 2000.0#n_time_steps_per_iteration
#o_min = 0.0#network_activity.min()

for iteration in xrange(n_iteration):
    fig = pylab.figure()
    ax = fig.add_subplot(111)

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

    o_max = summed_activities.max()
    o_min = summed_activities.min()
    norm = matplotlib.mpl.colors.Normalize(vmin=o_min, vmax=o_max)
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.Greys)#jet)
    m.set_array(np.arange(o_min, o_max, 0.01))
    for cell in xrange(n_cells):
        rgba_colors.append(m.to_rgba(summed_activities[cell]))
    print 'plotting stim', iteration, 'max activity', summed_activities.max()
    rgba_colors.append('r')
    ax.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], \
              angles='xy', scale_units='xy', scale=scale, color=rgba_colors, headwidth=4)
    ax.annotate('Stimulus', (mp[0]+.1*mp[2], mp[1]+0.1*mp[1]), fontsize=12, color='r')

    fig.colorbar(m)
    ax.set_xlim((-0.2, 1.2))
    ax.set_ylim((-0.2, 1.2))
    ax.set_title('Integrated activations during full stimulus %d' % iteration)
    output_fn = output_folder_fig + 'network_activity_full_stimulus_%d.png' % (iteration)
#    print 'output_fig', output_fn
#    print 'o_max o_min', o_max, o_min
    pylab.savefig(output_fn)

    print 'Average max activity:', summed_activities.max(), summed_activities.std()

pylab.show()
