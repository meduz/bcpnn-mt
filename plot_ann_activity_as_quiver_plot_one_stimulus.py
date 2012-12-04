import sys
import pylab
import numpy as np
import simulation_parameters as sp
import matplotlib
from matplotlib import cm
import time
import utils

PS = sp.parameter_storage()
params = PS.load_params()
tp = np.loadtxt(params['tuning_prop_means_fn'])
input_params = np.loadtxt(params['parameters_folder'] + 'input_params.txt')

try:
    stimuli = range(0, int(sys.argv[1]))
except:
    stimuli = [0]
    print 'Stimuli number:', stimuli
scale = 1


#network_activity_fn = 'Abstract/Parameters/all_output_activity_test_minus_training.dat'
network_activity_fn = 'Abstract/Parameters/all_output_activity.dat'
output_folder_fig  = params['figures_folder'] + 'Test/'

#network_activity_fn = 'Abstract/Parameters/all_inputs_scaled.dat'
#output_folder_fig  = params['figures_folder']

#iteration = 0
#network_activity_fn = 'Abstract/ANNActivity/output_activity_%d.dat' % iteration
#output_folder_fig  = params['figures_folder'] + 'Test/'

print 'Loading ', network_activity_fn
network_activity = np.loadtxt(network_activity_fn)
#network_activity = np.exp(network_activity)
n_cells = params['n_exc']
#n_time_steps = d[:, 0].size
n_time_steps_per_stimulus = 60
n_time_steps_for_averaging = 5
n_steps_offset = 0
n_steps = n_time_steps_per_stimulus / n_time_steps_for_averaging

for stimulus_number in stimuli:
    mp = input_params[stimulus_number, :]
    t0 = stimulus_number * n_time_steps_per_stimulus
    t1 = (stimulus_number + 1) * n_time_steps_per_stimulus
    network_activity_during_stim = network_activity[t0:t1, :]


    max_activities = np.zeros(n_steps)
    min_activities = np.zeros(n_steps)
    avg_activities = np.zeros(n_steps)
    # for different colorscales
    for step in xrange(n_steps_offset, n_steps):
        t1 = step * n_time_steps_for_averaging
        t2 = (step + 1) * n_time_steps_for_averaging
        summed_activities = np.zeros(n_cells)
        for cell in xrange(n_cells):
            activity = network_activity_during_stim[t1:t2, cell].sum()
            summed_activities[cell] = activity
        max_activities[step] = summed_activities.max()
        min_activities[step] = summed_activities.min()
        avg_activities[step] = summed_activities.mean()
        act_cnt, act_bins = np.histogram(summed_activities, bins=20)
    #    print 'act_cnt', act_cnt
    #    print 'act_bins', act_bins
        print '%d max activity %.6f\tmin activity %.6f\tmean activitiy %.6f' % (step, max_activities[step], min_activities[step], avg_activities[step])
    print 'Average max activity:', max_activities.mean(), max_activities.std()
    print 'Average min activity:', min_activities.mean(), min_activities.std()
    print 'Average mean activity:', avg_activities.mean(), avg_activities.std()

    o_max = max_activities.max()
    o_min = avg_activities.min()

    for step in xrange(n_steps_offset, n_steps):
    #    print 'Step', step
        fig = pylab.figure()
        ax = fig.add_subplot(111)

    #     if seperate colorscales:
    #    o_max = max_activities[step]
    #    o_min = min_activities[step]
        norm = matplotlib.mpl.colors.Normalize(vmin=o_min, vmax=o_max)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.Greys)#jet)
        m.set_array(np.arange(o_min, o_max, 0.01))

        fig.colorbar(m)

        t_ = (float(step) / n_steps) * params['t_sim'] / params['t_stimulus']
        stim_pos_x = mp[0] + mp[2] * t_
        stim_pos_y = mp[1] + mp[3] * t_

        t1 = step * n_time_steps_for_averaging
        t2 = (step + 1) * n_time_steps_for_averaging
#        cells_closest_to_stim, x_dist = utils.sort_gids_by_distance_to_stimulus(tp, mp)
#        cells_closest_to_stim.tolist().reverse()
        data = np.zeros((n_cells+1, 4), dtype=np.double)
        data[:n_cells,:] = tp
#        data[:n_cells,:] = tp[cells_closest_to_stim, :]

        data[-1,:] = stim_pos_x, stim_pos_y, mp[2], mp[3]

        rgba_colors = []
        for cell in xrange(n_cells):
#        for cell in cells_closest_to_stim:
            activity = network_activity_during_stim[t1:t2, cell].sum()
            rgba_colors.append(m.to_rgba(activity))
        rgba_colors.append('r')
        ax.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], \
                  angles='xy', scale_units='xy', scale=scale, color=rgba_colors, headwidth=4, pivot='middle')
        ax.annotate('Stimulus', (stim_pos_x, stim_pos_y), fontsize=12, color='r')

        ax.set_xlim((-0.2, 1.2))
        ax.set_ylim((-0.2, 1.2))
#        output_fn = output_folder_fig + 'network_activity_%02d_%02d.png' % (stimulus_number, step)
        output_fn = output_folder_fig + 'network_activity_%03d.png' % (stimulus_number * n_steps + step)
        print 'output_fig', step, output_fn
    #    print 'o_max o_min', o_max, o_min
        pylab.savefig(output_fn)
        
    print 'Average max activity:', max_activities.mean(), max_activities.std()

#pylab.show()
