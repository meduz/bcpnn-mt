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
stimulus_number = 11
mp = input_params[stimulus_number, :]
scale = 3

#network_activity_fn = 'AndersWij/activity_test.dat'
#output_folder_fig  = 'AndersWij/test_activity/'

#network_activity_fn = 'AndersWij/activity_test_minus_training.dat'
network_activity_fn = 'AndersWij/activity_test_minus_training_exp.dat'
#network_activity_fn = 'AndersWij/activity_test_minus_training_exp_clipped.dat'
output_folder_fig  = 'AndersWij/test_minus_training_activity/'

#network_activity_fn = 'AndersWij/activity_training.dat'
#output_folder_fig  = 'AndersWij/training_activity/'

#network_activity_fn = 'Abstract/ANNActivity/ann_activity_40iterations.dat'
#output_folder_fig  = params['figures_folder']

print 'Loading ', network_activity_fn
network_activity = np.loadtxt(network_activity_fn)
network_activity = np.exp(network_activity)
n_cells = params['n_exc']
#n_time_steps = d[:, 0].size
n_time_steps_per_stimulus = 300
n_time_steps_for_averaging = 5
n_steps_offset = 2
n_steps = n_time_steps_per_stimulus / n_time_steps_for_averaging
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
o_min = max_activities.mean()
#o_min = min_activities.mean()

for step in xrange(n_steps_offset, n_steps):
    print 'Step', step
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
#    print 'stim_pos', step, t_, stim_pos_x, stim_pos_y

    t1 = step * n_time_steps_for_averaging
    t2 = (step + 1) * n_time_steps_for_averaging
    data = np.zeros((n_cells+1, 4), dtype=np.double)
    data[:n_cells,:] = tp
    data[-1,:] = stim_pos_x, stim_pos_y, mp[2], mp[3]

    rgba_colors = []
    for cell in xrange(n_cells):
        activity = network_activity_during_stim[t1:t2, cell].sum()
#        summed_activities[cell] = activity
#        o_max = max(o_max, activity)
#        o_min = min(o_min, activity)
#        print 'max activity cell %d' % cell, activity.max()
        rgba_colors.append(m.to_rgba(activity))
#    print 'max activity %.3f\tmin activity %.3f' % (summed_activities.max(), summed_activities.min())
    rgba_colors.append('r')
    ax.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], \
              angles='xy', scale_units='xy', scale=scale, color=rgba_colors, headwidth=4, pivot='tip')
    ax.annotate('Stimulus', (mp[0]+.1*mp[2], mp[1]+0.1*mp[1]), fontsize=12, color='r')

    ax.set_xlim((-0.2, 1.2))
    ax.set_ylim((-0.2, 1.2))
    output_fn = output_folder_fig + 'network_activity_%d_%d.png' % (stimulus_number, step)
#    print 'output_fig', output_fn
#    print 'o_max o_min', o_max, o_min
    pylab.savefig(output_fn)
    
print 'Average max activity:', max_activities.mean(), max_activities.std()

#pylab.show()
