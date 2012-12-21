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


network_activity_fn = 'Abstract/Parameters/all_output_activity.dat'
output_folder_fig  = params['figures_folder'] + 'Test/'
#network_activity_fn = 'Abstract/Parameters/all_output_activity_test_minus_training.dat'
#output_folder_fig  = params['figures_folder'] + 'Test_minus_training/'

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
#        act_cnt, act_bins = np.histogram(summed_activities, bins=20)
    #    print 'act_cnt', act_cnt
    #    print 'act_bins', act_bins
        print '%d max activity %.6f\tmin activity %.6f\tmean activitiy %.6f' % (step, max_activities[step], min_activities[step], avg_activities[step])
    print 'Average max activity:', max_activities.mean(), max_activities.std()
    print 'Average min activity:', min_activities.mean(), min_activities.std()
    print 'Average mean activity:', avg_activities.mean(), avg_activities.std()

#    o_max = max_activities.max()
#    o_min = avg_activities.min()
    o_max = 1.0
    o_min = 0.

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
        """
        calculate center-of-mass
        # M : total network activity during t1:t2
        # R : resulting position
        # V : resulting v_predicted
        """
        idx = network_activity_during_stim[t1:t2, :] > 0.
        M = network_activity_during_stim[t1:t2, idx].sum()
#        M = network_activity_during_stim[t1:t2, :].sum()
        R_ = np.zeros((n_cells, 2))
        V_ = np.zeros((n_cells, 2))

        R = np.array([0., 0.])
        V = np.array([0., 0.])
        rgba_colors = []
        for cell in xrange(n_cells):
            activity = network_activity_during_stim[t1:t2, cell].sum()
            if activity > 0.:
                R_[cell, :] = activity / M * tp[cell, 0:2]
                V_[cell, :] = activity / M * tp[cell, 2:]

        R[0] = R_[:, 0].sum()
        R[1] = R_[:, 1].sum()
        V[0] = V_[:, 0].sum()
        V[1] = V_[:, 1].sum()
        print 'debug info'
        print 'R[0]', R[0], R_[:, 0].mean(), R_[:, 0].std()
        print 'R[1]', R[1], R_[:, 1].mean(), R_[:, 1].std()
        print 'V[0]', V[0], V_[:, 0].mean(), V_[:, 0].std()
        print 'V[1]', V[1], V_[:, 1].mean(), V_[:, 1].std()

        data = np.zeros((3+1, 4), dtype=np.double)
        # CMS
        data[0, 0:2] = R
        data[0, 2:] = V
        rgba_colors.append(m.to_rgba(o_max))

        # CMS +- std
#        data[1, 0] = R[0] + R_[:, 0].std()
#        data[1, 1] = R[1] + R_[:, 1].std()
#        data[1, 2] = V[0] + V_[:, 0].std()
#        data[1, 3] = V[1] + V_[:, 1].std()
#        c_std = .5 * (R_[:, 0].std() / R_[:, 0].mean() + R_[:, 1].std() / R_[:, 1].mean())
#        print 'c_std', c_std
#        rgba_colors.append(m.to_rgba(c_std))
#        
#        data[2, 0] = R[0] - R_[:, 0].std()
#        data[2, 1] = R[1] - R_[:, 1].std()
#        data[2, 2] = V[0] - V_[:, 0].std()
#        data[2, 3] = V[1] - V_[:, 1].std()
#        rgba_colors.append(m.to_rgba(c_std))

        data[-1,:] = stim_pos_x, stim_pos_y, mp[2], mp[3]
        rgba_colors.append('r')
        ax.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], \
                  angles='xy', scale_units='xy', scale=scale, color=rgba_colors, headwidth=4, pivot='middle')
        ax.annotate('Stimulus', (stim_pos_x, stim_pos_y), fontsize=12, color='r')

        ax.set_xlim((-0.2, 1.2))
        ax.set_ylim((-0.2, 1.2))
#        output_fn = output_folder_fig + 'network_activity_%03d.png' % (stimulus_number * n_steps + step)
        output_fn = output_folder_fig + 'prediction_%03d.png' % (stimulus_number * n_steps + step)
        print 'output_fig', step, output_fn
    #    print 'o_max o_min', o_max, o_min
        pylab.savefig(output_fn)
        
    print 'Average max activity:', max_activities.mean(), max_activities.std()

#pylab.show()
